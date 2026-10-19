# テンソルと自動微分の設計

## 対象範囲

`autograd`はdefine-by-runのfloat32グラフです。TensorはShape、Strides、DType、Deviceと共有ストレージを持ちます。CPUはスライス、CUDAはデバイスバッファ、Vulkan/DirectMLの推論は`gpu.Buffer`を使います。学習はCPU/CUDAが対象です。

Reshape、Transpose、Permute、Slice、Expandはviewを作ります。連続配置が必要な演算ではContiguousで実体化し、対応するGPU演算はstride / offsetを直接読みます。CUDAの値・勾配・optimizer stateはデバイス上に保ち、ToHostやstateの書出しで明示的に読み戻します。

ストレージはlease、グラフの辺は依存数で管理します。Close済みの先祖も計算に必要な間は内部で保持し、共有ノードは最後の依存が終わってから回収します。RetainData / RetainGradは履歴を保たず割当を保持できます。異なるdeviceと未対応dtypeの組合せは拒否します。

## グラフと演算

記録中の演算は親とvector-Jacobian積を保存し、Backwardが逆順に伝播します。末端の勾配は加算し、中間の勾配はRetainGradしない限り破棄します。通常のBackwardは履歴を消費し、RetainGraph指定時は再利用できます。保存した統計・im2col・target・indexなどはその履歴が終わるまで保持します。

ExecutionContextは記録・BF16・アテンションの方針を分離します。モデルの全stateをforward前にBindModuleし、入力をそのコンテキストで作ります。viewとDetachも同じコンテキストを保ち、異なる明示的コンテキストの混在は拒否します。NoGradとAutocastはコンテキスト全体に作用し、入れ子とpanicから復帰します。

精度とアテンションはforward時に保存し、後の設定変更やretainしたBackwardでも同じ方針を使います。既定はグローバル設定に依存しないFP32 / autoです。複数演算からなるforward全体は一括のスナップショットではないため、設定はforwardの合間に変えます。コンテキストは共有モデルの変更やストレージの寿命を同期しません。

推論結果もReleaseGraph用の親を保ちますが、導関数は持ちません。CopyFromとeagerなoptimizer更新はストレージのversionを進め、Backwardが勾配変更前に検証します。Dataへの直接書込と外部CUDA書込・再生は追跡しません。

CPU/CUDAの微分可能な公開演算はopレジストリでdispatchします。CPUはAVX2/FMAの8×8マイクロカーネル付きSGEMMを実行時に選び、`GOGRAD_SGEMM_8X8=0`なら4×8、それ以外の環境は純Go経路を使います。要素演算とreductionは複数コアへ分けます。

CUDAの通常バッチGEMMはcuBLASのstrided-batched経路を使い、ブロードキャストする重みの勾配はバッチ軸を1 GEMMへ平坦化します。不規則なバッチはループで処理します。rank 6までのview・broadcast・reductionと、要素演算・正規化・損失・dropoutなどはNVRTCカーネルで実装します。

CUDAのdevice・暗黙のstream・captureはOSスレッド単位です。cuBLASハンドルとクリップ用プランも同じ単位で持ちます。NewCUDAContextでスレッドを固定し、NewCUDAContextOnは以前のdeviceをClose時に復元します。SetCUDAで記録したdevice / streamをWithCUDAが一時適用します。

RunBranchesは指定された分岐を別streamへenqueueし、呼出し側streamへeventで合流します。分岐は自動では作りません。Captureは明示的streamで行い、同じスレッドの入れ子を拒否します。再生は記録したdeviceを選び、グラフ用割当はGraph.Closeまで予約します。

CUDAプールはdevice・stream・サイズクラスで分け、256 B、4 KiB、64 KiB、大きい領域は1 MiB刻みへ丸めます。既定の上限は2 GiBで、古いキャッシュから退避します。ReleasePoolは使用中・グラフ予約中の割当を解放しません。Bufferがdeviceを記録するため、コピー・memset・freeは所有するdeviceを選びます。

CUDA attentionはバッチGEMMと融合したmask / scale / softmaxを使います。tiled経路はonline softmaxで二次のscoreバッファを避け、Backwardで再計算します。加算maskに対応しますが、現在は実体化経路より遅いため、autoはscore scratchの見積もりが1 GiB以上の場合だけ選びます。

BF16はGEMMオペランドと、viewや対応するpointwise演算を通るシャドウに使います。GEMMExの累積、master weight、勾配、optimizerのモーメントはFP32です。CPU・Conv2d・tiled attentionはFP32のままです。CUDA Encoderはdropoutが0ならbias / GELU / residualを融合します。

Vulkan/DirectMLは`gpu.Backend`を通す推論専用経路です。対応演算はCPUへ戻さず実行し、勾配記録や学習を要求された場合は拒否します。Vulkanは埋め込みSPIR-V、DirectMLはコンパイルした演算子を使い、バッファと演算子のキャッシュを制限します。詳細は[GPUバックエンド](GPU_BACKENDS.md)を参照してください。

形状と演算の主な規則は次のとおりです。

| 演算 | 規則 |
| --- | --- |
| broadcast / MatMul | 軸を右から揃え、同じ長さか片方1。MatMulの末尾2軸が行列、前の軸がバッチ |
| reduction | 明示した軸を出力から除く。スカラーshapeは`[]` |
| Concat / Slice | 0始まりの軸、半開区間 |
| Embedding | 繰返すindexへ勾配をscatter加算。padding行は除外できる |
| Conv1d | 入力`[batch,time,channels]`、重み`[out,in,kernel]`。奇数kernelのsame-padding、dilation、cross-correlation |
| LayerNorm / GroupNorm | 最後の軸 / サンプル・groupごとのtimeとchannelsを正規化 |
| MaskedLoss | 非NaNのtargetフレームを使い、有効フレーム×channelsで割る。0件なら損失0 |
| Attention | scaleしたQKᵀ + 加算mask → softmax → Vとの積 |
| Dropout | seedとelement indexのhashでmaskを再現。PyTorchの乱数とは異なる |

MultiheadAttentionはPyTorchのpacked投影レイアウト、TransformerEncoderLayerはbatch-firstのpre-norm構成です。

## モジュールと学習

Moduleは名前付きパラメータ・バッファ・子を登録します。名前は安定したドット区切りで、同じ割当の別名は1回だけ保存します。学習パラメータはグラフの末端、running統計はoptimizer対象外のバッファです。Trainは子へ再帰伝播します。

AdamWはdecoupled decay、FP32モーメント、bias correctionを使います。グローバルノルムのクリップはStep前に行います。OneCycleはPyTorchの2段cosineで、initial=max/25、final=initial/10000、pct_startがwarmupの境界です。

汎用損失は既存の要素演算とreductionを合成します。重み付きmeanの分母はMSE/L1が重みの和、BCEが要素数です。GradAccumulatorは損失を1/groupsにスケールし、更新とクリップはgroup完了後に行います。

safetensorsは8-byte little-endianのヘッダ長、JSON記述子、連続payloadからなります。読込みは名前・shape・範囲・重複・ヘッダ上限を検証し、保存は同一ディレクトリの一時ファイル、sync、renameで行います。OpenSafeTensorFileはindexを先に検証してテンソル単位で読み、LoadSafeTensorsStreamはコピー前に全名前とshapeを検証します。旧nnのJSON形式も維持します。

TrainingLoopは更新単位のZeroGrad、optimizer / scheduleのStep、報告、停止時保存を担当します。readerのEOFではepochを進め、reader / Stepのエラーでは保存しません。context終了時は未完了groupの勾配を捨て、Samplerをgroup先頭へ戻してSaveします。モデルとoptimizerは最後の完了更新を保ちます。forwardが変更するstateは呼出し側で戻す必要があります。

## Conv2d

CPU/CUDAともNCHWのcross-correlationで、重みは`[out,in/groups,kH,kW]`です。stride・padding・dilationはheight / width別に指定し、grouped / depthwise、出力multiplier、任意のbiasに対応します。

1バッチ・1groupずつdepth-majorのタイルへ展開してGEMMを行います。出力位置は最大512、im2colと出力で4 MiBを目標にし、完全な展開行列は保持しません。Backwardはim2colを再計算し、重み勾配を累積、入力勾配をcol2imでscatterします。CUDAはcuBLASとatomic加算を使います。推論のscratchは返却前に解放し、BF16シャドウとは独立したFP32経路です。

## NCHW pooling

MaxPool2d、AvgPool2d、AdaptiveAvgPool2dは窓の幾何を共有し、実体化・出力確保前に検証します。maxの選択indexは記録中だけ保存し、retainした履歴でも保持します。average / adaptiveのBackwardは幾何と除数を使います。

CPUのBackwardはN*Cの面を分割し、面内の重複を直列加算して競合を避けます。CUDAは窓ごとに並列化し、Backwardはatomic scatter加算を使います。FP32でcapture / 再生に対応します。

ceil modeは右端・下端のpaddingだけで始まる窓を除きます。averageは明示paddingとその外側を除数で区別し、adaptiveはfloor / ceilのbinが重なる場合も勾配を加算します。

## BatchNormと可変state

rank 2〜6のチャネル以外を正規化し、outer / channel / innerでchannels-first / lastを扱います。BatchNormLayerはaffine、running mean / variance、バッチカウンタをModuleへ登録します。NoGradはrunning stateの更新を止めません。

学習の出力は偏りのある分散、running varianceは不偏分散を使います。モーメントはfloat64累積、各forwardの保存統計はFP32です。runningバッファはBackwardの依存ではないため、後の更新やCloseで以前のグラフは変わりません。入力とaffineはversion検証の対象です。

CUDAはチャネルごとのreductionと要素ごとのforward / VJPを使います。カウンタ更新を別カーネルで先行させ、累積momentumとの競合を防ぎます。state更新はデバイス上で完了し、capture / 再生に対応します。num_batches_trackedもF32で、整数精度は`2^24`までです。

## ホストのデータ処理

dataはDataset、Sampler、collateをテンソル生成と分離します。同期Loaderはindexをpeekし、取得・collate・キャンセル確認の成功後にだけカーソルを確定します。失敗はデータを飛ばさず再試行でき、callbackのEOFはepochの終端と区別します。

IndexSamplerはPCG、順列、位置、epochを保存し、復元前に構成とpayloadを検証します。Loader stateはバッチ構成を含みますが、データ内容・split・独自変換の同一性はアプリ側metadataで管理します。read-only Datasetは並行安全なら共有でき、変更可能なSampler / Loaderは逐次使用します。

Stack / PadSequencesは所有するホスト値、長さ、bool maskを返し、アップロードは学習側で行います。IDXはuint8で保持し、sample取得時に変換します。fingerprintは復号した画像・ラベルの内容から計算します。

TransformDatasetはindex / seedをcallbackへ渡し、コピーはcallbackに任せます。組込み反転とノイズはsampleごとのPCGを使い、worker数・読込順・epochに依存しません。Subsetで包む順序により、元indexかsubset内の位置かが決まります。

並列Loaderは上限付きでindexを先読みし、sample取得だけをworkerへ渡します。結果はSampler順へ並べ、collateはNextの呼出し元で実行します。CUDAスレッド管理は学習側に残ります。

キャンセル・エラー・Close・NextEpochでは古いreaderを止めてjoinします。Stateは先読みを捨て、配達済み位置だけを保存します。Workers / Prefetchはcheckpointの同一性に含めず、数を変えて再開できます。Getの副作用や乱数はアプリ側の責任です。

数値CSVは矩形性と有限値を検証します。Standardizerはtraining splitだけでfloat64の統計を求め、定数特徴量のscaleを1にします。統計・分割・データfingerprintをcheckpointへ保存し、検証・再開時にfitし直しません。

WAVの復号・符号化はdata、フレーム化・window・augmentationはアプリ側です。音声例のノイズは`(seed, window index)`から導くため、worker / prefetch / shuffle / 再開位置が変わっても同じ入力を再現します。

学習例のCPU/CUDA実機確認は、いずれもseed 7、batch 64です。以下は動作と再開の確認であり、データセット全体の品質やoptimizer間の優劣を示すものではありません。

| 例 | データ・更新数 | 検証結果 |
| --- | --- | --- |
| MNIST / AdamW | 学習先頭2,000件、テスト先頭1,000件、100更新 | accuracy 86.4%、loss 0.455661 |
| MNIST / SGD | 同じデータと更新数。momentum .9、Nesterov、OneCycleピーク .1 | accuracy 85.2%、loss 0.479619 |
| MNIST / SGD、biasのdecayなし | 同上、`-no-decay-suffix bias` | accuracy 85.2%、loss 0.479656 |
| 数値回帰 | 4特徴量、4,000行、うち800検証、200更新 | validation RMSE 12.164137 |
| 音声denoiser | 生成した8秒・16 kHzのトーン、window 2,000、うち400検証、300更新 | 検証SNR 4.8665→15.0490 dB |

全例で推論の再読込みの差は0でした。MNISTはCPUを40更新で止めて100まで再開し、CUDAは連続実行しました。AdamWはreader 4 / prefetch 3、40更新後に2 / 2へ変更したCPU、4 / 4のCUDAでも同じ結果でした。SGDもCPUのworker設定を変えて再開しています。数値回帰は100、音声は150更新で止め、reader設定を変えて再開しても中断なしと同一の重みになりました。

## OptimizerとSGDのstate

Optimizerはeagerな更新・リセット・学習率・寿命の共通APIです。CheckpointOptimizerはアルゴリズムのタグ付きstateを扱い、training-1のoptimizer_kindがなければ旧AdamWとみなします。live objectへのコピー前に種類と構成を検証します。

SGDはPyTorchのcoupled decay、momentum、dampening、Nesterov、ascentに合わせます。初回momentumはdampeningを適用せず、初期化はパラメータごとに遅延します。勾配なしは更新を飛ばし、0勾配は更新対象です。CUDAは更新後に初期化フラグを立て、stateの読書きの競合を避けます。

CUDA graphではデバイス上のステップと初期化フラグを保持し、ZeroGradは安定した勾配バッファを0にします。保存とeager復帰ではステップを同期し、eager更新はストレージのversionを進めてBF16シャドウを無効にします。

パラメータグループは名前付きの分割と学習率スケール・decayを持ちます。OneCycleは基準学習率だけを変えます。stateはindexではなく順序付きの名前を保存し、別グループへの読込みを拒否します。旧グループなしstateはスケールなしの単一グループだけが受け入れます。CUDA graphには記録時のスケールを固定します。

AdamWのステップカウンタはoptimizer全体で1つです。PyTorchは勾配を受けたパラメータごとに進めるため、勾配なしの期間がある場合はbias correctionが異なり得ます。モーメント更新のスキップは同じですが、この差は残っています。

## 計測結果

RTX 3060 Tiと12 CPUスレッドを使い、1セッションでgograd / PyTorchを対にして3回測った学習ステップ全体の中央値です。forward・backward・勾配クリップ・AdamWを含みます。単位はms。Transformerは4層、model 256、heads 4、FFN 1024、batch 16、系列長256。Acousticはhidden 384、batch 24、400 framesです。BF16はPyTorchのautocastとgogradのFP32値＋BF16シャドウを使い、適用範囲は異なります。CPU時間は測定中に変動しており、安定したスループット値ではありません。

| モデルと実行環境 | CPU FP32 | CUDA FP32 eager | CUDA FP32 graph | CUDA BF16 eager | CUDA BF16 graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acoustic PyTorch | 2505.9 | 58.5 | 53.7 | 47.1 | 42.6 |
| Acoustic gograd | 3876.4 | 82.3 | 82.0 | 63.8 | 64.1 |
| Transformer PyTorch | 1291.6 | 25.9 | 23.9 | 13.5 | 12.0 |
| Transformer gograd | 2276.1 | 27.2 | 27.8 | 22.4 | 23.2 |

CPU時間のPyTorch比はTransformer 1.76倍、Acoustic 1.55倍です。目標の2.5倍 / 1.5倍に対し、前者は範囲内、後者はわずかに超えました。graphはこのGEMMの多い処理ではほぼ短縮せず、BF16 eagerはgograd FP32より速いもののPyTorch BF16には届きません。TransformerのBF16 eagerはPyTorchの1.66倍です。

CPU SGEMMの8×8 / 4×8の中央値はAcoustic 3783.4 / 4157.8 ms、Transformer 2085.0 / 2242.9 msでした。変更後のCPUプロファイルではSGEMMが約61% / 33%を占め、続いてindexingと要素勾配の処理でした。

batch 1、heads 4、head幅64のattentionをCUDA eventで3回測った中央値と、使用中バッファのピーク増分です。

| 系列長 | 実体化ステップ | tiledステップ | 実体化のピーク | tiledのピーク |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.366 ms | 0.766 ms | 3.50 MiB | 1.01 MiB |
| 1024 | 1.601 ms | 9.606 ms | 50.00 MiB | 4.03 MiB |
| 4096 | 18.916 ms | 127.889 ms | 776.00 MiB | 16.12 MiB |

tiledは3系列長で出力・勾配が実体化経路と一致しました。メモリは削減しますが、現在のBackwardは実体化経路より遅いままです。Vulkan / DirectMLの推論計測は[別文書](GPU_BACKEND_PERFORMANCE.md)にあります。

## 可変長系列のCUDAプール

更新ごとにグラフを解放し、パディング後の長さが64通りある学習を再現しました。使用中は約11 MBのまま、サイズクラスと上限によりキャッシュが縮小しました。

| 64ステップ後 | 使用中bytes | キャッシュbytes | 予約bytes |
| --- | ---: | ---: | ---: |
| 旧・正確なサイズごとの無制限cache | 10,927,812 | 6,154,440,208 | 6,165,368,020 |
| サイズクラス＋上限 | 10,949,888 | 762,475,264 | 773,425,152 |
| 上限256 MiBでの別実行 | 10,949,888 | 265,028,352 | 275,978,240 |

200ステップでもキャッシュは同水準でした。cudaMalloc失敗時はキャッシュを解放して1回再試行します。ホスト転送のスライスはDLL呼出し中に保持します。

固定shapeのAcoustic学習を最後の3対で比較した中央値は77.3→77.8 ms、差0.7%でした。それ以前の対はdevice負荷で両バイナリとも遅くなったため、この値を採用しました。
