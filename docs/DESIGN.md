# テンソルと自動微分の設計

## 対象範囲

`autograd` はdefine-by-runのfloat32自動微分エンジンです。Tensorは `Shape`、`Strides`、`DType`、`Device` と共有ストレージを持ち、CPUはGoスライス、CUDAはデバイスバッファ、Vulkan / DirectMLの推論は `gpu.Buffer` をバックエンドとします（学習はCPU / CUDAのみ対象）。

`Reshape`、`Transpose`、`Permute`、`Slice`、`Expand` はviewを生成します。連続メモリ配置が必要な演算では `Contiguous` で実体化し、対応するGPU演算はstride / offsetを直接参照します。CUDA上の値・勾配・optimizer stateはデバイス上に留まり、`ToHost` やstate書き出し時に明示的にホストへ転送します。

ストレージはlease、グラフの辺は依存数で参照カウント管理します。`Close` 済みの先祖も計算に必要な間は内部で保持され、共有ノードは最後の依存が終了してから回収されます。`RetainData` / `RetainGrad` を使うと履歴を保持せずに割当だけを維持できます。

## グラフと演算

記録中の演算は親ノードとvector-Jacobian積を保持し、`Backward` で逆順に伝播します。末端の勾配は累積加算され、中間勾配は `RetainGrad` 指定がない限り破棄されます。通常の `Backward` は履歴を消費し、`RetainGraph` 指定時のみ再利用可能です。正規化統計・im2col・target・indexなどの補助データも履歴終了まで保持されます。

`ExecutionContext` は記録・BF16・アテンション方針をスコープ分離します。モデルの全stateをforward前に `BindModule` し、入力を同じコンテキストで生成します。`NoGrad` と `Autocast` はコンテキスト全体に作用し、入れ子とpanicから復帰します。精度とアテンション方針はforward時に保存されるため、後の設定変更やretainした `Backward` でも一貫性が保たれます（複数演算からなるforward全体が1スナップショットになるわけではないため、設定変更はforwardの合間に行います）。

推論結果も `ReleaseGraph` 用の親リンクを持ちますが、導関数は保持しません。`CopyFrom` とeagerなoptimizer更新はストレージのversionを進め、`Backward` が勾配書き込み前に検証します（`Data` スライスへの直接書き込みや外部CUDA書き込み・再生は追跡対象外）。

CPU / CUDAの微分可能演算はopレジストリ経由でdispatchされます。
- **CPU**：AVX2/FMAの4×16マイクロカーネル付きSGEMMを実行時に選択します（`GOGRAD_SGEMM_4X16=0` で4×8、非対応環境は純Go経路）。要素演算とreductionはマルチコアへ並列化します。
- **CUDA**：通常のバッチGEMMはcuBLASのstrided-batched経路を使い、ブロードキャストする重みの勾配はバッチ軸を1 GEMMへ平坦化します（不規則バッチはループ処理）。rank 6までのview・broadcast・reductionおよび要素演算・正規化・損失・dropoutなどはNVRTCカーネルで実装しています。

CUDAのデバイス・暗黙のstream・capture状態、およびcuBLASハンドル・クリップ用プランはOSスレッド単位で管理されます。`NewCUDAContext` でスレッドを固定し、`NewCUDAContextOn` は `Close` 時に以前のデバイスへ復元します。`SetCUDA` で設定したデバイス / streamは `WithCUDA` で一時適用されます。

`RunBranches` は指定された分岐を別streamへenqueueし、呼び出し側streamへeventで合流させます（自動分岐は行いません）。`Capture` は明示的stream上で実行し、同一スレッドでの入れ子は拒否します。再生は記録時と同じデバイスを選択し、グラフ用割当は `Graph.Close` まで予約されます。

CUDAメモリプールはデバイス・stream・サイズクラスで分離され、256 B、4 KiB、64 KiB、それ以上の領域は1 MiB刻みに丸められます。既定の上限は2 GiBで、超過時は古いキャッシュから退避します（`ReleasePool` でも使用中・グラフ予約中の割当は解放しません）。`Buffer` が所有デバイスを記録しているため、コピー・memset・freeはバッファ自身のデバイスを選択して実行します。

CUDA attentionはバッチGEMMと融合したmask / scale / softmaxを使用します。tiled経路はonline softmaxで二次サイズのscoreバッファを回避して `Backward` で再計算を行いますが、現状は実体化（materialized）経路より遅いため、`auto` ではscore scratchの見積もりが1 GiB以上の場合のみ選択されます。

BF16はGEMMオペランドと、viewや対応するpointwise演算を通過するシャドウに使用し、GEMMExの累積・master weight・勾配・optimizerモーメントはFP32で保持します。CPU、Conv2d、tiled attentionはFP32で動作します。CUDA Encoderはdropoutが0の場合にbias / GELU / residualを融合します。

Vulkan / DirectMLは `gpu.Backend` を介した推論専用経路です。対応演算はCPUへ戻さずGPU上で完結し、勾配記録や学習要求は拒否します。Vulkanは埋め込みSPIR-V、DirectMLはコンパイル済み演算子を使用し、バッファと演算子キャッシュに上限を設けています。詳細は [GPUバックエンド](GPU_BACKENDS.md) を参照してください。

形状と演算の主な規則は以下の通りです。

| 演算 | 規則 |
| --- | --- |
| broadcast / MatMul | 軸を右から揃え、同長または片方が1。MatMulは末尾2軸が行列、それ以前がバッチ軸 |
| reduction | 指定軸を出力から除外（スカラーshapeは `[]`） |
| Concat / Slice | 0始まりの軸、半開区間 |
| Embedding | 重複するindexへ勾配をscatter加算（padding行は除外可能） |
| Conv1d | 入力 `[batch,time,channels]`、重み `[out,in,kernel]`。奇数kernelのsame-padding、dilation、cross-correlation |
| LayerNorm / GroupNorm | 最終軸 / サンプル・groupごとのtimeとchannelsを正規化 |
| MaskedLoss | 非NaNのtargetフレームのみ使用し、`有効フレーム × channels` で除算（0件時は損失0） |
| Attention | scale済み `QKᵀ + 加算mask` → softmax → `V` との積 |
| Dropout | seedとelement indexのhashでmaskを再現（PyTorchの乱数列とは異なる） |

`MultiheadAttention` はPyTorchのpacked投影レイアウト、`TransformerEncoderLayer` はbatch-firstのpre-norm構成を採用しています。

## モジュールと学習

`Module` は名前付きパラメータ・バッファ・子モジュールをドット区切りの階層名で管理し、同一割当の別名は1回のみ保存します。学習パラメータはグラフ末端、running統計はoptimizer対象外のバッファとして登録され、`Train` は子モジュールへ再帰伝播します。

- **AdamW**：decoupled weight decay、FP32モーメント、bias correctionを使用します。グローバルノルムクリップは `Step` 前に実行します。
- **OneCycle**：PyTorch準拠の2段cosineスケジュール（`initial = max/25`、`final = initial/10000`、`pct_start` がwarmup境界）。
- **損失と勾配蓄積**：汎用損失は既存の要素演算とreductionの組み合わせで構成します。`GradAccumulator` は損失を `1/groups` にスケールし、クリップとパラメータ更新はgroup完了後に行います。
- **safetensors**：8-byte little-endianのヘッダ長、JSON記述子、連続payloadで構成されます。読み込み時に名前・shape・範囲・重複・ヘッダ上限を検証し、保存は同一ディレクトリ内の一時ファイル書き込み・sync・renameでアトミックに行います。`OpenSafeTensorFile` はindex検証後にテンソル単位で遅延読み込みし、`LoadSafeTensorsStream` はコピー前に全名前とshapeを検証します。
- **TrainingLoop**：更新単位の `ZeroGrad`、optimizer / scheduleの `Step`、進捗報告、停止時保存を担います。context終了時は未完了groupの勾配を破棄し、Samplerをgroup先頭へ戻して `Save` します。

## Conv2d

CPU / CUDAともにNCHWのcross-correlation（重み `[out,in/groups,kH,kW]`）で、stride・padding・dilation（height / width個別指定）、grouped / depthwise、出力multiplier、任意biasに対応します。

1バッチ・1groupずつdepth-majorタイルへ展開してGEMMを実行します（出力位置最大512、im2colと出力の合計メモリ目標4 MiB）。完全な展開行列を保持しないため、Backward時はim2colを再計算して重み勾配を累積し、入力勾配はcol2imでscatter加算します（CUDAはcuBLASとatomic加算を使用）。推論scratchは返却前に解放され、BF16シャドウとは独立したFP32経路で動作します。

## NCHW pooling

`MaxPool2d`、`AvgPool2d`、`AdaptiveAvgPool2d` は共通のウィンドウ幾何計算を持ち、実体化や出力メモリ確保の前にパラメータ検証を行います。maxの選択indexは勾配記録中（およびretainされた履歴）のみ保持します。

CPUのBackwardは `N*C` 面単位で並列化し、面内の重複加算を直列化して競合を回避します。CUDAはウィンドウ単位で並列化し、Backwardにはatomic scatter加算を使用します（FP32でcapture / 再生に対応）。

## BatchNormと可変state

rank 2〜6のテンソルを `outer / channel / inner` に分解してchannels-first / lastの両方を処理します。`BatchNormLayer` はaffineパラメータ、running mean / variance、バッチカウンタを `Module` に登録します（`NoGrad` 中もrunning stateの更新は停止しません）。

学習時の出力には偏りのある分散、running varianceには不偏分散を使用し、モーメントはfloat64で累積して保存統計はFP32で保持します。runningバッファはBackwardの依存対象外であるため、後続の更新や `Close` が既存グラフに影響することはありません。

CUDAではチャネルごとのreductionと要素ごとのforward / VJPカーネルを実行します。累積momentumとの競合を防ぐためカウンタ更新を別カーネルで先行させており、state更新までデバイス上で完結するためcapture / 再生に対応します（`num_batches_tracked` もFP32保持のため整数精度は `2^24` まで）。

## ホストのデータ処理

`data` パッケージはDataset、Sampler、collateをテンソル生成から分離しています。同期 `Loader` はindexをpeekし、取得・collate・キャンセル確認がすべて成功した後にカーソルを確定するため、失敗時はデータをスキップせず再試行できます（callbackのEOFはepoch終端と区別されます）。

- `IndexSampler` はPCG状態・順列・位置・epochを保存し、復元時に構成とpayloadを検証します。read-onlyなDatasetは並行共有可能ですが、状態を持つSampler / Loaderは逐次使用を前提とします。
- `TransformDataset` はindexとseedをcallbackへ渡し、コピー判断はcallback側に委ねます。組み込みの反転・ノイズ変換はサンプルごとのPCGを使うため、worker数・読み込み順・epochに依存せず再現性が保たれます（`Subset` でラップする順序により、元indexとsubset内位置のどちらを使うかを選択可能）。
- 並列 `Loader` は上限付きでindexを先読みし、`Dataset.Get` のみをworkerへ分散して結果はSampler順へ整列します。collateおよびCUDAスレッド管理は `Next` の呼び出し元スレッドに残ります。キャンセル・エラー・`Close`・`NextEpoch` 時は古いreaderを停止・joinし、`State` は先読み分を破棄して配達済み位置のみを保存します（`Workers` / `Prefetch` はcheckpoint同一性に含まれないため変更して再開可能）。

以下は各サンプルの動作確認および中断・再開テストの結果です（すべてseed 7、batch 64、CPU / CUDA両環境で推論再読み込み誤差0を確認）。

| 例 | データ・更新数 | 検証結果 |
| --- | --- | --- |
| MNIST / AdamW | 学習先頭2,000件、テスト先頭1,000件、100更新 | accuracy 86.4%、loss 0.455661 |
| MNIST / SGD | 同上（momentum 0.9、Nesterov、OneCycleピーク 0.1） | accuracy 85.2%、loss 0.479619 |
| MNIST / SGD（bias decayなし） | 同上（`-no-decay-suffix bias`） | accuracy 85.2%、loss 0.479656 |
| 数値回帰 | 4特徴量、4,000行（うち検証800行）、200更新 | validation RMSE 12.164137 |
| 音声denoiser | 生成8秒・16 kHzトーン、window 2,000（うち検証400）、300更新 | 検証SNR 4.8665 → 15.0490 dB |

MNIST（40更新で中断→100更新まで再開）、数値回帰（100更新で中断）、音声（150更新で中断）のいずれにおいても、再開時にreader / prefetch設定を変更しても中断なし実行と完全に同一の重みが得られました。

## OptimizerとSGDのstate

`Optimizer` はeagerな更新・リセット・学習率制御・寿命管理の共通インターフェースです。`CheckpointOptimizer` はアルゴリズムタグ付きのstateを扱い、`gograd-training-1` で `optimizer_kind` が省略されている場合は旧AdamWとみなします。

- **SGD**：PyTorchのcoupled weight decay、momentum、dampening、Nesterov、ascentに準拠します。初回momentumはdampeningを適用せず、パラメータごとに遅延初期化します。勾配を持たないパラメータは更新をスキップし、0勾配は更新対象となります。CUDA上では更新後に初期化フラグを立ててstate読み書きの競合を防ぎます。
- **CUDA graph連携**：デバイス上のステップ数と初期化フラグを保持し、`ZeroGrad` は既存の勾配バッファを0埋めします。保存およびeager復帰時にはステップ数を同期し、eager更新時はストレージversionを進めてBF16シャドウを無効化します。
- **パラメータグループ**：名前付き分割・学習率スケール・decayを保持し、OneCycleは基準学習率のみを変更します。stateはindexではなく順序付きのパラメータ名を保存し、異なるグループ構成への復元を拒否します（CUDA graphには記録時のスケールが固定されます）。
- **AdamWステップカウンタ**：optimizer全体で単一のステップカウンタを持ちます（勾配を受けたパラメータごとに進めるPyTorchとは、一部パラメータに勾配がない期間がある場合にbias correctionが異なり得ます）。

## 計測結果

RTX 3060 Tiと12 CPUスレッドを用い、同一セッションでgogradとPyTorchを交互に3回計測した1学習ステップ（forward・backward・勾配クリップ・AdamW）の中央値（単位：ms）です。
- **Transformer**：4層、model 256、heads 4、FFN 1024、batch 16、系列長256
- **Acoustic**：hidden 384、batch 24、400 frames
- **BF16**：PyTorchはautocast、gogradはFP32値＋BF16シャドウを使用

| モデルと実行環境 | CPU FP32 | CUDA FP32 eager | CUDA FP32 graph | CUDA BF16 eager | CUDA BF16 graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acoustic PyTorch | 2505.9 | 58.5 | 53.7 | 47.1 | 42.6 |
| Acoustic gograd | 3876.4 | 82.3 | 82.0 | 63.8 | 64.1 |
| Transformer PyTorch | 1291.6 | 25.9 | 23.9 | 13.5 | 12.0 |
| Transformer gograd | 2276.1 | 27.2 | 27.8 | 22.4 | 23.2 |

CPU時間の対PyTorch比はTransformerで1.76倍、Acousticで1.55倍でした（目標比はそれぞれ2.5倍 / 1.5倍）。GEMM比重の高い本構成ではCUDA graphによる短縮効果は限定的で、BF16 eagerはFP32より高速化したもののPyTorch BF16には及びません（Transformer BF16 eagerでPyTorch比1.66倍）。

※上表のCPU値は初期記録時のものです。現在の環境で同Transformerを交互に5回計測した中央値は、gograd 972.6 ms、PyTorch 474.3 ms（2.13.0+cu130、12スレッド）で比率は2.05倍でした。gograd自体の処理時間は約2.3倍高速化しましたが、PyTorch側の高速化幅がそれを上回っています。

同TransformerのCPUステップを `-cpuprofile` で計測した内訳は、SGEMM 31%、exp 10%、erf 8%、memmove 7%、要素演算勾配 6%、SGEMM packing 5%、memclr 5%（関数単位ではGELU 19%、softmax 8%）でした。`memmove` はSGEMMのpackingと、勾配の初回寄与をコピーする `addGrad` に由来します。

batch 1、heads 4、head幅64のattentionをCUDA eventで3回計測した中央値とピークメモリ増分は以下の通りです。

| 系列長 | 実体化ステップ | tiledステップ | 実体化のピーク | tiledのピーク |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.366 ms | 0.766 ms | 3.50 MiB | 1.01 MiB |
| 1024 | 1.601 ms | 9.606 ms | 50.00 MiB | 4.03 MiB |
| 4096 | 18.916 ms | 127.889 ms | 776.00 MiB | 16.12 MiB |

tiled経路は全系列長で出力・勾配ともに実体化経路と一致しメモリ消費を大幅に抑えましたが、Backwardの再計算コストにより実行時間は実体化経路より遅くなっています。Vulkan / DirectMLの推論計測については [GPU性能測定](GPU_BACKEND_PERFORMANCE.md) を参照してください。

## 可変長系列のCUDAプール

更新ごとにグラフを解放しつつ、パディング長が64通り変化する学習ワークロードでのメモリ推移です。サイズクラス化と上限設定によりキャッシュ肥大化を抑制しています（200ステップ継続時も同水準を維持、`cudaMalloc` 失敗時はキャッシュを解放して1回再試行）。

| 64ステップ後 | 使用中bytes | キャッシュbytes | 予約bytes |
| --- | ---: | ---: | ---: |
| 旧・正確なサイズごとの無制限cache | 10,927,812 | 6,154,440,208 | 6,165,368,020 |
| サイズクラス＋上限 | 10,949,888 | 762,475,264 | 773,425,152 |
| 上限256 MiBでの別実行 | 10,949,888 | 265,028,352 | 275,978,240 |

固定shapeのAcoustic学習におけるプールのオーバーヘッドは、77.3 ms → 77.8 ms（+0.7%）でした。