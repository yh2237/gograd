# gograd — 純Go機械学習基盤への大まかな実装計画

更新日: 2026-10-10

このファイルは、作業の間が空いても「何を目指していて、次に何を実装するか」を
読み直せるようにするための長期計画。期限や厳密なリリース順ではなく、依存関係と
実用上の到達点を優先する。細かな仕様は `docs/API.md`、実装上の説明は
`docs/DESIGN.md`、UtauTTS/Irodoriの移植状況は `docs/ROADMAP.md` を参照する。

## 最終目標

**PyTorchで行うような機械学習を、Goでも自然に行える、低依存の純Goライブラリにする。**

UtauTTSのPython依存をなくすために始めたが、今後は画像・音声・時系列・言語などにも
使える一般的なテンソル計算、自動微分、モデル構築、学習、推論の基盤へ広げる。

目標とする利用体験:

- `go get` で導入し、CPUではGoのツールチェーンだけで構築・実行できる。
- Tensor、Layer、Module、Loss、Optimizerを組み合わせて独自モデルを記述できる。
- forward → loss → backward → optimizer updateをGoで完結できる。
- データ読込、学習・評価、checkpoint保存・再開、推論までPythonなしで運用できる。
- CPUとGPUでできる限り同じ公開APIを使い、メモリ使用量や速度を把握できる。
- PyTorchの重みを読み込める互換性を保ち、移植時に演算・勾配の一致を検証できる。

「ほぼ依存なし」の境界:

- コアは標準ライブラリ中心。cgo、C/C++コンパイラ、Python、PyTorchを実行時の
  必須条件にしない。CPUには純Goのfallbackを維持する。
- SIMD用Go assemblyは利用してよい。特定命令が使えないCPUでも実行できるようにする。
- CUDA利用時のドライバ・runtime・cuBLAS・NVRTCなどは、GPUを利用するための
  外部要件として明示する。現在のWindows向けDLLバインディングもこの扱い。
- PyTorchによるfixture生成や旧checkpointの変換は、開発・移行用の任意ツール。
  通常のGo学習・推論や基本テストがそれらに依存しないことを目指す。

PyTorchの全API・分散システム・最新研究機能を一度に再現するのではなく、まず一般的な
モデルが学習できる一貫した基盤を完成させ、対応範囲を増やしていく。

## 現在地

- float32のdefine-by-run自動微分、CPU/CUDAの基本演算・view・broadcast・reduction。
- Linear、Embedding、Conv1d、normalization、attention、Transformer系の構成部品。
- Moduleの名前付きstate、safetensors、AdamW、gradient clipping、OneCycle。
- speech-timingの学習と再開、UtauTTS向けモデルの移植・PyTorch比較。
- Irodoriのtext/conditioning、DiT、codec、RF sampler、SilentCipherのencode経路。
- `dsp` にFFT/STFT・resampling、`torchrng` にPyTorch CPU乱数の互換経路。
- **汎用Conv2dとConv2dLayerを実装済み**: NCHW、groups/depthwise、stride/padding/dilation、
  CPU/CUDA forward・backward。PyTorch全勾配比較と数値微分が通っている。
- `cmd/cnn-train` で純GoのCNN学習 → 保存 → 再読込まで確認済み。
- **NCHW Poolingを実装済み**: MaxPool2d、AvgPool2d、AdaptiveAvgPool2dと各Layer。
  CPU/CUDA、ceil mode・padding/divisor・dilation、13ケースのPyTorch全VJPと数値微分を検証。
  view・retain graph・saved state・CUDA replayも確認し、CNN例にmax/adaptive poolingを組み込んだ。
  Linux/386のCPUテストも通過。CUDAサブテストはそれぞれOS threadを固定する。
- **汎用BatchNormを実装済み**: rank 2..6、channels-first/last、CPU/CUDA、affine/tracking無効化。
  biased/unbiased variance、固定momentum・累積平均、NoGrad training・eval、Module stateを追加。
  PyTorch 17ケースの全VJPと3系列のrunning state、保存再読込、view・retain・寿命・CUDA replayを検証。
  CNN例をConv2d/BatchNorm/ReLUに拡張し、40ステップの訓練accuracy 100%とreload差分0を確認。
  num_batches_trackedは現在のF32 stateで保持するため、正確な整数の範囲は2^24まで。
- **最小データAPIを実装済み**: 独立したdataパッケージにgeneric Dataset/Subset、Sampler、
  同期Loader、Stack/PadSequencesを追加。shuffle/seed/split、mask/lengths、epoch途中の再開を検証。
  callback失敗・cancelではsamplerを進めず、loader設定・順列・PCG stateをcheckpoint metadataへ保存。
  全packageのbuild/test/vet、data/MNISTのLinux race・Linux/386テストを通過。
  WSLでHTTP標準パッケージを初回buildする際は並列数を抑え（GOMAXPROCS=2、go test -p 2）、
  cacheが温まっていない状態のtimeoutを避ける。
- **Go-only実データ学習例を追加**: raw/gzip IDX readerとcmd/mnist-train（MLP/AdamW/OneCycle）。
  CPUの重み・moments・sampler/schedulerは中断なしと完全一致し、CUDAは数値一致を検証する。
  実MNIST train 2,000/test 1,000、seed 7、batch 64、100更新でCPU/CUDAともaccuracy 86.4%、
  test loss 0.455661、reload logits差分0。CPUは40更新で中断し、100まで再開した。
- **worker / bounded prefetchを追加済み**: サンプル読込のみ並列化し、collateは受取側で順番通り実行。
  pending batch数に上限を設け、fail/cancel/Close/NextEpochで読込goroutineを終了待ちする。
  checkpointは未受取の先読みを破棄してdelivered cursorだけ保存するため、worker設定を変更して再開できる。
  MNISTでworkers 4/prefetch 3 → 40更新で中断 → workers 2/prefetch 2で100まで再開し、
  CUDAではworkers 4/prefetch 4で確認。同期版同様accuracy 86.4%、test loss 0.455661、reload差分0。
  全package build/test/vet、data/MNISTのLinux race・Linux/386、gofmt/diffチェックを通過。
- **Optimizer共通APIとSGDを追加済み**: AdamW/SGDをOptimizer・CheckpointOptimizerに揃え、
  OneCycleとcombined checkpointを汎用化。旧AdamW・旧MNIST checkpointの読込を維持。
  SGDはCPU/CUDA、momentum/dampening・Nesterov・coupled decay・maximizeとlazy bufferを実装。
  PyTorch 7ケース、state再開・CUDA graph replay・BF16/version検証を通過。
  MNIST SGD + momentum .9 + Nesterov、peak LR .1、100更新でCPU/CUDAともaccuracy 85.2%、
  test loss 0.479619、reload差分0。CPUは40更新で中断・worker設定を変えて再開した。
  全package build/test/vet、SGD/checkpoint/MNISTのLinux race・Linux/386、整形・diff検証を通過。
- **Optimizer parameter groupsを追加済み**: ParameterGroupSpec/PartitionParametersと  NewSGDGroups/NewAdamWGroupsで層ごとのLR scale・weight decayを持てるようにし、group membership・
  scale・decayをoptimizer stateとcombined checkpointへ保存。旧unordered/flat checkpointは読込を維持し、
  group構成が異なるcheckpointはlive stateを変更する前に拒否する。
  PyTorch 2ケース（SGD/AdamWの2 group、欠落・ゼロ勾配）でCPU/CUDAの全パラメータ・momentum・
  group記録を照合。state再開・CUDA graph replay・scheduler scalingも検証。
  MNIST SGD + Nesterov + `-no-decay-suffix bias`（biasをdecayなしgroupへ）でCPU/CUDAとも
  accuracy 85.2%、test loss 0.479656、reload差分0。CPUは40更新で中断しworker設定を変えて再開した。
  全package build/test/vet、Linux race・386、整形・diff検証を通過。
- **数値データのGo-only読込例とアプリ側の再現性契約を追加済み**: dataにParseNumericCSV/
  NumericTable/RegressionSample・RegressionBatch・StackRegressionと、training splitでfit する
  Standardizer（float64で集計、stateのJSON保存・検証）を追加。cmd/tabular-trainはCSVを読み
  （-generateで決定的な合成データを生成）、RandomSplitで分割し、train統計だけで標準化してMLPをMSEで
  学習、standardizer state・loader順序・データfingerprintをAdamW/OneCycleと共に保存する。
  データ・分割・batch・steps・seed・hidden・optimizerの変更は読込前に拒否する。
  4,000行/800検証、seed 7、batch 64、200更新でCPU/CUDAともvalidation RMSE 12.164137、
  reload差分0。100更新で中断しworker設定を変えて再開した重みは未中断実行とSHA-256まで一致した。
  全package build/test/vet、data/tabularのLinux race・Linux/386、整形・diff検証を通過。
- **音声データのGo-only読込例とaugmentation再現性契約を追加済み**: dataにReadWAV/ReadWAVFile/
  ReadWAVMono/WriteWAV（PCM 8/16/24/32、float32/64、WAVE_FORMAT_EXTENSIBLE、mono平均、丸め付き
  整数encode、往復誤差≤1.5LSB）を追加。cmd/audio-trainはWAVを読み（-generateで決定的な合成音を
  生成）、固定windowへ分割し、noiseを(seed, window index)のPCGから導いて反復順に依存させない
  augmentation契約を示す。MLP denoiserをMSEで学習し入力/出力SNR(dB)を報告、model/optimizer/
  OneCycle/loader順序/WAV fingerprintをまとめて保存・再開する。
  8秒16kHz（2,000 window、400検証）、seed 7、batch 64、300更新で入力SNR 4.8665dB→出力15.0490dB、
  CPU/CUDAともreload差分0。150更新で中断しworker設定を変えて再開した重みは未中断実行とbyte一致。
  全package build/test/vet、data/audioのLinux race・Linux/386、整形・diff検証を通過。
- **CUDAのdevice/stream/capture分離を追加済み**: implicit streamとcapture可否をOS thread単位に
  し、Captureは明示stream必須・入れ子拒否・現在deviceを変更せず記録してreplay時に再選択する。
  割当poolをdevice×size classで分離し、Bufferがdeviceを保持してcopy/memset/freeで所有deviceを
  選び呼出側deviceを復元する。autograd.NewCUDAContextOnはdeviceを検証し、Closeでthreadの
  以前のdeviceを復元する。cmd/gputcn-trainはstream選択前にOS threadを固定する。
  2本のthreadが同時に別streamを保持しても互いのstreamを見ないこと、別deviceのcached blockを
  reuseしないこと、capture入れ子拒否とClose後のdevice復元をテストで確認。
  WindowsのCUDAテスト、Linux race（cuda/kernels/gputcn/autograd）・Linux/386を通過。
- **独立したCUDA学習のスレッド分離を完了**: cuBLASハンドル・勾配クリップのプラン/ノルム・
  プロファイルのカウンタをスレッド単位にし、割当プールをdeviceとstreamで分離した。
  分離前は、別々のstreamで同時に学習すると解放済みバッファが別streamへ再利用され、
  結果が逐次実行と一致しなくなった（同一seedでも0.08ずれることがあった）。
  stream単位のプール分離後は、混在seedの同時実行も逐次実行とbit一致することを確認。
  独立2スレッド同時学習（別context・別model・別optimizer・別stream、勾配クリップあり）、
  並行クリップ、stream別プールのテストを追加。`LoadKernel`はnvrtcコンパイル中の
  スレッド移動でコンテキストを失わないようOSスレッドを固定した。
  WindowsのCUDAテスト（cuda/autograd/kernels/gputcn、-count複数回）と、Linux race・386を通過。
- **stream単位の実行を追加済み**: cuBLASがスレッドのカレントstreamに追従するようにし、
  カーネル・memset・コピー・GEMMが同じstreamに乗るようにした。`cuda.WithStream`と
  `cuda.CurrentStream`、`ExecutionContext.SetCUDA`/`CUDA`/`WithCUDA`（deviceとstreamを
  スコープの間だけ適用し、panic時も含めて復元）を追加。1スレッドで2つのstreamへ
  独立した学習を重ねてenqueueし、既定streamの逐次実行と一致することを確認。
  captureがstreamオブジェクトを記録していなかったため、capture中にcuBLASが既定streamへ
  戻って落ちる問題も修正（`SetBLASStream`無しのcaptureテストで検証）。
  WindowsのCUDAテスト（cuda/autograd/kernels/gputcn）と、Linux race・386を通過。
- **stream間の合流primitiveを追加済み**: `Stream.WaitEvent`（`cudaStreamWaitEvent`）と
  `Event.Record`の生存チェックを追加し、1つの記録の分岐を別streamで走らせてeventで合流
  できるようにした。2分岐のforward（それぞれ別stream）→event合流→合流後の損失・backwardを
  1スレッドで実行し、パラメータと勾配が単一stream実行と一致することを確認。
  分岐の切り方と合流は利用者が指定し、記録の自動分割は行わない方針を明記した。
  WindowsのCUDAテストと、Linux race・386を通過。
- **分岐合流APIを追加済み**: `StreamBranch`/`RunBranches`で、1つの記録の中の独立した分岐を
  それぞれのstreamへenqueueし、呼び出したスレッドのカレントstreamが全分岐を待つeventを
  記録してから戻すようにした。`Event.WaitOn`（nilで既定stream）も追加。3分岐のforwardを
  3つのstreamで実行して合流し、パラメータと勾配が単一stream実行と一致することを、合流先が
  既定streamの場合と明示streamの場合の両方で確認。分岐記述の事前検証と、失敗時に先行分岐を
  合流させてからエラーを返す挙動もテストした。演算ごとの自動分割は行わない方針を明記。
  WindowsのCUDAテストと、Linux race・386を通過。
- **ExecutionContextで記録スコープを分離済み**: inputsとModuleのparameters/buffersを
  明示的に関連付け、NoGradの入れ子・panic復帰・混在拒否を実装。独立したCPUモデルの
  学習・推論を並行実行し、Linux/WSLのrace detectorでも確認済み。
- **共有ストレージとグラフ依存の参照管理を実装済み**: view・分岐・保存状態の寿命を分離。
  RetainDataとRetainGraph、CopyFrom/eager AdamWによる変更のversion検証を追加。
   CPU/CUDAの寿命・複数backward・保存統計・BF16/indicesの回帰テストと、Linux上のrace検証を通過。
- **実行設定をコンテキスト単位へ分離済み**: ExecutionOptions、SetOptions/Options、Autocastを追加。
  BF16・attention選択の独立性、scope終了後のbackward、retain graph、CUDA capture/replayを検証。
  backwardはforward時の設定を保持し、旧globalsの互換性も維持する。

これは一般的なPyTorch代替が完成したという意味ではない。データAPI、dtype、グラフの
実行・所有権、汎用演算群、GPUの対応プラットフォームなどは拡張の余地が大きい。
同じ処理が複数の旧経路に存在するため、共通化も段階的に必要。

## 開発方針

- **モデル専用コードで得た能力を、再利用できる演算・Layerへ引き上げる。**
  特定のモデル移植だけを積み重ねず、他のモデルでも利用できる形にする。
- 公開演算のshape、layout、dtype、device、エラー条件を説明する。
- 新しい一般演算はCPUの基準実装から始め、CUDA実装と勾配検証へ広げる。
  未対応バックエンドをホストへの暗黙転送で隠さない。
- 行列積など既存の共通基盤を利用する。巨大な一時行列や解放されないグラフを避ける。
- 他言語や他実装との一致は、まず数学的意味と許容誤差で評価する。
  bit一致が必要な乱数・状態・保存形式は別途厳密に検証する。
- APIが増える前に、実行コンテキストと所有権を整理する。
- 一つの段階ごとに、**実際に学習・保存・再読込できる例**を残す。
- 高速化は同じ条件の学習ステップ全体で測る。microbenchmarkだけで実用性を判断しない。
- 旧APIの利用箇所を確認し、互換ラッパー・移行説明を用意してから整理する。

## 実装順と到達点

### 1. グラフ実行・所有権の土台【基本スコープ・寿命を実装済み】

記録状態を実行コンテキスト単位へ移し、その上でグラフとストレージの寿命を整理する。
記録スコープ、共有ストレージの寿命、BF16/attention設定のAPIと基本契約は実装済み。
CUDA stream/device/captureの独立性は後続の拡張として扱う。

- [x] 独立したグラフが互いの記録状態に干渉しない仕組み。
- [x] NoGradの入れ子、panic時の復帰、detach、train/inferenceの混在を整理。
- [x] 複数goroutineで独立モデルを扱う際の契約とテスト。
- [x] CUDA OS thread・stream・captureの制約をCPUの並行実行と区別して説明。
  implicit stream/captureをOS thread単位にし、device検証と復元、deviceとstream別のpool、
  スレッド別cuBLASハンドルと勾配クリップstateを追加した。
  1ステップ内の複数stream重畳と、stream/deviceを束ねたExecutionContextの分離は残る。
- [x] shared storage/viewの所有権、graph release、保存済みテンソルの寿命を明確化。
- [x] managedなin-place変更のversion counter、retain graphを追加。
- [x] BF16・attention選択をコンテキストへ移し、forward/backwardの設定を一貫させる。

**到達点:** 複数の独立したCPU学習・推論を安全に並行実行でき、グラフとメモリの
寿命を説明できる。CUDAの並行実行は未対応の部分を明示したうえで段階的に進める。

記録スコープは `ExecutionContext` として実装した。旧 `New`/`NoGrad` は互換用の共有
スコープで維持し、新APIでは `execution.New` と `execution.BindModule` を利用する。
contextはTensorから演算・viewへ伝播する。共有allocationのleaseとグラフの依存数を分け、
保存した統計・target・indicesもbackwardが終わるまで保持する。version検証はCopyFromと
eager AdamWの変更を対象とし、Dataの直接書換えやCUDA外部kernel/replayは追跡しない。
実行設定はExecutionOptionsとしてforward時にコピーし、backward closureが保持する。
Autocastの入れ子・panic復帰とcontext独立性を検証。CPU・Conv2d・tiled flashはFP32のまま。
CUDA streamや自動GCによるグラフ管理まで完了した扱いにはしない。

### 2. 一般的なモデルを組める演算・Layer

- [x] 汎用Conv2d、Conv2dLayer、CNN学習例。
- [x] MaxPool2d、AvgPool2d、AdaptiveAvgPool2dと各Layer、CNN例への組込み。
- [x] BatchNormのtrain/eval、running statistics、backward、state保存、CNN例への組込み。
- [ ] reshape/index/gather/scatter、flatten、padding、各種activationの不足を補う。
- [x] MSE/L1/BCE系など一般的なlossとreduction/ignore/maskの契約を整える。
  `MSELoss`/`L1Loss`/`BCEWithLogitsLoss`（mean/sum、要素ごとの重み、PyTorch準拠の除数、
  安定版BCE）を追加し、12ケースのPyTorch fixtureとゼロ重みマスクをCPU/CUDAで検証した。
  NaNフレームマスクは従来どおり`MaskedLoss`が担当する。
- [ ] ConvTranspose、1D/2Dのgrouped convolutionなどを汎用自動微分経路に揃える。
- [ ] RMSNorm、RoPE、SwiGLU、GQA、causal/cross attentionなどの再利用性を改善。
- [ ] RNN/LSTM/GRUなどは必要な用途ができた時点で追加する。
- [ ] `nn`/`gputcn`/モデル専用経路との重複を、互換性を維持しながら整理。

**到達点:** MLP、一般的なCNN、時系列ネットワーク、Transformerの小規模モデルを
モデル専用の数値処理なしで構築・学習できる。

優先度は「実用モデルで必要」「複数の用途に再利用できる」「独立した勾配検証ができる」
の順で判断する。画像専用機能だけに偏らない。

### 3. データから学習・評価・再開までの標準的な流れ

- [x] Dataset、Sampler、Batch/Collate、同期Loaderを分離した最小データAPI。
- [x] variable-length、padding、mask、shuffle、seed、split、epoch途中の再開状態。
- [x] worker/bounded prefetchを追加し、CPUの同期経路も維持。順序・fail/cancel・再開・Closeを検証。
- [ ] 画像・音声・数値データの小さなGo-only読込例。任意の変換ツールと切り分ける。
  画像は raw/gzip IDX と MNIST 学習・保存・再開例、data.ReadImageFile/WriteIDX/ScaleImage で
  PNG/JPEGの読込・変換API、cmd/image-to-idxでディレクトリ配下の画像をIDXへ変換、
  cmd/cnn-train が real MNIST IDX 画像を学習できる経路を追加。
  CSVの標準化・回帰例とWAVのdenoise例も実装済み。大規模な実データでの品質・メモリ検証は別途必要。
- [x] seed/index固定の基本augmentationとworker・再開での再現性検証。
  `TransformDataset`の借用メモリ・並行呼出し・subset index・metadataの責任を文書化。
  水平反転のshape/確率検証とuint64 seed API、ノイズの有限性/overflow検証を追加。
  音声例も共通ノイズ関数へ移し、元の式との厳密一致をテストする。
  2026-10-16: 全packageのCGO_ENABLED=0 build/testとvet、data/audio/cnnのLinux race・386を通過。
  音声300更新のCPU基準値4.8665→15.0490 dBとreload差分0を維持。
  epochごとの変換変更や、複数変換の独立した乱数系列の指定は未実装。
- [ ] Optimizer共通API、parameter groups、SGD/momentumなどAdamW以外の選択肢。
  共通API、SGD/momentum、parameter groups は実装済み。次のoptimizer拡張は必要になった時点で追加する。
- [ ] 基本scheduler、勾配蓄積、clip、評価指標、ログ・中断・再開の共通化。
  勾配蓄積は`GradAccumulator`（損失の1/groupsスケール、group完了の報告、端数groupの契約）として
  実装し、MNIST例に`-accum`を追加した。評価指標は`metrics.Mean`/`Accuracy`/`Regression`を
  実装し、MNISTと数値回帰の例へ組み込んだ。MNISTの蓄積途中の中断はgroup先頭のsamplerへ戻し、
  数値回帰・音声もキャンセルを返す前にcheckpointを保存する。ログ・中断/再開は`TrainingLoop`（更新境界、報告間隔、蓄積途中のsampler巻戻し、停止時の保存）として共通化し、3つの例へ適用した。
- [x] model/AdamW/OneCycle/loader stateをまとめたcheckpoint APIの汎用利用（MNIST例）。

**到達点:** 外部データを読む実験を、学習・validation・保存・中断再開までGoで
書ける。単純な学習ループを利用者が制御できる余地は維持する。

### 4. 推論の実用性・GPU常駐・キャッシュ

- [ ] Irodoriで固定条件のcontext KV cacheを導入し、繰り返しの投影を減らす。
- [ ] 重み・中間テンソルをCUDAに保持し、演算ごとのupload/readbackを減らす。
- [ ] attention・normalization・codecなど主要ボトルネックをGPUへ広げる。
- [ ] autoregressive TransformerのKV cache、decode loop、sampling。
- [ ] CUDA graphやshape別の実行計画を、モデル専用実装から共通APIへ発展させる。
- [ ] 同じ重み・入力・seed・precisionでend-to-end時間とpeak memoryを測る。

**到達点:** 学習済みモデルをGoで読み込み、実用的な時間・メモリで推論できる。
UtauTTS/Irodoriはこの基盤を評価する実際の利用例として継続する。

### 5. dtype・mixed precision・ストレージ

- [ ] dtypeを正式なストレージ・dispatch・viewの設計に組み込む。
- [ ] int/boolのindices・mask、float64、FP16、BF16の用途を整理する。
- [ ] cast、dtype promotion、FP32 accumulation、master weightsの規則。
- [ ] BF16 shadow方式から、必要に応じて実ストレージ・compute dtypeへ拡張。
- [ ] autocast、FP16 loss scaling、optimizer stateのdtype。
- [ ] dtypeごとのforward・勾配・学習安定性・メモリの検証。

**到達点:** dtypeを利用者が制御でき、mixed precisionで性能・容量を改善できる。
現在のfloat32 APIを壊さず、非対応の組み合わせは明確に拒否する。

### 6. バックエンドと性能・プラットフォーム

- [ ] CPU SGEMM以外の重要なreduction・pointwise・normalizationを改善。
- [ ] 融合kernel、attention、convolutionの時間・一時メモリを改善。
- [ ] goroutine/thread数、memory pool、device/streamごとの分離を整理。
  deviceとstream単位のpool、スレッド単位のcuBLASハンドルとclip stateは実装済み。
  goroutine/thread数の制御と、stream別のメモリ使用量の計測は今後。
- [ ] Windows以外のGPU backend対応を検討する。まずLinux CUDAの実用性を評価。
- [ ] macOS/arm64などでもCPU fallbackと基本学習を安定して提供する。
  Windows/386は既存CUDA allocatorの2 GiB int定数がoverflowしてbuildできないため、
  32-bit Windows対応を行う場合はこの既知の制約を先に修正する。
- [ ] backendの追加方法を公開APIから分離し、CPU-only buildを維持。

**到達点:** 利用者がCPUで始め、必要に応じてGPUへ移れる。性能上の制約が計測・説明
されており、特定の開発マシンだけで動くライブラリにならない。

### 7. モデル・形式・ドキュメント・使いやすさ

- [ ] 推奨のTensor/Module APIを明確にし、旧経路の使い分けを減らす。
- [ ] safetensors、モデル構成metadata、version付きcheckpointの互換方針。
- [ ] PyTorchからの移植手順、layout差、dtype差、seed差の説明。
- [ ] MNIST相当の画像分類、音声・時系列、テキストの学習例をGoで用意する。
- [ ] 入門 → 独自モデル → 学習再開 → GPU → 実用運用の順で説明を整える。
- [ ] API安定化・リリース・互換性の方針を定める。

**到達点:** gogradを初めて使う人も、PyTorchで書いていたモデルや実験をGoで
再構成できる。変換に必要なPythonは移行時だけの任意依存にする。

### 8. コアが安定してから広げる高度な機能

一般用途の土台が完成してから、需要と実用性を見て進める。

- higher-order autograd、gradient checkpointing、compiled execution。
- multi-GPU・distributed training、通信backend、data sharding。
- quantization、sparse tensor、より高度なkernel/compiler最適化。
- その他PyTorchの広い機能群との互換性。

**到達点:** 基本機能と互換性を維持しながら、より大きなモデルや新しい用途を扱える。

## 検証と実用化

各段階の完了判断には、必要な範囲で以下を用いる。

1. **演算の正しさ:** PyTorch fixture、独立した数値微分、境界shape。
2. **合成の正しさ:** 実際の学習でlossが下がること、保存・読込・再開後の整合性。
3. **バックエンド:** CPU/CUDA双方の出力・勾配、GPU residency、一時メモリ解放。
4. **並行性:** contextの独立性、必要なrace test、入れ子・エラー時の状態復帰。
5. **運用:** end-to-end時間、peak memory、可変長や反復でのメモリ増加。
6. **CI:** build、vet、test、gofmt、複数OS。Pythonがない環境の基本検証も維持する。

一時的な許容誤差拡大だけで失敗を解消しない。例えばfloat32式のFMA融合やCUDAの
atomic reductionの順序は結果を変え得るため、原因を調べ、必要な一致レベルを説明する。

Irodoriなど個別のアプリケーションで未検証のrelease gateは、汎用ライブラリの機能完成と
区別する。ある演算の一致はモデル全体の品質・速度の証明ではない。

## 次の作業と更新方法

### 保留（2026-10-11）: UtauTTS向けIrodori F0推論

**保留中。** UtauTTSは推論時にIrodoriを動かす方針をやめ、Irodoriを教師にした抑揚モデル（v12）を
作る方針に変えた（コストに対して効果が見合わない、ユーザー判断）。途中の実装はブランチ
`wip/irodori-f0-inference`（`F0Inference`、話者状態の保存・読込、文脈K/Vのcache、8ステップjointの
全文parity fixture）にある。s1（2.26秒）は8ステップで全配列parity通過（latent 6.1e-4、波形4.1e-4）。
CPU end-to-endは約383秒（sampling約372秒、decode約16秒）、常駐F32約3.1 GB、host peak約7 GB。
重みの常駐・文脈K/V cacheは汎用部品として再利用できる。以下は当時の計画の記録。


UtauTTSで、推論時にIrodoriへ文を読ませてF0の形だけを使う方式が、現行の抑揚モデルに
聴取で14-2と大勝した（UtauTTS `notes/2026-10-11-reference-tts-inference.md`）。これを
Python/PyTorchなしで製品に入れるため、汎用基盤の作業（parameter groups以降）より先に進める。
Irodoriの音声はF0の分析にだけ使い、外へ出さない（WAV出力のrelease gateとは別の経路）。

使う設定（PyTorchで調査済み。この設定でF0の形は既定40ステップの1回の読みと同等）:

- 8ステップ、`joint` CFG（`cfg_scale` 3、text・speakerを同時に外す1枝＋条件つき1枝）、caption・truncationなし。
- 話者は事前に計算した状態をファイルから読む（推論中に参照WAVのDACVAE encodeをしない）。参照なしは不可（形が崩れる）。
- seedは文ごとに固定でよい。PyTorchとbit一致の乱数（`torchrng`）は既にある。
- 出力は48 kHzの波形（float32）。UtauTTS側でF0を推定する。透かし・WAV書き出しは不要。

到達点（ゲート）:

- [ ] **正しさ:** 上の設定の全文（UtauTTSの評価9文、2〜6秒）で、PyTorchと同じ重み・話者状態・seedの
  最終latent・波形が一致する（既存のparity閾値。8ステップ分の増幅を確認する）。
- [x] **API:** `irodori` に「重みを1回読み込み常駐 → 文ごとにtext＋話者状態＋seed → 波形」を返す
  推論用の型を用意する（UtauTTSからライブラリとして呼ぶ）。話者状態の保存・読込（safetensors）と、
  参照WAVから話者状態を作るGoの関数（事前計算用、遅くてよい）。
- [ ] **速度:** 2.3秒の文で、CUDAは1.5秒以下、CPUは15秒以下を目安（PyTorch CPUの目安: DiT 1回約0.33秒×
  12回＋decode約3秒≒7秒）。CPU/CUDAとも同じ条件でend-to-endの時間とpeak memoryを記録する。
- [ ] **メモリ:** 常駐の重み（DiT・ModernBERT・DACVAE decoder）がGPUの共有予算約3 GBを超える場合は、
  BF16の重み保持かCPUとの分担を選べるようにする。

作業の候補（汎用部品として作れるものは共通APIへ）:

1. 重みの常駐（DiT・ModernBERT・decoder）と、文ごとのファイル読込の排除。
2. CFGの2枝をbatch 2でまとめて1回のforwardにする。
3. textと話者の文脈は全ステップで一定なので、文脈側のK/V投影をステップ間でcacheする。
4. CPU: attention・RMSNorm・AdaLN・SwiGLUの融合と、行列積のマルチスレッド化。現状はDiT 1回が
   PyTorch CPUの約75倍遅い（2.3秒の文の1ステップ4枝でGo 99秒、PyTorch 1.3秒）。
5. CUDA: 投影ごとのupload/readbackをやめ、DiT全体を常駐で実行する（attention・normもGPUへ）。
6. DACVAE decoderの高速化（現状 Go 14秒、PyTorch約3秒）。

2026-10-11 進捗: `F0Inference` は必要な text/DiT/duration/decoder 重みを常駐し、
8-step joint CFG 3 の透かし前48 kHz波形と最終latentを返す。話者状態は safetensors で保存・読込でき、
8本のWAVからGoで作る関数を追加した。呼び出しとCloseはmutexで直列化する。
話者なしの `InferNoReference` も、全falseの話者maskとdurationのnull-speaker分岐で追加した
（UtauTTS側の品質評価とGo/PyTorchのno-ref全文parityは未了）。

実測済みの評価文 `s1`（2.26秒）は8ステップで全latent・全波形のPyTorch parityを通過。
右paddingと空captionの省略後、最大絶対誤差はlatent 6.07e-4、波形 4.10e-4
（既存閾値0.02/0.05のまま）。CPU end-to-end 382.95秒（初回重み読込44.8秒を除く）、
内訳は条件0.566秒、duration 0.045秒、sampling 362.70秒、decode 19.50秒。
常駐重み3,142,208,652 byte、観測host peak working set約5.68 GB。
変更前は392.60秒、latent 3.90e-4、波形2.65e-4、host peak約6.96 GBだった。
文脈K/Vをステップ間でcacheする経路は2ステップの実checkpointでcache無効時とbit一致。
9文全体・ステップ別誤差・CUDA時間とpeak memoryは検証中。CPU 15秒、CUDA 1.5秒の
目標は未達であり、CFG batch 2、GPU常駐DiT、BF16重み保持、dense計算とdecoderの高速化が残る。

### 今すぐの次の作業

1. CUDA stream/device/captureの独立性は、thread単位の分離、stream別プール、stream/deviceを
   束ねたExecutionContext、`RunBranches`による分岐合流まで実装済み。演算ごとの自動分割は
   行わない方針とする。
2. 勾配蓄積と評価指標は実装済み。ログ・中断/再開は`TrainingLoop`として共通化し、3つの例へ適用した。
   次は実データ（より大きな画像・音声・時系列）と変換ツールの切り分け、augmentation契約の拡張を進める。
3. parameter groupsの実利用箇所（層ごとのLR/decay）を増やし、必要ならoptimizer拡張を追加する。

記録スコープの独立性、入れ子・panic復帰、CPU並行実行、共有ストレージの基本寿命と
実行設定の独立性は実装済み。作業再開時は `autograd/execution_context.go`、
`execution_options.go`、`storage.go`、`backward.go` と対応するテストを確認する。
全packageのbuild/test/vetと、Linux/WSLのCPU race検証を通過。公開説明は `docs/API.md` にある。

phase 3の最小データAPI、worker/prefetch、Optimizer共通API・SGD・parameter groups、画像/数値/音声のデータ例、
CUDAのdevice/stream/capture thread分離は実装済み。次は**1ステップ内の複数stream重畳とstream/deviceを
束ねたExecutionContext**、実データの大規模化、augmentation契約の拡張を進める（Irodori F0推論は保留）。
CUDA分離は `cuda/graph_windows.go` のthread別launchState、`cuda_windows.go` のdevice別pool、
`autograd/cuda_context.go` のNewCUDAContextOnを参照する。
`irodori.ReadWAVMono` は data.ReadWAVMono へ統合できる重複候補（モデル経路の検証を保ったまま整理する）。
CNNの実データ学習や追加データ形式もこのAPIを利用する。
Irodoriのcache/GPU最適化は共通部品を活用できる
単位で並行して進めるが、汎用基盤の改善を止めない。

### 再開時に見るもの

- このファイルの現在地とチェックリスト。
- `git status` と直近の変更。未コミットでも実装・検証済みの作業があり得る。
- `docs/API.md` の公開API・制約。
- `docs/DESIGN.md` の構造。
- 関連テスト、CI結果、計測結果。

### 計画の更新ルール

- 実装・必要な検証が終わった項目を `[x]` にする。
- 大きな段階が終わったら現在地と「今すぐの次の作業」を更新する。
- 仕様変更や新しい制約は、該当のAPI/設計ドキュメントにも反映する。
- 優先順位を変えるときは理由を短く残す。終わっていないことを終わった扱いにしない。
- 完了したタスクの細かい記録をここへ延々と追加せず、再開に必要な情報を優先する。
