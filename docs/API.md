# 学習API

gogradの学習で使う公開APIの一覧です。学習checkpointは`safetensors`の`gograd-training-1`形式で保存し、推論用の重みは名前とレイアウトをそのまま保ちます。

## 学習で使うAPI

| パッケージ | API | 役割 |
| --- | --- | --- |
| `tensor` | `Device`、`CPU`、`CUDA` | テンソルやモデルを構築するときにバックエンドを選びます。 |
| `autograd` | `NewCUDAContext`、`NewCUDAContextOn`、`CUDAContext.Device`、`CUDAContext.Close` | 学習ステップの間、呼び出したOSスレッドにCUDAコンテキストを保持します。deviceを選んで検証し、`Close`でそのスレッドの以前のdeviceを復元できます。 |
| `autograd` | `New`、`Tensor.ToHost`、`Tensor.Backward`、`Tensor.ReleaseGraph`、`Tensor.Close` | float32テンソル、明示的な読み戻し、寿命の管理。 |
| `autograd` | `RetainData`、`RetainGrad`、`BackwardWithOptions`、`BackwardOptions` | 共有ストレージの寿命、forward値と勾配の保持、複数回のbackwardのための履歴保持。 |
| `autograd` | `NewExecutionContext`、`ExecutionContext.New`、`Zeros`、`Bind`、`BindModule`、`NoGrad`、`Recording`、`SetCUDA`、`CUDA`、`WithCUDA`、`Tensor.ExecutionContext` | テンソルとviewが伝える明示的な記録スコープ。独立したCPUモデルは別々のコンテキストで学習と推論を並行できます。`SetCUDA`でdeviceとstreamを記録し、`WithCUDA`が呼び出したOSスレッドの選択を一時的に切り替えて元に戻します。 |
| `autograd` | `ExecutionOptions`、`ExecutionContext.SetOptions`、`Options`、`Autocast` | コンテキストごとのBF16とアテンションの方針、スコープ付きの精度、backwardに渡すforward方針のスナップショット。 |
| `autograd` | `NewEmbeddingLayer`、`NewConv1dLayer`、`NewLayerNormLayer`、`NewLinearLayer`、`DropoutLayer`、`GELULayer`、`Sequential` | 再利用できるグラフモジュール。畳み込みの活性化は`[batch,time,channels]`、重みは`[out,in,kernel]`です。 |
| `autograd` | `Conv2d`、`Conv2dOptions`、`NewConv2dLayer`、`Conv2dLayer` | NCHWの活性化、`[out,in/groups,kH,kW]`の重み、任意のbias、grouped/depthwise畳み込み、CPU/CUDAの勾配。 |
| `autograd` | `MaxPool2d`、`AvgPool2d`、`AdaptiveAvgPool2d`、各オプションと層 | CPU/CUDAのNCHW pooling。重複する窓の勾配、ceil mode、明示的な平均の除数を持ちます。 |
| `autograd` | `BatchNorm`、`BatchNormOptions`、`NewBatchNormLayer`、`BatchNormLayerOptions`、`BatchNormLayer` | rank 2〜6のチャネル正規化、CPU/CUDAの学習・評価の勾配、名前付きのaffineパラメータとrunning state。 |
| `autograd` | `Module.StateDict`、`LoadStateDict`、`SaveSafeTensors`、`SaveSafeTensorsMetadata`、`LoadSafeTensors` | 名前付きのfloat32 state。metadataの値は文字列です。 |
| `autograd` | `MaskedLoss(pred,target,false)`、`ClipGradNorm`、`NewAdamW`、`NewOneCycle` | NaNフレームをマスクしたL1、クリップ、AdamW、PyTorchの2段cosineのOneCycle LR。 |
| `autograd` | `AdamW.State`、`LoadState`、`SaveSafeTensors`、`LoadSafeTensors`、`Close` | CPU/CUDAの名前付きパラメータ形状、モーメント、ハイパーパラメータ、ステップ数。optimizerが持つバッファを解放します。 |
| `autograd` | `Optimizer`、`CheckpointOptimizer`、`OptimizerSnapshot`、`LearningRate`、`SetLearningRate` | 共通の更新・リセット・学習率・寿命のAPIと、AdamW/SGDの型付きcheckpointスナップショット。OneCycleはOptimizerを受け取ります。 |
| `autograd` | `NewSGD`、`SGDOptions`、`SGD.State`、`SGDState`、`LoadState`、`SaveSafeTensors`、`LoadSafeTensors` | CPU/CUDAのSGD。momentum/dampening、Nesterov、coupled decay、ascent、momentumの遅延初期化、stateの再開。 |
| `autograd` | `ParameterGroupSpec`、`PartitionParameters`、`NewSGDGroups`、`NewAdamWGroups`、`OptimizerGroupState` | SGD/AdamWの名前付きパラメータグループ。グループごとの学習率スケールとweight decay、optimizer stateに記録するgroup membership。 |
| `autograd` | `SaveTrainingCheckpoint`、`LoadTrainingCheckpoint`、`TrainingCheckpointMetadata` | model/bufferとAdamW/SGDとOneCycleを、呼び出し側のmetadataと共に1つのファイルへ原子的に保存するcheckpoint。名前・形状・stateはlive objectへコピーする前に検証します。 |
| `autograd` | `Module.Train`、`SetTraining` | train/evalモードの再帰的な伝播。`OnTrainingChange`はdropoutなどのstatefulな層をモジュールツリーへ接続します。 |
| `autograd` | `IndexLoader.State`、`LoadState`、`WindowSampler.Batch`、`State`、`LoadState` | 再開できるepoch順またはPCGのランダムwindow。短い系列のパディングとcontextの破棄の判断を含みます。 |
| `autograd` | `StreamBranch`、`RunBranches` | 1つの記録の中の独立した分岐をそれぞれのstreamでenqueueし、呼び出したスレッドのカレントstreamへeventで合流させます。分岐の切り方は呼び出し側が指定します。 |
| `cuda` | `SetCurrentStream`、`CurrentStream`、`CurrentStreamHandle`、`WithStream` | 呼び出したOSスレッドの暗黙のstreamの選択と照会。`WithStream`はスコープの終了時に前の選択へ戻します。 |
| `cuda` | `Event`、`NewEvent`、`Event.Record`、`Stream.WaitEvent`、`Event.WaitOn`、`ElapsedTime` | stream間の順序付けと計時。あるstreamの完了をeventとして記録し、別のstreamがそれを待ってから読み取ります。`Event.WaitOn`は待たせる側をstreamの値または既定のstreamで指定します。 |
| `cuda` | `MemoryStats`、`ResetAllocationPeak`、`SetPoolCacheLimit`、`ReleasePool` | 使用中・キャッシュ・予約バイトの照会。再利用するCUDA割当の上限設定と解放。 |
| `data` | `Dataset[T]`、`SliceDataset`、`NewSubset`、`RandomSplit` | ホスト側のランダムアクセスsample、index view、決定的で重複しない分割。 |
| `data` | `Sampler`、`StatefulSampler`、`NewIndexSampler`、`IndexSamplerState` | トランザクション的なindex選択、seed付きの順列、epoch途中と将来のepochの再開。 |
| `data` | `NewLoader[T,B]`、`LoaderOptions`、`LoaderState`、`CollateFunc`、`Loader.Close` | contextを考慮したバッチ処理、任意の順序付きホストprefetch、drop-last、不透明なsampler checkpoint payload。 |
| `data` | `FloatSample`、`FloatBatch`、`Stack`、`PadSequences`、`PaddingOptions`、`OpenIDX` | 所有権を持つfloat32のホストバッチ、系列長とboolマスク、raw/gzipのbyte画像・ラベルデータセット。 |
| `data` | `ParseNumericCSV`、`ParseOptions`、`NumericTable`、`RegressionSamples`、`RegressionSample`、`RegressionBatch`、`StackRegression` | 数値のCSV/TSVテーブルと回帰の行。有限値、ヘッダとコメントの処理、特徴量とtarget列の指定、同じ長さのstacking。 |
| `data` | `Standardizer`、`FitStandardizer`、`StandardizerState`、`Apply`、`ApplyBatch`、`State`、`LoadState` | training splitでfitする決定的な特徴量の中心化とスケール。直列化でき、学習・検証・推論で同じ変換を共有します。 |
| `data` | `ReadWAV`、`ReadWAVFile`、`ReadWAVMono`、`WAV`、`Mono`、`WriteWAV`、`WAVEncoding` | 非圧縮のPCM/float RIFF音声（`WAVE_FORMAT_EXTENSIBLE`を含む）、チャネル平均のmono、例で使う書き出し。 |

`cuda.MemoryStats()`は、gogradのCUDAアロケータが持つ物理バイト数を返します。`LiveBytes`、`CachedBytes`、`ReservedBytes`、`PeakLiveBytes`、`PeakReservedBytes`、`CachedBuffers`、`CacheLimitBytes`です。既定のキャッシュ上限は2 GiBです。`SetPoolCacheLimit`は上限を下げたときに、最も長く使われていないキャッシュをすぐに解放します。使用中のテンソルと、記録したグラフが予約している割当は解放しません。`ResetAllocationPeak`はピークの値を現在の値に戻します。この値には、gogradのプールを通さずCUDAライブラリが直接確保した分は含まれません。`Buffer.Size()`は、物理的なサイズクラスが大きくても、要求した論理バイト数を返します。

CUDAのdevice、stream、captureの状態は、それを使うOSスレッドに属します。`SetDevice`と`CurrentDevice`でdeviceを選び、`autograd.NewCUDAContextOn`はgoroutineを検証済みのdeviceに固定し、`Close`でそのスレッドの以前のdeviceを復元します。`Buffer.Device()`は割当が属するdeviceを返します。再利用するプールの項目はdeviceとstreamとサイズクラスをキーにします。解放したブロックは同じstreamにだけ返すため、あるstreamがまだ読み取り中のブロックを別のstreamが再利用することはありません。`CopyFromHost`、`CopyToHost`、`Memset`/`MemsetAsync`、`Free`はバッファ自身のdeviceを選んで呼び出し側のdeviceを復元するため、複数deviceのコードは周囲のdeviceに依存しません。`SetCurrentStream`と`CurrentStreamHandle`は、呼び出したスレッドでカーネル起動とevent記録が使う暗黙のstreamを選びます。何も選ばないときは既定のstreamを使います。`CurrentStream`は選択中のstreamを返し、`WithStream(stream, fn)`はスコープの終了時（panic時も含む）に前の選択へ戻します。cuBLASのハンドルもこの暗黙のstreamに追従するため、NVRTCカーネル、memset、コピー、GEMMが同じstreamに乗ります。

1つの記録の中で処理を複数のstreamへ分けるには、`RunBranches`が使えます。分岐ごとにstreamと処理を渡すと、ライブラリが各分岐をそのstreamへenqueueし、呼び出したスレッドのカレントstreamが全分岐を待つようにeventを記録します。合流後は、結果を呼び出し側のstreamでそのまま消費できます。分岐の切り方は呼び出し側が決め、自動で分割はしません。

```go
// leftとrightは呼び出し側が用意したstreamです。
err := autograd.RunBranches([]autograd.StreamBranch{
    {Stream: left, Run: func() error { ya = branchA(x); return nil }},
    {Stream: right, Run: func() error { yb = branchB(x); return nil }},
})
// エラーを処理します。この時点でカレントstreamは両方の分岐を待ちます。
// yaとybを合流させ、Backwardとoptimizer.Step()まで通常どおり実行します。
```

`Event.Record`と`Stream.WaitEvent`／`Event.WaitOn`を使えば、合流点を自分で書くこともできます。あるstreamの結果を別のstreamが読むときは、作り手のstreamでeventを記録し、読み手のstreamが待ってから続きをenqueueします。割当プールはstreamごとに分かれているため、片方のstreamが解放したバッファをもう片方が再利用することはありません。

```go
// rightの完了をeventに記録し、leftがそれを待ってから合流します。
event, err := cuda.NewEvent(false)
// エラーを処理します。終わったらevent.Destroy()を呼びます。
_ = err
err = event.Record(right)
err = event.WaitOn(left)
// エラーを処理します。以降のleftの処理はrightの完了後に実行されます。
````Capture`は明示的なstreamを必要とし、現在のdeviceを変更せずに記録し、同じスレッドでの入れ子のcaptureを拒否し、記録したdeviceでグラフを再生します。したがって独立したスレッドは、それぞれ別の暗黙のstreamとcaptureの状態を保ちます。

## 実行コンテキストと勾配の記録

独立した実行には`ExecutionContext`を使います。コンテキストはテンソルが持ち、演算とviewが伝え、`Detach`が保ちます。goroutineごとの状態は使いません。ゼロ値のコンテキストは勾配を記録します。

```go
execution := autograd.NewExecutionContext()
model, err := autograd.NewLinearLayer(2, 3, tensor.CPU,
    rand.New(rand.NewSource(7)))
// エラーを処理します。
_ = err
err = execution.BindModule(model.StateModule())
// エラーを処理します。Forwardやworkerの起動より前に、全パラメータとバッファをbindします。
x, err := execution.New([]float32{1, 2}, []int{1, 2}, tensor.CPU, true)
// エラーを処理します。
_ = err
loss := autograd.Sum(model.Forward(x), 0, 1)
err = loss.Backward()
// エラーを処理します。グラフを解放する前に勾配を読みます。
loss.ReleaseGraph()

execution.NoGrad(func() {
    prediction := model.Forward(x)
    // prediction.RequiresGradはfalseで、ToHostで値を読めます。
    prediction.ReleaseGraph()
})
// コールバックがpanicしても、execution.Recording()は再びtrueになります。
```

- `BindModule`は登録済みのパラメータとバッファを再帰的に扱い、コンテキストを変える前に全ての末端を検証します。グラフを組む前にbindします。パラメータだけの枝（重みのtranspose、embeddingなど）も対象です。`Bind`は既存の末端を扱います。別のコンテキストの割当や、既に組んだグラフのbindは拒否します。パラメータとoptimizerの値は複製しません。
- 異なる2つの明示的コンテキストを組み合わせる演算はpanicします。bindしていない定数は、bind済みのグラフに加われます。完全に分離するには、勾配の要らない入力も含めて、全てのモデルstateと入力・マスクをそのコンテキストで作ります。
- `execution.NoGrad`はそのコンテキストだけに効き、安全に入れ子にできます。factoryの`requiresGrad`引数を保ち、既存のグラフを変えません。記録済みの損失に対するBackwardは、NoGradの中でも実行できます。記録していない定数のスカラーに対するBackwardは、CPUとCUDAのどちらでもエラーを返します。
- パッケージの`autograd.New`/`Zeros`と`autograd.NoGrad`は、bindしていないテンソル向けの互換APIとして残ります。パッケージのNoGradは共有の既定スコープを制御し、明示的コンテキストを無効にしません。明示的コンテキストを使う新しいコードは、パッケージ関数ではなくそのコンテキストのNoGradを呼びます。
- NoGradはコンテキスト全体を制御し、1つのgoroutineだけを対象にしません。並行して動かすworkerごとに、コンテキストと独立したモデルstateを用意します。共有するモデルでは、テンソルの値、勾配、optimizerのstate、train/evalの切り替えを並行して変更できません。bind自体は初期化の手順です。
- 推論の結果は寿命の親を保ち、`ReleaseGraph`がCPUの中間値とCUDAの補助バッファを回収できるようにします。導関数のコールバックは保ちません。グラフの回収後も値が必要な場合はretainしたハンドルを使います。`Detach`は独立したストレージを作り、履歴を切り、非連続なviewも含めてコンテキストの同一性を保ちます。

このコンテキストが分離するのは、勾配の記録、BF16モード、アテンションの選択です。CUDAのstreamやcapture、割当プールの寿命は持ちませんが、`SetCUDA(device, stream)`で使うdeviceとstreamを記録できます。記録した組は`WithCUDA(fn)`が呼び出したOSスレッドに適用し、`fn`の終了時（panic時も含む）に以前の選択へ戻します。CUDAを使う呼び出し側は、これまでどおり呼び出したOSスレッドで`NewCUDAContext`を保持します。`NewCUDAContextOn`はdeviceを選び、`Close`でそのスレッドの以前のdeviceを復元します。`cuda`パッケージは暗黙のstreamとcaptureの可否をOSスレッド単位で持つため、別々のコンテキストを持つgoroutineが互いのstreamを上書きしません。1回の記録の中で処理を複数のstreamへ分けることもできますが、分け方と`Event`を使った合流は呼び出し側が指定します。フレームワークが記録を自動で複数のstreamへ分割することはありません。`cmd/cnn-train`は明示的コンテキストのAPIを使います。

```go
stream, err := cuda.NewStream() // 呼び出し側が寿命を管理します
// エラーを処理します。終わったらstream.Destroy()を呼びます。
execution.SetCUDA(0, stream)
err = execution.WithCUDA(func() error {
    // このスコープのkernel、cuBLAS、コピーはstreamに乗ります。
    // forward、Backward、optimizer.Step()を通常どおり実行します。
    return nil
})
// エラーを処理します。スレッドのdeviceとstreamは前の選択に戻っています。
```

### 精度とアテンションの方針

明示的コンテキストの既定は、ゼロ値のコンテキストも含めて`ExecutionOptions{BF16Autocast: false, AttentionAlgorithm: "auto"}`です。パッケージレベルの`BF16Autocast`や`AttentionAlgorithm`は継承しません。`Options`はコピーを返すため、forwardの間に`SetOptions`で更新します。不正なアテンション名はコンテキストを変えずにエラーを返し、空の名前は`auto`に正規化します。

```go
err = execution.SetOptions(autograd.ExecutionOptions{
    AttentionAlgorithm: "materialized",
})
// エラーを処理します。上と同じように、モデルstateをbindし、executionで入力を作ります。
var mixedLoss *autograd.Tensor
execution.Autocast(true, func() {
    mixedLoss = autograd.Sum(model.Forward(x), 0, 1)
})
// 精度はFP32に戻ります。CUDAでは、直前のforwardで対象の演算が選んだBF16の方針を
// backwardも使います。
err = mixedLoss.Backward()
// エラーを処理します。
mixedLoss.ReleaseGraph()
```

- `Autocast(enabled, fn)`が変えるのは、このコンテキストのBF16フラグだけです。入れ子のscopeとpanicは以前の精度を復元し、アテンションの選択には影響しません。NoGradと同じく、1つのgoroutineではなくコンテキスト全体に効きます。独立したworkerには別々のコンテキストと、変更可能なモデルstateが必要です。
- 精度に影響する演算は、forwardのときに設定をスナップショットします。その後backwardは、scopeを抜けた後や`SetOptions`の変更後、互換用のグローバル設定の変更後でも、保持した履歴の再実行も含めてその方針を保ちます。記録したCUDAグラフも、記録時のカーネルと精度の選択を再生時に保ちます。選択を変えるには、新しく記録し直します。
- BF16はCUDAのGEMMオペランドとシャドウを使い、累積・出力・勾配・master weight・optimizerのモーメントはFP32です。CPUの計算と、Conv2dのように未対応のCUDA演算はFP32のままです。tiled flash attentionもFP32で計算します。
- アテンション名は`materialized`、`flash`、`auto`です。dropoutが有効でない対応可能なCUDAアテンションでは、`materialized`はGEMMとsoftmaxを、`flash`はtiled経路を使います（Q/KとVのhead次元は128まで）。`auto`は、実体化したときのスコアのscratchの見積もりが1 GiBに達するとtiledアテンションを選びます。CPU、dropoutが有効な場合、融合できないレイアウトは、演算を組み合わせた経路を使います。
- bindしていないテンソルは、パッケージレベルの互換設定を使います。このグローバル設定は共有なので、並行して変更しないでください。明示的コンテキストはこれを参照せず、bindしていない定数はコンテキストに束ねた演算へ加われます。コンテキストのオプションはforwardの合間に設定します。複数の演算からなるforwardの途中で変えても、そのforward全体が1つのスナップショットになるわけではありません。

## テンソルのストレージとグラフの寿命

テンソルとviewの割当は、グラフの辺とは別に参照カウントします。依存するviewや保存した計算がまだ使っている間は、ベースのハンドルをcloseしても割当はCPU/CUDAのプールへ戻りません。あるグラフの枝をreleaseしても、別の枝が使っている先祖は回収しません。これは明示的な寿命の管理で、放棄したグラフも`ReleaseGraph`/`Close`が必要です。

- `New`/`Zeros`が作る末端（入力、パラメータ、バッファ）は呼び出し側が所有します。`ReleaseGraph`は中間値と内部の一時的な末端を回収し、永続する入力とパラメータは`Close`が必要です。
- `Close`はそのハンドルの公開利用を終わらせ、繰り返し呼べます。割当と保存したstateは、依存するグラフの辺が終わるまで**内部で**読み取れます。生き残ったviewは読み取りやbackwardに使えますが、closeしたベースの`ToHost`、`GradToHost`、`CopyFrom`はエラーを返します。
- `RetainData`はforwardの値を固定し、`RetainGrad`は値と中間の勾配を固定します。どちらも導関数の履歴は保ちません。読み取った後に解放するには`Close`を使います。retainしたviewは、ベースのグラフを回収した後、共有する割当だけを保てます。
- `Backward`は導関数のコールバックと保存した統計を使います。CUDAでは未使用の中間値も自動的に回収し、CPUではforwardの値が`ReleaseGraph`まで残ります。retainしていないCUDAの出力は、backwardの前に読み取ります。
- `BackwardWithOptions(BackwardOptions{RetainGraph: true})`は、どちらのバックエンドでも履歴と保存した割当を保ち、もう一度実行できます。末端の勾配は加算され、中間のadjointは伝播のために新しくなります。要求したretain済みの勾配は、2度伝播せずに加算します。
- 消費した履歴を通す2度目のbackwardはエラーを返します。損失を1つのスカラーにまとめて1回backwardする場合、履歴の保持は不要です。retainした実行が終わったら`ReleaseGraph`が必要です。

```go
x := autograd.Must([]float32{2, 3}, []int{2}, true)
shared := autograd.Mul(x, x)
left := autograd.Sum(shared, 0)
right := autograd.Sum(autograd.MulScalar(shared, 3), 0)
err := left.BackwardWithOptions(autograd.BackwardOptions{RetainGraph: true})
// エラーを処理します。xの勾配は[4,6]です。
_ = err
err = right.Backward()
// エラーを処理します。加算後のxの勾配は[16,24]です。
left.ReleaseGraph()
right.ReleaseGraph()
x.Close()
```

マスク付き損失のtargetと、CUDAのembedding用indexバッファは、微分しないものの保存された寿命の依存です。呼び出し側が元のハンドルをcloseした後も、retainしたbackwardから使えます。CPUの正規化の統計とim2colのバッファも、履歴を使い切るか解放したときにだけプールへ戻ります。BF16のシャドウは共有のleaseを持ち、対応するFP32ストレージが変わると別名も含めて無効になります。

`CopyFrom`とeagerなAdamWの更新は、viewが共有するストレージのversionを進めます。Backwardは、勾配を変える**前に**記録したオペランドと出力のversionを検証し、forward以降の変更を拒否します。retainした実行の間の変更も拒否します。この検証は必要以上に厳しく、導関数によっては全てのオペランドを必要としないこともあります。CPUの`Data`への直接書き込みと、CUDAバッファへの外部からの書き込み・再生はversionを追跡しません。グラフが生きている間に、これらの経路で保存した値を変えないでください。重みの初期化はグラフを組む前に済ませます。

`Data`スライスと`Buffer()`は借用であり、独立した所有者ではありません。`Data`の差し替え、テンソルが所有するバッファのFree、ハンドルが終わった後の借用メモリの利用はしないでください。TensorとIndexBufferは所有状態をコピーせずポインタで保ちます。独立したCPUコンテキストは読み取り専用の定数を共有できますが、共有テンソルに対する並行の変更・close・backwardは対応していません。

## Conv2dとCNNの学習

`Conv2d(x, weight, bias, options)`は、`[batch,channels,height,width]`の入力にゼロパディング付きのcross-correlationを行います。重みはPyTorchと同じ`[out_channels,in_channels/groups,kernel_height,kernel_width]`のレイアウトです。biasは`nil`または`[out_channels]`のテンソルを渡します。カーネルの大きさは重みから決まり、矩形や偶数でもかまいません。

`Conv2dOptions`はheight/widthの順の`[2]int`の組を持ちます。

- `Stride`と`Dilation`: どちらも0の組は`[1,1]`が既定です。それ以外は両方の値が正でなければなりません。
- `Padding`: 各軸に対する非負の対称なゼロパディングです。既定はパディングなしのvalid convolutionです。
- `Groups`: 0は1が既定です。どちらのチャネル数もgroupsで割り切れる必要があります。`Groups == in_channels`はdepthwise畳み込みになり、出力チャネルのmultiplierも扱えます。

各軸の出力サイズは`floor((input + 2*padding - dilation*(kernel-1) - 1)/stride) + 1`です。deviceの組み合わせ、形状、オプションが不正な場合は演算の境界でpanicし、層のコンストラクタはエラーを返します。入力、カーネル、出力の各次元は正で、各テンソルの要素数はCUDAの符号付き32ビットのindex上限に収まる必要があります。非連続のviewも受け取り、実体化を通して勾配を受け取ります。

```go
device := tensor.CPU // NewCUDAContextを保持している間はtensor.CUDA
rng := rand.New(rand.NewSource(7))
conv, err := autograd.NewConv2dLayer(4, 8, [2]int{3, 5},
    autograd.Conv2dOptions{
        Stride: [2]int{1, 2}, Padding: [2]int{1, 2}, Groups: 2,
    }, true, device, rng) // trueでbiasを登録し、falseではBiasはnilのままです
// エラーを処理します。conv.StateModule()は"weight"と、biasがあれば"bias"を登録します。
_ = err
root := autograd.Module{Children: []autograd.NamedModule{
    {Name: "conv", Module: conv.StateModule()},
}}
x, err := autograd.New(make([]float32, 2*4*8*10), []int{2, 4, 8, 10}, device, true)
// エラーを処理します。
_ = err
y := conv.Forward(x) // [2,8,8,5]
loss := autograd.Mean(y, 0, 1, 2, 3)
err = loss.Backward() // 入力、重み、biasの勾配
// エラーを処理します。グラフを解放する前に出力と勾配を読みます。
loss.ReleaseGraph()
err = root.SaveSafeTensors("conv.safetensors")
// エラーを処理します。終わったら入力、パラメータ、optimizerが持つバッファをcloseします。
```

CPUとCUDAはどちらも、1バッチ1グループずつを空間タイルへ展開します。im2colと出力タイルの組は4 MiBまで（カーネルだけでこの予算を超える場合は1列まで）で、1タイルの出力位置は最大512です。Backwardは完全なim2col行列を保存せず、展開を再計算します。CPUは共有のSGEMMを、CUDAはcuBLASとデバイス側の展開・scatterカーネルをホストへの転送なしで使います。Conv2dはBF16 autocastが有効でもFP32で計算します。CUDAの入力勾配のscatterはatomic加算を使うため、bit単位ではなく数値として再現します。

`cmd/cnn-train`は、Conv2d/BatchNorm/ReLUの2ブロック、maxによるダウンサンプリング、adaptive global average pooling、線形分類器、cross-entropy、AdamWを、縦横のノイズ付きストライプ画像で動かします。保存時は新しいモデルを作り直し、全てのパラメータを厳密に再読込し、推論のlogitsを比較します（CUDAのreductionは和の順序が変わり得ます）。外部データセットもPythonも不要です。

```sh
go run ./cmd/cnn-train -device cpu -steps 60 -out out/cnn.safetensors
go run ./cmd/cnn-train -device cuda -steps 60 -out out/cnn-cuda.safetensors
```

`out`の親ディレクトリは先に作っておきます。これはアーキテクチャと学習の動作確認用で、画像認識の品質を測るものではありません。リポジトリに含めたConv2dのfixtureは、Goのテスト中にPythonを必要とせず、全ての出力とVJPの要素をPyTorchと照合します。作り直すには`python tools/gen_conv2d_fixture.py`を使います。

## NCHW pooling

`MaxPool2d`、`AvgPool2d`、`AdaptiveAvgPool2d`は、CPUまたはCUDAのfloat32の`[batch,channels,height,width]`入力を受け取ります。非連続のviewは実体化し、勾配を元のストレージへ戻します。入力と出力の各次元は正で、全要素数はConv2dと同じく符号付き32ビットのindexに収まる必要があります。不正なオプションや形状は、実体化や出力の確保より前にpanicします。

```go
down := autograd.MaxPool2d(x, autograd.MaxPool2dOptions{
    KernelSize: [2]int{2, 2}, // strideの既定も[2,2]です
})
smoothed := autograd.AvgPool2d(down, autograd.AvgPool2dOptions{
    KernelSize: [2]int{3, 3}, Stride: [2]int{1, 1},
    Padding: [2]int{1, 1}, ExcludePad: true,
})
pooled := autograd.AdaptiveAvgPool2d(smoothed, [2]int{1, 1})
features := autograd.Reshape(pooled, pooled.Shape[0], pooled.Shape[1])
// featuresは[batch,channels]です。LinearLayerと損失を組み合わせます。
```

- `MaxPool2dOptions`は`KernelSize`、`Stride`、`Padding`、`Dilation`の組と`CeilMode`を持ちます。`AvgPool2dOptions`は`KernelSize`、`Stride`、`Padding`、`CeilMode`、`ExcludePad`、`DivisorOverride`を持ちます。組はheight/widthの順です。`KernelSize`は必須です。strideが全て0なら`KernelSize`、max poolingのdilationが全て0なら`[1,1]`が既定です。その他のstrideとdilationの値は正でなければなりません。paddingは対称で、各軸で0からカーネルサイズの半分までです。average poolingのdilationは1です。
- 各軸の出力サイズは`floor((input + 2*padding - dilation*(kernel-1) - 1)/stride) + 1`です。`CeilMode`はfloorをceilに替え、その後で右端または下端のパディングだけで始まる最後の窓を除きます。窓は入力の外へはみ出せます。
- max poolingのパディングは負の無限大として扱います。最大値が同じときは、row-majorで最初の有効な要素を選びます。NaNは最後のNaNを選びます。選んだindexは勾配を記録するときだけ保存し、retainしたbackwardでも保ちます。dilationした窓に有効な要素がない場合は負の無限大を返し、入力の勾配を生みません。このAPIは値だけを返し、indexのテンソルは返しません。
- average poolingは既定でゼロパディングも除数に数え、PyTorchの`count_include_pad=true`と同じです。有効な入力位置だけを数えるには`ExcludePad: true`を設定します。ceil modeの部分的な窓では、要求したパディングを超える位置は既定の設定でも除外します。正の`DivisorOverride`は数えた個数を置き換え、0は通常の個数を使います。負の値は拒否します。
- adaptive poolingは正の`[output_height,output_width]`を必要とします。各binは`floor(position*input/output)`で始まり、`ceil((position+1)*input/output)`で終わります。binは重なり合い、出力サイズが入力サイズより大きくてもかまいません。backwardは全てのbinの寄与を加算します。
- `MaxPool2dLayer{Options: ...}`、`AvgPool2dLayer{Options: ...}`、`AdaptiveAvgPool2dLayer{OutputSize: ...}`は`Sequential`用の`TensorLayer`を実装します。パラメータ、バッファ、train/evalに依存するstateは持ちません。bindした入力のコンテキストの同一性とNoGradの動作は通常どおり伝わります。
- poolingはBF16 autocastの下でもFP32で計算します。CUDAはホストへの転送なしでデバイスカーネルを使い、captureと再生に対応し、重複する勾配をatomic加算で積みます。勾配はbit単位ではなく、数値で比較します。

リポジトリに含めたPyTorchのfixtureは、矩形カーネル、dilationしたmax pooling、ceil modeの補正、パディングを含める・除く場合、除数の上書き、割り切れないadaptive bin、adaptiveのアップサンプリングについて、全ての出力と入力のVJPを確認します。作り直すにはPyTorchが必要で（`python tools/gen_pool2d_fixture.py`）、GoのテストはJSONをそのまま使います。3つの演算は有限差分でも独立に確認します。

## BatchNorm

`BatchNorm(x, weight, bias, runningMean, runningVar, options)`は、チャネル軸以外の全ての次元を正規化します。入力はrank 2〜6で、チャネルは軸1（`NC`、`NCL`、`NCHW`など）か、`ChannelsLast: true`のときは最後の軸（`NC`、`NLC`、`NHWC`など）にあります。形状は保たれます。重みとbiasはそれぞれ独立に任意の`[channels]`ベクトルで、省略すると重みは1、biasは0になります。全てのテンソルは同じdeviceと、互換な実行コンテキストを共有する必要があります。

`BatchNormOptions`は`Training`、`ChannelsLast`、`Eps`、`Momentum`を持ちます。

- epsilonが0なら`1e-5`が既定です。それ以外は正で有限でなければなりません。
- 学習はチャネルごとに2サンプル以上を必要とします。サンプル数はチャネル以外の次元の積です。出力は偏りのあるバッチ分散（サンプル数で割る）を使います。running分散の更新は不偏推定量（サンプル数から1を引いて割る）を使います。
- running meanとvarianceは、両方nilであるか、両方とも連続した勾配の要らない`[channels]`バッファでなければなりません。nilでの学習はバッチ統計を使い、running stateを記録しません。評価は両方のバッファを必要とし、チャネルごとに1サンプルでもかまいません。
- runningの更新は`(1-Momentum)*old + Momentum*batch`で、momentumは`[0,1]`の有限値です。関数版でmomentumが0のときはrunningの値を凍結し、学習の正規化をrunning統計に切り替えるわけではありません。変更可能なrunningバッファは、他の入力やstateの割当と別名になってはいけません。
- Backwardは入力と任意のaffineパラメータを微分します。各forwardはmeanと逆標準偏差の割当を自分用に保存します。forwardの後にrunningバッファを更新・closeしたり、層のモードを変えたりしても、既に組んだグラフの導関数は変わりません。入力やaffineの管理された変更は、通常どおり保存したversionの検証を起こします。

層のAPIは、学習するパラメータと勾配の要らないstateをModuleへ登録します。

```go
norm, err := autograd.NewBatchNormLayer(8, autograd.BatchNormLayerOptions{}, device)
// エラーを処理します。オプションが全て0ならaffineと統計の追跡が有効になり、
// epsilon 1e-5、momentum 0.1、channels-first、学習モードになります。
_ = err
defer norm.Close()
root := autograd.Module{Children: []autograd.NamedModule{
    {Name: "norm", Module: norm.StateModule()},
}}
execution := autograd.NewExecutionContext()
err = execution.BindModule(&root)
// エラーを処理します。Forwardの前にexecutionで入力を作ります。
root.Train(true)
y := norm.Forward(x)
// 損失、backward、更新を組み合わせます。終わったらグラフを解放します。
_ = y
root.Train(false)
execution.NoGrad(func() {
    prediction := norm.Forward(x)
    prediction.ReleaseGraph()
})
err = root.SaveSafeTensors("norm.safetensors")
// エラーを処理します。同じ構成を作り直してstateを読み込み、Train(false)にします。
```

`BatchNormLayerOptions`は既定でaffineとrunning stateを使います。`DisableAffine`は重みとbiasを省きます。`DisableRunningStats`は全てのrunningバッファを省き、学習と評価の両方でバッチ統計を使います（学習のサンプル数要件も含みます）。`Momentum`は任意のポインタで、nilは`0.1`が既定、0を指すポインタはrunningの値を明示的に凍結します。`CumulativeMomentum: true`は`1/num_batches_tracked`を更新の重みに使い、観測した**バッチ**の平均に一致します（サンプルで重み付けした平均ではありません）。オプションは構築時にコピーするため、後から渡したmomentumの変数を変えても層は変わりません。

- パラメータは`weight`と`bias`、バッファは`running_mean`、`running_var`、スカラーの`num_batches_tracked`です。いずれもModule、safetensors、学習checkpointを通して保存・再読込します。バッファは`NamedParameters`とoptimizerの更新から除きます。現在のstate形式はバッチカウンタも含めてF32で、整数を正確に表せるのは`2^24`までです。PyTorchのI64カウンタとは形式が異なるため、PyTorchのstateを取り込むときは現在のF32形式へ変換します。
- `Train`と再帰的な`Module.Train`が統計を制御し、NoGradは導関数の記録だけを制御します。学習中のNoGradのforwardでもrunningの値とカウンタは更新します。追跡ありの評価では更新しません。モードと構成は呼び出し側が再現し、StateDictには保存しません。
- この層は`TensorLayer`、`Trainer`、`ParameterizedLayer`を実装します。`Close`はaffineパラメータとバッファを解放し、繰り返し呼べます。独立したworkerは別々のコンテキストと、変更可能な層のstateを使います。
- CPUとCUDAは、BF16 autocastの下でもFP32の出力と勾配を計算します。モーメントのreductionはfloat64で累積し、FP32の統計を保存します。CUDAのrunning更新とバッチ数の計上はデバイス上に留まり、累積平均も含めてグラフの再生に対応します。通常どおりモデルをbindし、`NewCUDAContext`を保持します。stateの確認や書き出しにだけ、明示的なホストへの読み戻しが必要です。

PyTorchのfixtureは、forward、入力とaffineの全てのVJP、偏りあり・なしの分散、channels-first/lastのレイアウト、rank 2〜6、サンプルが1つの評価、reductionの端を確認します。複数ステップの列では、固定・累積の更新、NoGradの学習、追跡を無効にした評価を追加で比較します。Goだけのテストは、有限差分、選択的な勾配、共有・retainした履歴、closeした先祖、独立したCPU worker、stateの再読込、CUDAのcaptureとメモリ回収を確認します。作り直すには`python tools/gen_batchnorm_fixture.py`を使います。

## OptimizerとSGD

`Optimizer`は`Step()`、`ZeroGrad()`、`LearningRate() float32`、`SetLearningRate(float32)`、`Close()`を公開します。AdamWとSGDがこれを実装し、`OneCycle.Step(optimizer)`はスケジュールを進めて次の更新の学習率をインターフェース経由で設定します。AdamWの学習率フィールドや直接の呼び出しはこれまでどおり使えます。`SetLearningRate`は有限で非負の値を受け取り、不正な入力ではpanicします。`CheckpointOptimizer`は`Snapshot`、`ValidateSnapshot`、`LoadSnapshot`を追加し、`adamw`または`sgd`のタグを持つバージョン付きの`OptimizerSnapshot`を扱います。AdamWの型付きのState/LoadStateはそのまま残ります。独自の更新を実装する場合もOptimizerとOneCycleを組み合わせられますが、現在のcheckpointのエンコーダは組み込みの2つに対応します。

```go
schedule := autograd.NewOneCycle(.1, 100, .3)
sgd, err := autograd.NewSGD(model.NamedParameters(), autograd.SGDOptions{
    LR: float32(schedule.LR()), Momentum: .9,
    Nesterov: true, WeightDecay: .0001,
})
// エラーを処理します。
_ = err
defer sgd.Close()
var optimizer autograd.Optimizer = sgd
optimizer.ZeroGrad()
// 損失を組み、Backwardしてから、通常どおりグラフを解放します。
optimizer.Step()
schedule.Step(optimizer)
err = autograd.SaveTrainingCheckpoint("run.safetensors", model, sgd, schedule, metadata)
// エラーを処理します。LoadTrainingCheckpointがモデル、オプション、バッファ、スケジュールを復元します。
```

`NewSGD(params, options)`は、構成やパラメータが不正なときにエラーを返します。パラメータは、同じdevice上の、名前が異なり連続したfloat32の末端で、要素数が正である必要があります。コンストラクタはパラメータのスライスとオプションをコピーするため、パラメータの同一性と順序を保ってください。学習率は`SGD.LR`フィールドでも参照できます。その他のハイパーパラメータは構成をコピーし、stateを通して復元します。

`SGDOptions`はLR、Momentum、Dampening、WeightDecay、Nesterov、Maximizeを持ちます。

- 数値は有限で非負です。学習率が0でもmomentumは更新します。Nesterovは正のmomentumとdampening 0を必要とします。momentumが0なら通常のSGDで、dampeningは効きません。
- `d = grad`（Maximizeでは`-grad`）から始め、coupled decayの`WeightDecay * parameter`を加えます。これはAdamWのdecoupled decayと異なります。
- momentumが正のとき、最初のバッファはdampeningなしの`d`そのものです。以降は`buffer = Momentum*buffer + (1-Dampening)*d`です。降下にはバッファを、Nesterovでは`d + Momentum*buffer`を使い、パラメータから`LR * update`を引きます。Maximizeは生の勾配の符号だけを反転し、decayはパラメータを縮めます。
- 勾配がないパラメータは、パラメータ、decay、バッファの更新を飛ばします。0の勾配は存在するものとして、decayとmomentumを適用します。遅延初期化はパラメータごとなので、後のステップで初めて勾配を受けたパラメータも、dampeningなしの最初のバッファを得ます。StepCountは、全勾配がないステップも含めてoptimizerのStep呼び出しを数え、スケジューラと揃います。
- eagerな`SGD.ZeroGrad`は、どちらのdeviceでも勾配を無い状態に戻します。準備済みまたは記録済みのCUDAグラフは、再生に安定したアドレスが必要なため、勾配の割当を保って0で埋めます。AdamWはこれまでのZeroGradの方針を保ちます。
- CPU/CUDAの更新は管理されたストレージのversionを進め、eagerな変更でBF16のシャドウを無効にします。SGDはautocastの下でもFP32で更新します。CUDAの更新、momentum、初期化はデバイス上に留まり、ステップごとの読み戻しはありません。`Close`はoptimizerのstateだけを解放し、モデルのパラメータは呼び出し側が持ちます。

`SGD.State`は順序付きの名前と形状、全てのオプション、ステップ数、momentumバッファ、パラメータごとのInitializedフラグを持ちます。使われていないパラメータのバッファは、スナップショットにもファイルにもありません。`LoadState`は全てのオプション、名前、形状、フラグ、バッファを検証してからstateを置き換えます。スナップショットは独立したコピーです。単体のSGDのsafetensorsは`gograd-sgd`のmetadataを使い、統合checkpointは`gograd-training-1`を保ったままoptimizerの種類をタグで示します。種類のタグがない既存のAdamWの学習ファイルはAdamWとして扱います。optimizerの種類が一致しない読込は、liveのモデル値を変える前に拒否します。

CUDAのcaptureは、AdamWと同じく`SGD.PrepareGraph`、`GraphStepCount`、`SyncGraphStepCount`を使います。パラメータごとのデバイス上のフラグが最初のmomentumの初期化を扱い、更新の完了後にフラグを立てるカーネルを続けます。記録した再生はデバイス上のステップ数を進めます。stateやcheckpointの書き出しはその数を読み、eagerのStepへ戻るときに同期します。記録したハイパーパラメータは記録時に固定され、stateの復元や割当のやり直しは、先に古いグラフを終わらせる必要があります。外部からの再生による書き込みは、ホスト側のストレージversionの追跡対象外です。

リポジトリに含めたPyTorchのSGD fixtureは、通常、decay、momentum、dampening、Nesterov、ascent、学習率の変更、遅れて来る勾配と無い勾配、0の勾配を扱います。Goのテストは、全てのパラメータとバッファの値、遅延stateの往復、checkpointの再開、旧AdamWの読込、保存したversionの検証、CUDAグラフとBF16キャッシュの動作を比較します。作り直すには`python tools/gen_sgd_fixture.py`を使います。

### パラメータグループ

`ParameterGroupSpec{Name, Params, LRScale, WeightDecay}`は、学習率スケールとweight decayを共有する1組のパラメータに名前を付けます。`NewSGDGroups`と`NewAdamWGroups`がこのspecを受け取り、平坦な`NewSGD`/`NewAdamW`は`default`という1つの暗黙のグループと同じです。

```go
specs, err := autograd.PartitionParameters(params, func(name string) string {
    if strings.HasSuffix(name, ".bias") {
        return "no_decay"
    }
    return "decay"
})
// エラーを処理します。
_ = err
for i := range specs {
    specs[i].WeightDecay = 1e-4
}
specs[0].WeightDecay = 0 // biasのグループは最初のラベルの0のdecayを保ちます
optimizer, err := autograd.NewSGDGroups(params, options, specs)
```

- specはパラメータを過不足なく分割します。全てのパラメータがいずれか1つのグループに属し、同じパラメータが2度現れず、名前は一意で空でなく、各グループは空でありません。コンストラクタの時点での違反はエラーを返します。
- `LRScale`はoptimizerの基準の学習率に掛かるため、`OneCycle`と`SetLearningRate`は1つの学習率から全てのグループを動かします。有限で非負である必要があり、0は基準の学習率をそのまま使うことを選びます。WeightDecayはそのグループ自身の絶対的なdecayです。どちらも`AdamWState.Groups`/`SGDState.Groups`に、グループのパラメータ名を持つ順序付きの`OptimizerGroupState`として記録します。
- `NewAdamWGroups(params, lr, specs)`は基準のdecayを取りません。`NewSGDGroups(params, options, specs)`はmomentum/dampening/Nesterov/Maximizeを共有し、`options.WeightDecay`が0でない場合は拒否するため、decayは常にグループごとに設定します。公開フィールドの`AdamW.WeightDecay`と、stateの基準の`WeightDecay`/`Options.WeightDecay`は、最初のグループのdecayを写します。
- 読込は、liveのstateに触れる前にグループ数、名前、membership、スケール、decayを検証します。グループ以前に書かれたstateはグループの一覧を持たず、スケールを持たない1つのグループのoptimizerにだけ読み込め、decayは保存された基準のハイパーパラメータから取ります。グループ化したstateは平坦なoptimizerが拒否し、その逆も同じです。checkpointは学習の意味を保ちます。
- CUDAはグループごとのステップサイズとdecayを、それぞれのカーネル引数として適用します。記録したグラフは記録時に基準の学習率とグループのスケールを焼き込むため、スケジューラが基準の学習率を変えた後はeagerの更新とずれます。

リポジトリに含めたパラメータグループのPyTorch fixtureは、異なる学習率とdecayを持つ2つのグループのSGDとAdamWを、勾配がない場合と0の勾配も含めて扱います。Goのテストは、CPUとCUDAで全てのパラメータ、momentumとmomentの値、グループの記録を比較し、stateの往復、レイアウトの再利用、統合checkpointの再開、スケジューラによるスケール、CUDAグラフの再生も確認します。作り直すには`python tools/gen_param_groups_fixture.py`を使います。

## 汎用のホストデータとバッチ処理

`data`パッケージはテンソルやバックエンドに依存しません。`Dataset[T]`は`Len() int`と`Get(context.Context, index) (T, error)`を公開し、sampleは任意のGoの型でかまいません。`SliceDataset[T]`はメモリ上のスライスを適合させます。`NewSubset`はindexを検証してコピーし、元のsampleはコピーしません。`Indices()`はコピーを返します。`RandomSplit(size, validationCount, seed)`はデータセット全体を覆う、重複しない学習と検証のindexを返します。反復とcheckpointの復元の間、データセットの長さ、内容、indexの同一性を保ってください。

```go
dataset := data.SliceDataset[data.FloatSample]{
    {Values: []float32{1, 2}, Shape: []int{2}, Label: 0},
    {Values: []float32{3, 4}, Shape: []int{2}, Label: 1},
}
sampler, err := data.NewIndexSampler(dataset.Len(), data.SamplerOptions{
    Shuffle: true, Seed: 7,
})
// エラーを処理します。
_ = err
loader, err := data.NewLoader[data.FloatSample, data.FloatBatch](
    dataset, sampler, data.Stack, data.LoaderOptions{BatchSize: 2},
)
// エラーを処理します。
_ = err
defer loader.Close() // 読み手が動いている途中で止めるときに必要です
for {
    batch, err := loader.Next(context.Background())
    if err == io.EOF { break } // このepochの終わりで、以降のepochの終わりではありません
    // その他のエラーを処理します。
    x, err := execution.New(batch.Values, batch.Shape, device, false)
    // エラーを処理します。batch.LabelsをCrossEntropyに渡します。xは呼び出し側がcloseします。
    _ = err
    _ = x
}
err = loader.NextEpoch() // indexを戻します。shuffleの順序は決定的に変わります
// エラーを処理します。
```

- `IndexSampler`の既定は昇順です。shuffleは`SamplerOptions.Seed`でseedした専用のPCGストリームを使い、パッケージのグローバルな乱数は使いません。`Peek(count)`は次に来るindexのコピーを返し、状態を進めません。`Advance(count)`は消費したindexを確定します。`NextEpoch`は次の順序を始め、未使用の現在の残りを捨てます。空のデータセットはすぐにEOFを返します。
- 既定では、`Loader.Next`はsampleを逐次取得してcollateし、キャンセルを確認してからsamplerを進めます。読込、collate、キャンセルのエラーはカーソルを変えずに残すため、再試行できます。コールバックのEOFは`UnexpectedEOF`に変換し、切り詰めた入力が正常なepochの終わりに見えるのを防ぎます。キャンセルはコールバックの境界で協調的に行い、`Dataset.Get`も自身のcontextに従う必要があります。
- `BatchSize`は正でなければなりません。最後のバッチは既定で小さくなります。`DropLast`はその不完全な末尾を消費してEOFを返します。samplerとデータセットのサイズの不一致や、データセットの長さの変化はエラーです。`Loader`と変更可能なsamplerは逐次使うため、独立したworkerには別々のsamplerとloaderが必要です。読み取り専用のデータセットは、その`Get`が並行呼び出しに対応していれば共有できます。
- `Stack`はsampleの形状が等しいことを検証し、値とラベルをコピーしてバッチの軸を前に付けます。スカラーも扱えます。`PadSequences`は末尾の次元が一致することを必要とし、最初のsampleの軸をバッチの最大か指定した`Length`まで埋めます。短い固定の`Length`には`Truncate: true`が必要で、暗黙には切り詰めません。`PaddingOptions.Value`は埋める値を選びます。`FloatBatch.Shape`は`[batch,time,...]`、`Lengths`は残した長さ、`Mask`は`[batch,time]`の平坦なboolスライスで、残した入力位置だけがtrueです。空の系列も有効で、全て空の自動バッチではtimeの大きさが0になります。
- 組み込みのfloatのcollateは、所有権を持つ新しいホストスライスを返します。sampleがデータセットのストレージを借用することがあり、独自のcollateは借用したsampleを変更してはいけません。バッチの変換とアップロードは呼び出し側の`ExecutionContext`で行います。整数のラベルとboolのマスクは、アプリがテンソル表現を選ぶまでホストのデータのままです。

### 数値のCSVテーブルと回帰のsample

`ParseNumericCSV(r, ParseOptions)`は、任意のコメント行とヘッダを持つ矩形の数値CSV/TSVテーブルを読みます。フィールドは前後の空白を除き、有限でなければなりません。最初の不正な行は1始まりの行番号を報告します。`NumericTable.RegressionSamples`は行を`RegressionSample{Features, Target}`へ分けます。特徴量は既定で最後の列以外の全てで、`targetColumn`に-1を渡すと最後の列を選びます。`StackRegression`は同じ長さのsampleを、`Values`/`Shape`が`[batch,features]`で`Targets`がスカラーの`RegressionBatch`へcollateします。`Stack`と同じ汎用の`Loader`にそのまま使えます。

```go
file, err := os.Open("data.csv")
// エラーを処理します。ファイルをcloseします。
table, err := data.ParseNumericCSV(file, data.ParseOptions{})
samples, err := table.RegressionSamples(nil, -1)
batch, err := data.StackRegression(samples[:16])
```

`Standardizer`は特徴量ごとの決定的な中心化とスケールの変換です。`FitStandardizer`はtraining splitの平均と母分散をfloat64で累積するため、統計がfloat32の和の順序に依存しません。定数の特徴量はスケール1になり、`Apply`の後で正確に0になります。`Apply`はコピーを返し、`ApplyBatch`はcollate済みのバッチをその場で標準化します。`State`/`LoadState`は統計をJSONで往復させます。これが実行の再現性を支える約束で、training splitでfitし、stateを学習checkpointに保存し、検証と推論で再利用します。`LoadState`は不正なversion、サイズの不一致、非有限または正でないスケールを拒否します。

`cmd/tabular-train`は、このテーブルを読むGoだけの回帰の例です（`-generate`で決定的な合成データを書きます）。`RandomSplit`で学習と検証に分け、standardizerは学習の行だけでfitし、小さなMLPを平均二乗誤差で学習します。standardizerのstate、loaderの順序、データセットのfingerprintを、AdamW/SGDとOneCycleのstateと共に保存するため、中断した実行は厳密に再開します。データ、分割、batch、steps、seed、hiddenの大きさ、optimizerの変更は、読み込む前に拒否します。時刻は同一性に含みません。

### WAV音声とaugmentation

`ReadWAV(r)`は非圧縮のRIFF音声を復号します。8/16/24/32ビットの整数PCM（8ビットは符号なし）と32/64ビットのIEEE float、`WAVE_FORMAT_EXTENSIBLE`のfmtチャンクを含みます。整数のsampleは`2^(bits-1)`で割り、floatは書かれたまま使います。一般的なPyTorchの読込と同じ扱いです。結果は`SampleRate`、`Channels`、`Bits`、`Float`、`Samples`（インターリーブ）、`Mono()`を公開します。`ReadWAVFile`と`ReadWAVMono`はファイルとチャネル平均の場合を扱います。圧縮コーデック、未知のレイアウト、フレーム数が整数にならないデータサイズは、推測せずに拒否します。

`WriteWAV(w, samples, WAVEncoding{Channels, SampleRate, Bits, Float})`は同じレイアウトを書き出します。整数の出力は表せる範囲にclampし、最も近い段へ丸めます。floatの出力は非有限のsampleを拒否します。したがって往復の誤差は、整数PCMでは最小段の1.5倍以内、floatでは誤差なしです。

```go
wav, err := data.ReadWAVFile("tone.wav")
// エラーを処理します。
mono := wav.Mono()
file, err := os.Create("copy.wav") // この後WriteWAV(file, wav.Samples, data.WAVEncoding{...})を呼びます
```

`cmd/audio-train`は音声の例です。monoの信号を固定長のwindowに分け、各windowへ`(seed, window index)`から専用のPCGストリームで導いたガウスノイズを加えます。反復の順序は使いません。これがアプリ側のaugmentationの約束で、worker、prefetch、再開は入力を変えられず、seedはcheckpointの同一性なので、中断した実行は同じノイズの入力と同じ重みを再現します。例はMLPのdenoiserを平均二乗誤差で学習し、入力と出力のSNRをdBで表示し、モデル、optimizer、スケジュール、loaderの順序、WAVのfingerprintをまとめてcheckpointに保存します。

### データのcheckpoint

`IndexSampler.State`はversion、サイズ、seedとshuffleの設定、epoch、現在位置、完全な順序、PCGのstateを記録します。復元はstateを変える前に構成、順列、乱数のバイト列を検証します。したがって再開は、現在のepochの残りと以降のshuffleされたepochの両方を保ちます。

`Loader.State() (LoaderState, error)`は、データセットのサイズ、batchサイズ、DropLastを、不透明なsamplerのpayloadに加えます。samplerは`StatefulSampler`（MarshalBinary/UnmarshalBinary）を実装する必要があり、`IndexSampler`は実装しています。`Loader.LoadState`は互換でない構成を、samplerを復元する前に拒否します。stateのスナップショットは順序とpayloadをコピーし、学習checkpointの呼び出し側metadataにJSONで入れられます。データセットの内容、subsetやsplitの同一性、独自のcollateの構成は呼び出し側のmetadataです。長さが一致するだけでは、同じデータであることの証明になりません。

### 並列readerと上限付きprefetch

`LoaderOptions.Workers`に正の数を設定すると、ホスト側のreaderのgoroutineが有効になります。`Workers=0`は同期経路を保ちます。`Prefetch`は現在のバッチも含めて、未処理のバッチ数を制限します。0は並列時に2が既定で、同期時は0のままにします。負の値や、容量が溢れる設定はエラーです。readerの数は、利用できるsampleの枠までに制限します。`Dataset.Get`は並行呼び出しに対応し、collateまで有効なsampleストレージを返す必要があります。

```go
loader, err := data.NewLoader[data.FloatSample, data.FloatBatch](
    dataset, sampler, data.Stack,
    data.LoaderOptions{BatchSize: 64, Workers: 4, Prefetch: 3},
)
// エラーを処理します。indexは先読みしますが、渡す順序はsamplerの順です。
_ = err
defer loader.Close()
```

- workerが呼ぶのは`Dataset.Get`だけです。collateは`Next`の呼び出し元のgoroutineで直列に行い、先読みしません。テンソルの生成とCUDAのスレッドの所有権は、受け取る側に留まります。未処理のjob、結果、sampleのバッチは、設定したprefetchの枠までに制限し、完了した順序は渡す順序を変えません。
- 失敗したバッチは、どのreaderが先に終わったかではなく、そのバッチのsampler順で最初に失敗したsampleを報告します。読込やcollateのエラーは未処理の先読みを捨て、readerをjoinし、確定済みのカーソルを再試行のために保ちます。worker内のデータセットのpanicは順序付きの読込エラーになり、collateのpanicはreaderの後片付けの後に受け取る側で再びpanicします。
- readerは、その作業を始めた`Next`呼び出しのcontextを引き継ぎます。`Next`の間のキャンセルはreaderをキャンセルしてjoinし、そのバッチは未消費のままです。呼び出しの合間に以前の作業のcontextがキャンセルされていても、生きているcontextでの次の`Next`は確定済みのカーソルからやり直します。collateの後のキャンセル確認は、indexを確定する前に起きます。`Dataset.Get`はキャンセルに協調する必要があり、後片付けは進行中のコールバックの終了を待ちます。
- `Close() error`は繰り返し呼べ、readerをキャンセルしてjoinし、未配達のsampleを捨てます。EOFと最後に配達したバッチも、終了したreaderを解放します。`NextEpoch`はindexの順序を変える前に古いreaderを止めます。メソッドは`Close`/`State`/`LoadState`も含めて逐次で、`Next`のcontextをキャンセルしてブロックした呼び出しを中断します。別の`Loader`のメソッドを並行して呼ばないでください。
- `State`はreaderを止めて先読みを捨ててから、**配達済み**のsampler位置を直列化します。`LoadState`も同様に、古い読込を止めてから復元します。WorkersとPrefetchは論理的な`LoaderState`に含まれないため、既存の同期のcheckpointは、並列readerや別のreader数で再開できます。batchサイズ、DropLast、データの同一性、独自のcollateやaugmentationのstateは、元の約束のままです。反復中はsamplerをloaderが所有するものとして扱い、先読みをかかえたまま別途進めたり戻したりしないでください。

`Dataset.Get`の副作用と、呼び出しに依存する乱数は、`Loader`が巻き戻したりcheckpointに保存したりしません。再現できるprefetchには、決定的なsampleの取得（またはアプリが管理するsampleごと・epochごとの乱数）が必要で、独自のcollateのstateは呼び出し側のmetadataに置きます。

### raw/gzip IDXとMNISTの例

`OpenIDX(imagesPath, labelsPath)`はMNIST形式のbyte IDX（magic 2051/2049）を読み、次元、個数、payloadの正確な長さを検証し、ファイル名の拡張子によらずrawまたはgzipのデータを受け取ります。データセットはコンパクトなuint8の画像とラベルをメモリに保ちます。`Get`は所有権を持つ、`[0,1]`に正規化したfloat32の`[1,height,width]`画像と、その整数ラベルを返します。`ImageSize`と`Classes`は幾何と最大ラベル+1を返します。`Fingerprint`は復号した両ファイルのSHA-256なので、gzipのコンテナのバイト列だけを変えても同一性は変わりません。

`cmd/mnist-train`は784→64→10のMLP、cross-entropy、AdamWまたはSGD、OneCycleを使います。MNISTの学習・テストのIDXの組を読み、学習はshuffleしたバッチ、テストは昇順の評価を使い、推論用の重みを保存してから、作り直して再読込し、logitsを比較します。ダウンロードは任意で、GoのHTTPとgzipで行います。公開ミラーから取得するのは不足しているファイルだけで、代わりにローカルのファイルを用意できます。通常のテストは小さなローカル生成のIDXとHTTPのテストサーバーを使い、ネットワークもPythonも必要としません。

```sh
go run ./cmd/mnist-train -download -device cpu -steps 100 -stop-after 40 -train-limit 2000 -valid-limit 1000
go run ./cmd/mnist-train -device cpu -steps 100 -train-limit 2000 -valid-limit 1000 -resume out/mnist.safetensors.training.safetensors
go run ./cmd/mnist-train -device cuda -steps 100 -train-limit 2000 -valid-limit 1000 -out out/mnist-cuda.safetensors
go run ./cmd/mnist-train -device cpu -workers 4 -prefetch 3 -steps 100 -train-limit 2000 -valid-limit 1000 -out out/mnist-prefetch.safetensors
go run ./cmd/mnist-train -device cpu -optimizer sgd -momentum .9 -nesterov -steps 100 -train-limit 2000 -valid-limit 1000 -out out/mnist-sgd.safetensors
```

データの既定のディレクトリは`out/mnist`、出力は`out/mnist.safetensors`、batchサイズは64、seedは7です。ディレクトリは必要に応じて作ります。`TrainLimit`/`ValidLimit`は、それぞれの公式の分割の先頭からN件を選びます（0は全て）。`StopAfter`は完了した更新の絶対数で、`Steps`は計画したスケジュール全体を表し、再開時に変えてはいけません。checkpointの既定は出力に`.training.safetensors`を付けたもので、推論と学習のパスは別にする必要があります。`-workers`と`-prefetch`は学習と評価の両方の読込を制御し、既定は0（同期）です。checkpointは配達済みのデータ順だけを記録するため、再開時に変えられます。コマンドのloaderは、完了時もエラー時もcloseします。

`-optimizer adamw|sgd`は更新を選びます。`-lr`はOneCycleのピークを選び（0ならAdamWは.005、SGDは.1が既定）、`-momentum`はSGDで.9が既定、`-nesterov`はNesterov版を有効にします。この例のcoupled/decoupled decayは.0001です。optimizerの選択、学習率、momentum、Nesterovはcheckpointの同一性で、再開時の変更は拒否します。古いMNISTのAdamWのmetadataは、元の.005のピークと既定の設定として認識します。`-no-decay-suffix`（既定は`bias`、空で無効）は、一致するパラメータをdecayなしのグループへ移します。これもcheckpointの同一性で、別の値で書かれたcheckpointの再開は拒否します。グループ以前に書かれたcheckpointはsuffixを記録していないため、再開には空の値が必要です。

学習のstateには、モデル、AdamWのモーメントまたはSGDのmomentumとオプション、OneCycleとloaderのstate、データセット全体のfingerprint、実効のsubsetサイズ、seed、batchの構成を含みます。これらの同一性の変更は、liveのモデルstateを読み込む前に拒否します。中断は、キャンセルを返す前に最後に完了した更新のcheckpointを保存します。テストは、中断なしの場合とepoch途中で再開した場合の重み、モーメント、スケジュール、データ順を比較します。CPUの値は厳密に一致し、CUDAの浮動小数点は、既存のbias勾配のカーネルがatomic加算を使うため、数値として比較します。

## 学習の再開

```go
sampler, err := autograd.NewWindowSampler(frameCounts, 400, 16, 0.15, 0)
// エラーを処理します。返されたwindowを選んだdeviceのテンソルへ集めます。
windows := sampler.Batch()
_ = windows
sampleJSON, err := json.Marshal(sampler.State())
// エラーを処理します。optimizerの更新とscheduler.Step(opt)の後にだけ保存します。
err = autograd.SaveTrainingCheckpoint("training.safetensors", &model.Module,
    opt, schedule, map[string]string{"sampler": string(sampleJSON)})
// エラーを処理します。モデル、optimizer、スケジューラは先に構築しておきます。
metadata, err := autograd.LoadTrainingCheckpoint("training.safetensors",
    &model.Module, opt, schedule)
// エラーを処理します。次のバッチをサンプリングする前にstreamを復元します。
var state autograd.WindowSamplerState
err = json.Unmarshal([]byte(metadata["sampler"]), &state)
// エラーを処理します。
err = sampler.LoadState(state)
// エラーを処理します。学習の終わりにopt.Close()が自身のバッファを解放します。
```

パラメータの順序、名前、形状は一致する必要があります。読込は学習率、decay、beta、epsilonを復元し、新しい既定を黙って使うことはありません。モデルのモード、dropoutのseed、データセットの同一性、samplerのstateは呼び出し側の責任です。アプリケーション側がこれらを記録・検証し、決定的なdropoutのseedには完了したステップ数を使います。`OneCycle`は統合checkpointが直列化し、推論用やoptimizer単体のファイルは完全な学習checkpointではありません。CUDAグラフでは、保存の前に未完了のlaunchを同期し、新しいグラフを記録する前にstateを復元します。optimizerのスナップショットは、古いホストのステップ数ではなく、記録したデバイス上のカウンタを読みます。

## 互換性の注意

- 一般のautogradの演算、CUDAグラフの記録、BF16 autocast、デバイスバッファのプール、opレジストリは実験的な位置づけです。deviceとdtypeの扱いはfloat32を中心に整えており、BF16は任意のシャドウバッファを使います。
- 互換用の`NoGrad`スコープは、bindしていないテンソルが共有します。CPUで並行実行するには、別々の明示的コンテキストと独立したモデルstateを使います。bindしていないテンソルはプロセス共有のBF16・アテンション設定を使うため、このグローバル設定を並行して変更しないでください。明示的コンテキストはこれらの設定を持ちます。CUDAのstreamとcaptureの状態はOSスレッド単位で、各スレッドは自分のcuBLASハンドル、勾配クリップのプラン、プールのキャッシュを持ちます。割当プールはdeviceとstreamをキーにするため、`NewCUDAContextOn`のコンテキストを持つ独立したgoroutineは、暗黙のstream、capture、device、再利用するバッファを互いに共有しません。1回の記録の中での複数のstreamの重ね合わせと、複数deviceを1つの実行コンテキストで扱うことには対応していません。
- 統合checkpointはAdamWまたはSGDのoptimizerとOneCycleのstateを復元します。旧`nn`/`gputcn`の学習器は、これまでどおりのcheckpointの経路を使います。サンプリングはホストで行い、`data`パッケージはprefetchを任意で備えた汎用のloaderを別に提供します。
- safetensorsの書き込みは、同じディレクトリの一時ファイル、sync、renameを使います。読込は不正またはオーバーフローする形状、重複する範囲、16 MiBを超えるヘッダを拒否します。ファイル全体をメモリに読み込むため、ファイルサイズの上限はありません。deviceへの転送の失敗は、検証済みの読込を途中で止めることがあります。
- GoとPyTorchの学習は、サンプリング、初期化、dropoutに別々の乱数ストリームを使います。演算ごとの照合には初期重みを固定し、明示的なdropoutマスクを使います。学習全体の検証は、同じ軌跡ではなく、測定した品質で比較します。
