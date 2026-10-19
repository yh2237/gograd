# 学習API

CPU/CUDAで使う学習APIです。学習checkpointはsafetensorsの`gograd-training-1`形式で保存します。Vulkan/DirectMLの推論は[GPUバックエンド](GPU_BACKENDS.md)にあります。コード断片ではエラー処理を省略しています。

## 学習で使うAPI

| パッケージ | API | 用途 |
| --- | --- | --- |
| `torchpth` | `Read` | PyTorch zipのtensor / state_dictをfloat32で読む |
| `tensor` | `Device`、`CPU`、`CUDA` | 構築時のdevice選択 |
| `autograd` | `NewCUDAContext`、`NewCUDAContextOn`、`CUDAContext.Device`、`Close` | OSスレッド固定、CUDA deviceの選択と復元 |
| `autograd` | `New`、`Zeros`、`ToHost`、`Backward`、`ReleaseGraph`、`Close` | テンソル、勾配、寿命管理 |
| `autograd` | `RetainData`、`RetainGrad`、`BackwardWithOptions`、`BackwardOptions` | 値・勾配・履歴の保持 |
| `autograd` | `NewExecutionContext`、`New`、`Zeros`、`Bind`、`BindModule`、`NoGrad`、`Recording`、`Tensor.ExecutionContext` | 記録スコープの分離 |
| `autograd` | `ExecutionOptions`、`SetOptions`、`Options`、`Autocast` | BF16とアテンションの選択 |
| `autograd` | `ExecutionContext.SetCUDA`、`CUDA`、`WithCUDA` | device / streamの記録と一時適用 |
| `autograd` | `NewEmbeddingLayer`、`NewConv1dLayer`、`NewLayerNormLayer`、`NewLinearLayer`、`DropoutLayer`、`GELULayer`、`Sequential` | グラフモジュール。Conv1dは入力`[batch,time,channels]`、重み`[out,in,kernel]` |
| `autograd` | `Conv2d`、`Conv2dOptions`、`NewConv2dLayer`、`Conv2dLayer` | grouped / depthwiseを含むNCHW畳み込み |
| `autograd` | `MaxPool2d`、`AvgPool2d`、`AdaptiveAvgPool2d`、各Options / Layer | NCHW pooling |
| `autograd` | `BatchNorm`、`BatchNormOptions`、`NewBatchNormLayer`、`BatchNormLayerOptions` | 正規化とrunning state |
| `autograd` | `Module.StateDict`、`LoadStateDict`、`SaveSafeTensors`、`SaveSafeTensorsMetadata`、`LoadSafeTensors` | 名前付きF32 state。metadataの値は文字列 |
| `autograd` | `Module.Train`、`SetTraining`、`OnTrainingChange` | 子モジュールへのtrain/eval伝播 |
| `autograd` | `TrainingLoop[B]`、`BatchReader[B]`、`PositionSampler`、`UpdateSchedule`、`OneCycle.Updates` | 更新・報告・停止時保存 |
| `autograd` | `GradAccumulator`、`NewGradAccumulator`、`Backward`、`Pending`、`Reset`、`Groups` | micro-batchの勾配蓄積 |
| `autograd` | `MSELoss`、`L1Loss`、`BCEWithLogitsLoss`、`LossOptions`、`Reduction` | mean / sumと要素ごとの重み付き損失 |
| `autograd` | `MaskedLoss(pred,target,false)`、`ClipGradNorm`、`NewAdamW`、`NewOneCycle` | NaNフレームをマスクしたL1、クリップ、AdamW、OneCycle |
| `autograd` | `Optimizer`、`CheckpointOptimizer`、`OptimizerSnapshot`、`LearningRate`、`SetLearningRate` | 更新・学習率・checkpointの共通API |
| `autograd` | `AdamW.State`、`LoadState`、`SaveSafeTensors`、`LoadSafeTensors`、`Close` | モーメント・設定・ステップの保存と復元 |
| `autograd` | `NewSGD`、`SGDOptions`、`SGD.State`、`SGDState`、`LoadState`、`SaveSafeTensors`、`LoadSafeTensors` | momentum / Nesterov / decay / ascentとstateの復元 |
| `autograd` | `ParameterGroupSpec`、`PartitionParameters`、`NewSGDGroups`、`NewAdamWGroups`、`OptimizerGroupState` | グループごとの学習率スケールとdecay |
| `autograd` | `SaveTrainingCheckpoint`、`LoadTrainingCheckpoint`、`TrainingCheckpointMetadata` | model / optimizer / OneCycle / metadataの一括保存 |
| `autograd` | `IndexLoader.State`、`LoadState`、`WindowSampler.Batch`、`State`、`LoadState` | epoch順・PCG windowの再開 |
| `autograd` | `StreamBranch`、`RunBranches` | 独立したCUDA分岐のenqueueとeventによる合流 |
| `cuda` | `SetCurrentStream`、`CurrentStream`、`CurrentStreamHandle`、`WithStream` | 呼出しスレッドのstream選択 |
| `cuda` | `Event`、`NewEvent`、`Record`、`Stream.WaitEvent`、`Event.WaitOn`、`ElapsedTime` | stream間の順序付けと計時 |
| `cuda` | `MemoryStats`、`ResetAllocationPeak`、`SetPoolCacheLimit`、`ReleasePool` | 割当量・ピーク・キャッシュ上限 |
| `data` | `Dataset[T]`、`SliceDataset`、`NewSubset`、`RandomSplit` | sample取得と分割 |
| `data` | `Sampler`、`StatefulSampler`、`NewIndexSampler`、`IndexSamplerState` | index選択、shuffle、途中再開 |
| `data` | `NewLoader[T,B]`、`LoaderOptions`、`LoaderState`、`CollateFunc`、`Close` | バッチ処理と上限付きprefetch |
| `data` | `FloatSample`、`FloatBatch`、`Stack`、`PadSequences`、`PaddingOptions`、`OpenIDX` | ホストバッチ、系列長・マスク、画像データ |
| `data` | `TransformDataset[T]`、`RandomHorizontalFlipFloat32`、`FlipsHorizontallyFloat32`、`RandomVerticalFlipFloat32`、`FlipsVerticallyFloat32`、`AddGaussianNoiseFloat32` | seed / indexに基づくaugmentation |
| `data` | `ParseNumericCSV`、`ParseOptions`、`NumericTable`、`RegressionSamples`、`RegressionSample`、`RegressionBatch`、`StackRegression` | 数値CSV / TSVと回帰バッチ |
| `data` | `Standardizer`、`FitStandardizer`、`StandardizerState`、`Apply`、`ApplyBatch`、`State`、`LoadState` | 学習データの統計による標準化 |
| `data` | `ReadWAV`、`ReadWAVFile`、`ReadWAVMono`、`WAV`、`Mono`、`WriteWAV`、`WAVEncoding` | 非圧縮RIFF音声の入出力 |
| `metrics` | `Mean`、`Accuracy`、`Regression`、各型の`Count` / `Reset` | ホスト側の損失平均・正解率・回帰誤差 |

CUDAのdevice・暗黙のstream・capture状態はOSスレッド単位です。処理の間は`NewCUDAContext`を保持します。`NewCUDAContextOn`はdeviceを検証して選択し、`Close`で以前のdeviceへ戻します。

割当プールはdevice・stream・サイズクラスで分離します。`Buffer.Device()`は割当先、`Buffer.Size()`は論理バイト数を返します。コピー・memset・freeはバッファ自身のdeviceを選び、呼出し側のdeviceを復元します。カーネルとcuBLASは選択中のstreamに従い、`WithStream`はpanic時も以前の選択へ戻します。

`MemoryStats`は`LiveBytes`、`CachedBytes`、`ReservedBytes`、`PeakLiveBytes`、`PeakReservedBytes`、`CachedBuffers`、`CacheLimitBytes`を返します。キャッシュ上限は既定で2 GiBです。上限を下げると古いキャッシュから解放しますが、使用中とCUDAグラフが予約した割当は解放しません。CUDAライブラリ独自の割当は含みません。`ResetAllocationPeak`はピークを現在値へ戻します。

独立した分岐は呼出し側が指定します。`RunBranches`は各streamへ処理をenqueueし、カレントstreamが全分岐を待つeventを記録します。

```go
err := autograd.RunBranches([]autograd.StreamBranch{
    {Stream: left, Run: func() error { ya = branchA(x); return nil }},
    {Stream: right, Run: func() error { yb = branchB(x); return nil }},
})
```

手動の合流には`Event.Record`と`Stream.WaitEvent` / `Event.WaitOn`を使います。`Event.WaitOn`は待たせる側のstreamを指定します。streamはそのstreamのバッファを解放するまで保持してください。

`Capture`は明示的なstreamが必要で、同じスレッドでの入れ子を拒否します。現在のdeviceを変えずに記録し、記録したdeviceで再生します。

## 実行コンテキストと勾配の記録

`ExecutionContext`は入力とモデルstateに結び付く記録スコープです。演算・view・`Detach`が継承し、異なる明示的コンテキストを組み合わせる演算はpanicします。ゼロ値は勾配を記録します。

```go
execution := autograd.NewExecutionContext()
model, err := autograd.NewLinearLayer(2, 3, tensor.CPU,
    rand.New(rand.NewSource(7)))
err = execution.BindModule(model.StateModule())
x, err := execution.New([]float32{1, 2}, []int{1, 2}, tensor.CPU, true)
loss := autograd.Sum(model.Forward(x), 0, 1)
err = loss.Backward()
loss.ReleaseGraph()

execution.NoGrad(func() {
    prediction := model.Forward(x)
    prediction.ReleaseGraph()
})
```

入力・モデル・optimizerの所有者が各リソースを解放します。

- `BindModule`は登録済みのパラメータとバッファを再帰的に検証してbindする。重みのtransposeなど、パラメータだけの部分式を含め、グラフを組む前に行う。既存のグラフや別コンテキストの割当のbindは拒否し、値は複製しない。
- bindしていない定数はbind済みの演算へ加われる。完全に分離するには、勾配の要らない入力とマスクも含め、モデルstateと入力を同じコンテキストへ結び付ける。
- `NoGrad`はコンテキスト全体に効く。入れ子とpanicから復帰し、factoryの`requiresGrad`や既存グラフは変えない。記録済み損失のBackwardはNoGrad内でも実行できるが、未記録の定数スカラーのBackwardはエラー。
- パッケージの`New` / `Zeros` / `NoGrad`はbindしていないテンソル用の互換API。共有の既定スコープを使い、明示的コンテキストには作用しない。
- 並行workerには独立したコンテキストと変更可能なモデルstateを用意する。同じモデルの値・勾配・optimizer state・train/evalを並行して変更しない。

コンテキストはストレージ、stream、capture、割当プールの寿命を管理しません。CUDAではOSスレッドを固定したうえで`SetCUDA` / `WithCUDA`を使います。

```go
stream, err := cuda.NewStream()
execution.SetCUDA(0, stream)
err = execution.WithCUDA(func() error {
    // kernel、cuBLAS、コピーをこのstreamで実行する。
    return nil
})
// streamを使う全バッファの解放後にstream.Destroy()を呼ぶ。
```

`WithCUDA`は終了時とpanic時に以前のdevice / streamへ戻します。1回の記録を複数streamへ分ける場合も、分割と合流は呼出し側で指定します。

### 精度とアテンションの方針

明示的コンテキストの既定は`ExecutionOptions{BF16Autocast: false, AttentionAlgorithm: "auto"}`です。互換用のグローバル設定を継承しません。`Options`はコピーを返します。`SetOptions`は不正なアテンション名なら状態を変更せずエラーを返し、空の名前を`auto`に正規化します。

```go
err = execution.SetOptions(autograd.ExecutionOptions{
    AttentionAlgorithm: "materialized",
})
execution.Autocast(true, func() {
    loss = autograd.Sum(model.Forward(x), 0, 1)
})
err = loss.Backward()
loss.ReleaseGraph()
```

`Autocast`はBF16フラグだけを一時変更し、入れ子・panicの後に復元します。forwardが保存した精度とアテンションの方針は設定変更後やretainしたBackwardでも保ちます。CUDAグラフも記録時のカーネルを再生するため、選択を変えるには記録し直します。

BF16はCUDA GEMMのオペランドとシャドウに使い、累積・出力・勾配・master weight・optimizerのモーメントはFP32です。CPU、Conv2dなどの未対応演算、tiled flash attentionはFP32のままです。

| アテンション名 | CUDAでの選択 |
| --- | --- |
| `materialized` | GEMMとsoftmax |
| `flash` | Q/KとVのhead次元128までのtiled経路 |
| `auto` | スコアscratchの見積もりが1 GiB以上ならtiled、それ未満なら実体化 |

CPU、dropout有効、融合できないレイアウトは合成経路を使います。オプションはforwardの合間に設定してください。複数演算からなるforward全体が1つのスナップショットになるわけではありません。互換用のグローバル設定を並行して変更しないでください。

## テンソルのストレージとグラフの寿命

ストレージの参照数とグラフの依存数は別に管理します。ベースをCloseしても、viewや保存した計算が必要とする割当は保持します。ある枝のReleaseGraphが別の枝の先祖を回収することはありません。GoのGCによるグラフ回収は行いません。

| 操作 | 保持・解放するもの |
| --- | --- |
| `New` / `Zeros` | 呼出し側が所有する入力・パラメータ・バッファを作る |
| `Close` | そのハンドルの公開利用を終了する。繰り返し呼べる |
| `ReleaseGraph` | 中間値と内部の一時的な末端を回収する。永続する末端はCloseが必要 |
| `RetainData` | 値と共有する割当を保持する。履歴は保持しない |
| `RetainGrad` | 値と中間の勾配を保持する。履歴は保持しない |
| `Detach` | 独立したストレージへコピーし、履歴を切る。非連続viewとコンテキストも扱う |
| `BackwardWithOptions(RetainGraph: true)` | 履歴と保存した割当を保持し、再度Backwardできる |

Close済みのハンドルの`ToHost` / `GradToHost` / `CopyFrom`はエラーですが、依存するviewやbackwardは内部の値を使えます。retainしたハンドルも読み終えたらCloseしてください。

通常のBackwardは履歴を消費し、CUDAの未使用中間値を回収します。retainしていないCUDA出力はBackward前に読み取ります。CPUのforward値はReleaseGraphまで残り、消費済み履歴の再利用はエラーです。retainした実行では末端の勾配を加算し、中間のadjointは伝播用に作り直します。実行後はReleaseGraphしてください。

マスクのtarget、embeddingのindex、正規化の統計、im2colなども必要な履歴が終わるまで保持します。BF16シャドウはFP32ストレージの変更で別名も含めて無効になります。

`CopyFrom`とeagerなoptimizer更新はストレージのversionを進めます。Backwardは勾配を書き込む前に、保存したオペランドと出力のversionを検証します。CPUの`Data`への直接書込、外部からのCUDA書込・再生は追跡しないため、グラフが生きている間は保存値を変更しないでください。

`Data`と`Buffer()`は借用です。Dataの差替え、Tensor所有バッファのFree、ハンドル終了後の借用メモリ利用はしないでください。TensorとIndexBufferは所有状態をコピーせずポインタで保持し、共有テンソルの変更・Close・Backwardを並行して行わないでください。

## Conv2dとCNNの学習

`Conv2d(x, weight, bias, options)`はCPU/CUDAのNCHW cross-correlationです。入力は`[batch,channels,height,width]`、重みは`[out_channels,in_channels/groups,kH,kW]`、biasはnilまたは`[out_channels]`です。矩形・偶数カーネル、非連続viewに対応します。

| `Conv2dOptions` | 条件・既定 |
| --- | --- |
| `Stride` / `Dilation` | height / width順。両方0なら`[1,1]`、それ以外は両方正 |
| `Padding` | 各軸の非負な対称パディング。既定0 |
| `Groups` | 0なら1。両チャネル数を割り切ること。`Groups == in_channels`でdepthwise |

出力サイズは各軸で`floor((input + 2*padding - dilation*(kernel-1) - 1)/stride) + 1`です。入力・カーネル・出力の次元は正、要素数は符号付き32ビットindex内である必要があります。不正な演算はpanicし、層のコンストラクタはエラーを返します。

```go
conv, err := autograd.NewConv2dLayer(4, 8, [2]int{3, 5},
    autograd.Conv2dOptions{
        Stride: [2]int{1, 2}, Padding: [2]int{1, 2}, Groups: 2,
    }, true, device, rand.New(rand.NewSource(7)))
```

CPU/CUDAは空間タイルへ展開してGEMMを行います。im2colと出力タイルの予算は4 MiB、出力位置は最大512です。カーネルだけで予算を超える場合は1列を使います。Backwardは完全なim2colを保存せず再計算します。BF16有効時もFP32で計算し、CUDAの入力勾配はatomic加算のためbit単位の再現を保証しません。

CNN例はConv2d / BatchNorm / ReLU、max pooling、adaptive global average pooling、分類器、cross-entropy、AdamWを使い、保存後の再読込みを確認します。出力先ディレクトリを作ってから実行してください。

```sh
go run ./cmd/cnn-train -device cpu -steps 60 -out out/cnn.safetensors
go run ./cmd/cnn-train -device cuda -steps 60 -out out/cnn-cuda.safetensors
```

## NCHW pooling

`MaxPool2d`、`AvgPool2d`、`AdaptiveAvgPool2d`はCPU/CUDAのFP32 NCHW入力を扱います。次元は正、要素数は符号付き32ビットindex内である必要があります。非連続viewは実体化し、不正な形状・設定は割当の前にpanicします。

- `KernelSize`は必須。Strideが全て0ならKernelSizeを使う。maxのDilationが全て0なら`[1,1]`。その他のstride / dilationは正、Paddingは各軸0〜カーネルサイズの半分。averageのdilationは1。
- 出力式はConv2dと同じ。`CeilMode`はceilを使い、右端・下端のパディングだけで始まる最後の窓を除く。
- maxのパディングは負の無限大。同値はrow-majorで最初の有効要素、NaNは最後のNaNを選ぶ。dilationした窓に有効要素がなければ負の無限大となり、入力勾配を生まない。indexのテンソルは返さない。
- averageは既定でパディングも除数に含める。`ExcludePad: true`は有効位置だけを数える。ceilの部分窓でも、要求したパディングを超える位置は除数から除く。正の`DivisorOverride`は除数を上書きし、0は通常、負はエラー。
- adaptiveの正の出力サイズに対し、各binは`floor(position*input/output)`から`ceil((position+1)*input/output)`。binの重なりと入力より大きい出力に対応し、勾配は全binの寄与を加算する。

各Layerは`TensorLayer`を実装し、パラメータ・バッファ・train/eval stateを持ちません。BF16下でもFP32です。CUDAの重複する勾配はatomic加算で、capture / 再生に対応します。

## BatchNorm

`BatchNorm(x, weight, bias, runningMean, runningVar, options)`はrank 2〜6のチャネル以外を正規化します。チャネル軸は既定1、`ChannelsLast: true`なら最後です。weight / biasは独立に任意の`[channels]`で、省略時は1 / 0です。テンソルのdeviceと実行コンテキストは互換でなければなりません。

- `Eps`は0なら`1e-5`、それ以外は正で有限。
- 学習はチャネルごとに2サンプル以上必要。出力は偏りのあるバッチ分散、running varianceは不偏分散を使う。
- running mean / varianceは両方nilか、両方とも連続した勾配不要の`[channels]`。nilの学習はstateを更新しない。関数版の評価は両方が必要で、1サンプルでもよい。
- `Momentum`は有限の`[0,1]`。更新式は`(1-Momentum)*old + Momentum*batch`。関数版の0はrunning値を凍結するが、学習の正規化はバッチ統計のまま。変更するrunningバッファは他の入力・stateとaliasしない。
- 各forwardはmean / 逆標準偏差を保存する。後のrunning更新・Close・モード変更で既存グラフの導関数は変わらない。入力とaffineの管理された変更はversion検証の対象。

`BatchNormLayerOptions{}`はaffineとrunning統計、epsilon `1e-5`、momentum `0.1`、channels-first、学習モードが既定です。`DisableAffine`は重み・biasを省き、`DisableRunningStats`は学習・評価ともバッチ統計を使うため学習時のサンプル数要件も適用します。

層のMomentumはポインタで、nilは`0.1`、0を指せば明示的に凍結します。`CumulativeMomentum`は`1/num_batches_tracked`でバッチ平均を取ります。サンプル数で重み付けした平均ではありません。オプションは構築時にコピーします。

Moduleへ`weight` / `bias`と`running_mean` / `running_var` / スカラーの`num_batches_tracked`を登録します。バッファはoptimizerの対象外です。全stateがF32のためカウンタの整数精度は`2^24`までで、PyTorchのI64を取り込む際は変換が必要です。

`Train` / `Module.Train`が統計更新を制御し、NoGradは勾配記録だけを制御します。モードと構成はStateDictに保存しません。層は`TensorLayer` / `Trainer` / `ParameterizedLayer`を実装し、Closeはaffineとバッファを解放します。

CPU/CUDAとも出力・勾配・保存統計はFP32、モーメントのreductionはfloat64累積です。CUDAのrunning更新とカウンタはデバイス上で行い、累積平均もcapture / 再生に対応します。

## 汎用の損失

`MSELoss`、`L1Loss`、`BCEWithLogitsLoss`は同shape・同deviceの予測とtargetからスカラーを返します。

| 設定 | 仕様 |
| --- | --- |
| `ReductionMean` | ゼロ値の既定 |
| `ReductionSum` | 総和。`none`は未提供 |
| `LossOptions.Weight` | 同shape・同deviceの勾配不要テンソル |
| MSE / L1の重み付きmean | 重みの和で割り、0の重みを平均から除く |
| BCEの重み付きmean | 要素数で割る。0の重みも除数に含む |

BCEは`max(x,0) - x*y + log(1+exp(-|x|))`で計算し、targetは`[0,1]`です。確率そのものを渡すBCEは未提供です。NaNフレームのマスクにはrank 3専用の`MaskedLoss`を使います。形状・device不一致、未知のreductionはpanicします。

### 学習ループ

`TrainingLoop[B]`はreaderからmicro-batchを取得し、groupの先頭でZeroGrad、完了した更新ごとにOptimizer.StepとSchedule.Stepを1回呼びます。呼出し側の`Step`がグラフを作り、`GradAccumulator.Backward`へ渡します。

- ReaderがEOFなら次のepochへ進む。
- Reportは最初・LogEvery間隔・最後の更新で呼ばれ、その更新の最後のmicro-batchの損失を受け取る。
- Saveは予定更新の完了時、またはcontext終了時に1回呼ぶ。reader / Stepのエラーでは保存しない。
- `Accumulate >= 2`にはSamplerが必要。途中のキャンセルでは勾配を捨て、Samplerをgroupの先頭へ戻し、そのpayloadをSaveへ渡す。model / optimizer / scheduleは最後の完了更新を保つ。
- 巻き戻すのはSamplerだけ。forwardが更新したrunning統計や乱数状態は呼出し側で戻す。

## 勾配蓄積

`GradAccumulator`は損失を`1/groups`にスケールしてBackwardし、group完了を返します。蓄積器はループ外で作り、各micro-batchの損失を渡します。ZeroGrad、クリップ、optimizer / scheduleのStepは呼出し側で行い、クリップと更新はgroup完了後に行ってください。

```go
accumulator, err := autograd.NewGradAccumulator(4)
```

各micro-batchで次の処理を行います。

```go
if accumulator.Pending() == 0 {
    optimizer.ZeroGrad()
}
complete, err := accumulator.Backward(loss)
loss.ReleaseGraph()
if complete {
    autograd.ClipGradNorm(params, 1)
    optimizer.Step()
    schedule.Step(optimizer)
}
```

端数groupでも`1/groups`のスケールを保ちます。途中で更新した場合はResetしてください。未完了の勾配はcheckpointに含まれないため、保存はgroup完了後に行います。

平均するのはmicro-batchごとの損失です。大きな1バッチと同じ勾配になるのは、有効件数と分母が等しい場合だけで、短いバッチや異なるマスクは呼出し側で重み付けします。

## 評価指標

`metrics`はホスト配列を集計します。各型はゼロ値から使え、空の値は0、Countで観測の有無を区別し、Resetで消去します。同じ集計への並行書込は非対応です。

- `Mean.Add(value, weight)`は平均値を対象件数で重み付けする。マスク付き損失は有効件数を渡す。
- `Accuracy.UpdateLogits(scores, labels, classes, ignoreIndex)`は`[len(labels),classes]`のrow-majorスコアを扱う。同点は小さいクラス番号、ignoreIndexは分子・分母から除く。Correctは正解件数、Countは有効件数。
- `Regression.Update(prediction, target)`は同長のfloat32配列からMSE / RMSE / MAEを求める。差分と和はfloat64、Countと除数はサンプル数ではなく要素数。

形状不一致、NaN/Inf、不正な件数、overflowは集計を変えずに拒否します。無視する分類行のスコアも有限値が必要です。

## 中断時の保存

MNIST・数値回帰・音声の例は`TrainingLoop.Save`で最後の完了更新を保存します。更新前の中断でも初期checkpointから再開できます。中断時には評価と推論用重みの書出しは行いません。

蓄積はepochをまたぎ、短いバッチも1 micro-batchとして数えます。未完了groupはデータを読み直します。例のモデルは変更されるrunning統計・dropoutの乱数状態を持ちません。他のモデルではそれらの巻戻しも必要です。

## OptimizerとSGD

`Optimizer`は`Step()`、`ZeroGrad()`、`LearningRate() float32`、`SetLearningRate(float32)`、`Close()`を公開します。OneCycleはこのAPIで次の学習率を設定します。学習率は有限で非負、不正値はpanicします。

`CheckpointOptimizer`は`Snapshot` / `ValidateSnapshot` / `LoadSnapshot`を追加し、`adamw` / `sgd`のタグ付きstateを扱います。独自OptimizerとOneCycleは組み合わせられますが、checkpointのエンコーダは組込みの2種類だけに対応します。

`NewSGD(params, options)`のパラメータは同deviceの名前が一意な連続F32末端で、要素数は正である必要があります。スライスとオプションをコピーするため、パラメータの同一性・順序を保ってください。

| `SGDOptions` | 仕様 |
| --- | --- |
| LR / Momentum / Dampening / WeightDecay | 有限・非負。LR=0でもmomentumは更新 |
| Nesterov | Momentum > 0かつDampening=0が必要 |
| Maximize | 生の勾配だけ符号反転。decayは値を縮める |

`d = grad`（Maximize時は`-grad`）に`WeightDecay * parameter`を加えます。momentumの最初のバッファはdampeningなしの`d`、以後は`buffer = Momentum*buffer + (1-Dampening)*d`です。更新は通常buffer、Nesterovは`d + Momentum*buffer`を使い、`LR * update`を引きます。Momentum=0では通常のSGDでDampeningは作用しません。

勾配がないパラメータはdecayとstate更新も飛ばします。0の勾配は更新対象です。momentumの遅延初期化はパラメータごと、StepCountは全勾配がないStepも数えます。eagerなZeroGradは勾配を無い状態へ戻しますが、CUDAグラフ用に準備した割当は保持して0で埋めます。

stateは名前・形状・全オプション・ステップ・初期化フラグと、初期化済みのmomentumを保存します。LoadStateは全てを検証してから置換します。単体ファイルのmetadataは`gograd-sgd`、統合形式は`gograd-training-1`です。種類タグのない旧学習checkpointはAdamWとして扱い、種類の不一致はmodel変更前に拒否します。

CPU/CUDAのeager更新はストレージのversionを進め、BF16シャドウを無効にします。SGDはFP32、CUDAの更新とstateはデバイス上に留まります。Closeが解放するのはoptimizer stateで、パラメータは呼出し側の所有です。

CUDAグラフでは`PrepareGraph` / `GraphStepCount` / `SyncGraphStepCount`を使います。stateの書出しとeagerへの復帰ではデバイス上のステップ数を同期します。state復元や割当変更は古いグラフを終了してから行ってください。外部再生による書込はホスト側versionの追跡対象外です。

### パラメータグループ

`ParameterGroupSpec{Name, Params, LRScale, WeightDecay}`を`NewSGDGroups` / `NewAdamWGroups`へ渡します。平坦なコンストラクタは暗黙の`default`グループに相当します。`PartitionParameters`で名前により分割できます。

- specは全パラメータを過不足なく1回ずつ含み、名前は一意・非空、各グループも非空。
- LRScaleは基準学習率への倍率で有限・非負。0は基準学習率をそのまま使う指定。WeightDecayはグループごとの絶対値。
- `NewSGDGroups`の`options.WeightDecay`は0が必要。momentum / dampening / Nesterov / Maximizeは全グループで共有する。
- stateは順序付きのパラメータ名とグループ構成を保存する。復元はmembership・スケール・decayを変更前に検証する。
- グループのない旧stateはスケールを持たない1グループのoptimizerだけが受け入れ、保存済みの基準decayを使う。グループ化したstateと平坦なoptimizerの相互読込みは拒否する。
- 公開の基準WeightDecayは最初のグループの値を写す。CUDAグラフは記録時の基準学習率とスケールを固定するため、後のスケジューラ更新を自動では反映しない。

## 汎用のホストデータとバッチ処理

`Dataset[T]`は`Len() int`と`Get(context.Context, index) (T, error)`を公開します。`SliceDataset`はスライスを使い、`NewSubset`はindexだけをコピーします。Indicesはコピーを返します。`RandomSplit(size, validationCount, seed)`は全件を覆う重複しない分割を作ります。反復・再開中はデータの長さ・内容・indexの同一性を保ってください。

```go
sampler, err := data.NewIndexSampler(dataset.Len(), data.SamplerOptions{
    Shuffle: true, Seed: 7,
})
loader, err := data.NewLoader[data.FloatSample, data.FloatBatch](
    dataset, sampler, data.Stack, data.LoaderOptions{BatchSize: 2},
)
defer loader.Close()
```

`IndexSampler`の既定は昇順です。shuffleは専用のseed付きPCG、Peekは状態を進めず次のindexをコピーし、Advanceで確定します。NextEpochは残りを捨て次の順序を始めます。空のデータセットはEOFを返します。

`Loader.Next`は取得・collate・キャンセル確認が成功した後にだけカーソルを進めます。読込やcollateの失敗は再試行でき、callbackのEOFはUnexpectedEOFへ変換します。BatchSizeは正、末尾は既定で小さいバッチとなり、DropLastでは消費してEOFを返します。

`Stack`は同shapeのsampleをコピーしてバッチ軸を付けます。`PadSequences`は末尾次元の一致が必要で、時間軸を最大長か指定Lengthまで`PaddingOptions.Value`で埋めます。短いLengthへの切詰めにはTruncateが必要です。Lengthsは残した長さ、Maskは`[batch,time]`のbool配列で、有効位置だけtrueです。空系列に対応します。

組込みcollateは所有するホストスライスを返します。独自collateは借用sampleを変更しないでください。ラベル・マスクのテンソル化とdeviceへの転送は呼出し側が行います。Loaderと変更可能なSamplerは逐次使用し、独立workerには別インスタンスを用意してください。

### 数値のCSVテーブルと回帰のsample

`ParseNumericCSV`はコメント・ヘッダ付きの矩形CSV/TSVを読みます。空白を除いた各値は有限で、不正な行は1始まりの行番号を報告します。`RegressionSamples`は特徴量とスカラーtargetへ分け、targetColumn=-1なら最後の列です。`StackRegression`は同長の特徴量を`[batch,features]`へまとめます。

`FitStandardizer`は学習splitだけで平均・母分散をfloat64累積し、定数特徴量のscaleを1にします。Applyはコピー、ApplyBatchはその場で変換します。State / LoadStateで統計を保存し、検証・推論に再利用します。不正version・サイズ・非有限値・正でないscaleは拒否します。

`cmd/tabular-train`は`-generate`またはCSVを使い、standardizer・loader順・fingerprintをoptimizer / OneCycleと保存します。再開時にデータ・分割・batch・steps・seed・hidden・optimizerの変更を拒否し、時刻は比較しません。

### 再現可能なaugmentation

`TransformDataset[T]`は`Base.Get`の結果を`Apply(index, seed, sample)`へ渡します。indexとSeedを自動で混ぜず、Apply=nilならそのまま返します。nilのcontext / Base、キャンセル、不正indexはエラーです。

sampleはコピーしません。借用スライスを変更するcallbackでコピーし、worker使用時はBase.GetとApplyを並行安全にしてください。先読みの破棄・再取得があるため、共有乱数や呼出し回数に依存させないでください。

- 水平・垂直反転はCHW順、正の次元、長さの一致、有限の確率`[0,1]`、非負indexが必要。確率0でも検証し、失敗時は入力を変更しない。`FlipsHorizontallyFloat32` / `FlipsVerticallyFloat32`は常に反転する。
- 新しい反転APIはuint64のseedを受け取る。旧int-seed APIの`FlipEarly`は32-bit環境でseedが切り詰められるため、新規コードでは使わない。
- `AddGaussianNoiseFloat32`は非負index、有限かつ非負のstd、有限の入力が必要。std=0と空配列を受け入れ、検証失敗・float32 overflowでは変更しない。出力はclipしない。

組込み変換はseed / indexだけで乱数を作り、epochが変わっても同じsampleには同じ変換となります。元indexを保つには変換したdatasetをSubsetで包みます。逆順ではsubset内の位置を使います。元データ・分割・変換の種類 / version / 設定 / seedはアプリ側metadataに保存してください。

CNNのIDX経路は`-augment`時だけ横反転.5、縦反転.3、ノイズstd .02を学習画像に適用します。検証画像は変えません。反転がラベルの意味を保つかはデータ次第です。

### WAV音声とaugmentation

`ReadWAV`は非圧縮RIFFの8/16/24/32ビット整数PCM、32/64ビットIEEE float、`WAVE_FORMAT_EXTENSIBLE`に対応します。整数は`2^(bits-1)`で割り、8ビットは符号なし、floatは値をそのまま使います。Samplesはインターリーブ、Monoはチャネル平均です。圧縮・未知のレイアウト・フレーム数が整数にならないサイズは拒否します。

`WriteWAV`は整数を表現範囲にclampして最も近い段へ丸め、floatの非有限値を拒否します。往復誤差は整数PCMで最小段の1.5倍以内、floatでは誤差なしです。

`cmd/audio-train`はmonoをwindowへ分け、`(seed, window index)`の専用PCGでノイズを作ります。worker / prefetch / 再開で入力を変えず、MLPのdenoiserをMSEで学習します。SNRをdBで表示し、WAVのfingerprint、seed、model / optimizer / schedule / loader順を保存します。

### データのcheckpoint

`IndexSampler.State`はversion・サイズ・seed・shuffle・epoch・位置・全順序・PCG stateを保存します。復元前に構成・順列・乱数payloadを検証し、現在と将来のepochの順序を保ちます。

`Loader.State`はサイズ・BatchSize・DropLastとStatefulSamplerのpayloadを持ちます。LoadStateは構成を検証してから復元します。データ内容・subset / split・独自collateの同一性はアプリ側metadataで管理します。長さの一致だけでは同じデータとは判定しません。

### 並列readerと上限付きprefetch

Workers=0は同期、正ならホストreaderを使います。Prefetchは現在のバッチを含む未処理バッチ数の上限で、並列時の0は既定2、同期時は0です。負値・容量overflowはエラーです。reader数はsampleの枠に制限します。

- workerが呼ぶのはDataset.Getだけ。collateはNextの呼出し元で直列に行う。テンソル生成とCUDAスレッド管理も呼出し側に残る。
- 結果は完了順ではなくsampler順に渡す。複数の失敗はsampler順で最初のsampleを報告する。
- 読込 / collate失敗時は未処理の先読みを捨て、readerをjoinし、確定済みカーソルを保つ。workerのpanicは読込エラー、collateのpanicは後片付け後に再panicする。
- contextのキャンセルはreaderの完了を待つ。Getはキャンセルに協調し、collateまで有効なsampleを返す必要がある。生きているcontextで次のNextを呼べば未消費のバッチから再試行する。
- Closeは繰返し呼べ、readerをキャンセル・joinする。EOF、最終バッチ、NextEpochでも古いreaderを解放する。Close / State / LoadStateを含む同じLoaderのメソッドは逐次で呼ぶ。
- Stateはreaderと先読みを止め、配達済み位置だけを保存する。LoadStateも古い読込を停止する。Workers / Prefetchはstateの同一性に含まれないため、変更して再開できる。

反復中のSamplerはLoaderが所有するものとして扱い、外から進めたり戻したりしないでください。Getの副作用、共有乱数、独自collateのstateはLoaderが巻き戻しません。

### raw/gzip IDXとMNISTの例

`OpenIDX`はmagic 2051/2049の画像・ラベルを読み、次元・個数・payload長を検証します。拡張子によらずraw / gzipに対応します。内部はuint8で保持し、Getは所有する`[0,1]`のfloat32 CHW画像と整数ラベルを返します。Fingerprintは復号した両ファイルのSHA-256です。

MNIST例は784→64→10のMLP、cross-entropy、AdamW / SGD、OneCycleを使います。学習はshuffle、評価は昇順です。`-download`は不足ファイルだけ取得し、ローカルIDXも使えます。

```sh
go run ./cmd/mnist-train -download -device cpu -steps 100 -stop-after 40 -train-limit 2000 -valid-limit 1000
go run ./cmd/mnist-train -device cpu -steps 100 -train-limit 2000 -valid-limit 1000 -resume out/mnist.safetensors.training.safetensors
go run ./cmd/mnist-train -device cuda -steps 100 -train-limit 2000 -valid-limit 1000 -out out/mnist-cuda.safetensors
```

既定はデータ`out/mnist`、出力`out/mnist.safetensors`、batch 64、seed 7です。TrainLimit / ValidLimitの0は全件、それ以外は公式splitの先頭N件です。StopAfterは完了更新の絶対数、Stepsは再開でも変えないスケジュール全体の更新数です。学習checkpointは既定で出力に`.training.safetensors`を付け、推論出力とは別パスにします。

`-workers` / `-prefetch`は学習と評価の両方に作用し、既定は同期の0です。`-accum N`はN micro-batchごとに更新し、既定1です。データ・subset・seed・batch構成・蓄積数・optimizer設定は再開時に照合します。

`-optimizer adamw|sgd`はoptimizer、`-lr`はOneCycleのピークを選び、0ならAdamW .005 / SGD .1です。SGDのmomentumは既定.9、`-nesterov`で有効化します。decayは.0001です。`-no-decay-suffix`は既定`bias`で、一致するパラメータのdecayを無効にし、空ならグループ分けを無効にします。これもcheckpointの同一性に含み、グループ導入前のcheckpointの再開は空にします。

## 学習の再開

model、optimizer、スケジューラ、samplerを先に構築し、更新完了後に保存します。

```go
sampleJSON, err := json.Marshal(sampler.State())
err = autograd.SaveTrainingCheckpoint("training.safetensors", &model.Module,
    opt, schedule, map[string]string{"sampler": string(sampleJSON)})
metadata, err := autograd.LoadTrainingCheckpoint("training.safetensors",
    &model.Module, opt, schedule)
var state autograd.WindowSamplerState
err = json.Unmarshal([]byte(metadata["sampler"]), &state)
err = sampler.LoadState(state)
```

名前・順序・形状を検証してからmodel / optimizerへコピーします。学習率・decay・beta・epsilon・OneCycleを復元し、モデルのモード、dropout seed、データとsamplerの同一性は呼出し側で管理します。推論用重みやoptimizer単体のファイルだけでは学習全体を再開できません。

CUDAグラフは未完了launchを同期してから保存し、stateを復元してから新しく記録します。optimizerのスナップショットはホストの古い値ではなくデバイス上のステップ数を読みます。

## 互換性の注意

- autograd、CUDA graph、BF16、割当プール、opレジストリは実験的な位置づけです。dtypeはfloat32を中心とし、BF16は任意のシャドウを使います。
- CUDAのdevice / stream / captureはOSスレッド単位で、プールはdevice / stream単位です。複数deviceを1つの実行コンテキストで扱うことは未対応です。独立分岐の複数stream実行にはRunBranchesを使います。
- safetensorsは同じディレクトリの一時ファイル、sync、renameで保存します。読込みは不正形状・overflow・重複範囲・16 MiB超のヘッダを拒否します。通常のLoadSafeTensorsはファイル全体をメモリへ読み、大きいファイルにはOpenSafeTensorFile / LoadSafeTensorsStreamを使えます。deviceへの転送失敗は読込みを途中で止める場合があります。
- 旧`nn`は独自のcheckpoint経路を保ちます。学習checkpointはAdamW / SGDとOneCycleを対象にします。
- GoとPyTorchはサンプリング・初期化・dropoutの乱数ストリームが異なります。演算比較は重みとマスクを固定し、学習全体は測定した品質で比較します。
