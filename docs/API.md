# 学習API

CPU / CUDA向けの学習APIリファレンスです。学習checkpointはsafetensorsの `gograd-training-1` 形式で保存します。Vulkan / DirectMLでの推論については [GPUバックエンド](GPU_BACKENDS.md) を参照してください。

## 学習で使うAPI

| パッケージ | API | 用途 |
| --- | --- | --- |
| `torchpth` | `Read` | PyTorch zip形式のtensor / state_dictをfloat32で読み込み |
| `tensor` | `Device`、`CPU`、`CUDA` | 構築時のデバイス選択 |
| `autograd` | `NewCUDAContext`、`NewCUDAContextOn`、`CUDAContext.Device`、`Close` | OSスレッド固定、CUDAデバイスの選択と復元 |
| `autograd` | `New`、`Zeros`、`ToHost`、`Backward`、`ReleaseGraph`、`Close` | テンソル生成、転送、逆伝播、寿命管理 |
| `autograd` | `Add`、`Sub`、`Mul`、`Div`、`Max`、`Min`、`AddScalar`、`SubScalar`、`MulScalar`、`DivScalar`、`PowScalar`、`Exp`、`Log`、`Sqrt`、`Sigmoid`、`Tanh`、`Sin`、`Cos`、`Softplus`、`Erf`、`Log1p`、`Expm1`、`Abs`、`ReLU`、`GELU`、`Clamp` | 要素演算とbroadcast |
| `autograd` | `Reshape`、`Unsqueeze`、`Squeeze`、`Flatten`、`Transpose`、`Permute`、`Slice`、`Expand`、`Contiguous`、`Concat`、`Stack` | viewと実体化（`Flatten` の範囲は両端を含む） |
| `autograd` | `RetainData`、`RetainGrad`、`BackwardWithOptions`、`BackwardOptions` | 値・勾配・計算履歴の保持 |
| `autograd` | `NewExecutionContext`、`New`、`Zeros`、`Bind`、`BindModule`、`NoGrad`、`Recording`、`Tensor.ExecutionContext` | 記録スコープの分離 |
| `autograd` | `ExecutionOptions`、`SetOptions`、`Options`、`Autocast` | BF16とアテンション方式の選択 |
| `autograd` | `ExecutionContext.SetCUDA`、`CUDA`、`WithCUDA` | デバイス / streamの記録と一時適用 |
| `autograd` | `NewEmbeddingLayer`、`NewConv1dLayer`、`NewLayerNormLayer`、`NewLinearLayer`、`DropoutLayer`、`GELULayer`、`FlattenLayer`、`Sequential` | グラフモジュール（Conv1dは入力 `[batch,time,channels]`、重み `[out,in,kernel]`） |
| `autograd` | `Conv2d`、`Conv2dOptions`、`NewConv2dLayer`、`Conv2dLayer` | grouped / depthwiseを含むNCHW畳み込み |
| `autograd` | `Conv1d`、`Conv1dGEMM`、`Conv1dGEMMStride` | time-majorの1D畳み込み（biasはnilまたは `[out_channels]`） |
| `autograd` | `MaxPool2d`、`AvgPool2d`、`AdaptiveAvgPool2d`、各Options / Layer | NCHW pooling |
| `autograd` | `BatchNorm`、`BatchNormOptions`、`NewBatchNormLayer`、`BatchNormLayerOptions` | 正規化とrunning state |
| `autograd` | `NewRMSNormLayer`、`NewGroupNormLayer`、`RMSNormLast` | 最終軸のRMS正規化、group単位の正規化 |
| `autograd` | `RotaryHalf`、`GeGLU`、`SwiGLUProjection`、`KeyPaddingAttention` | 音声Transformer向けのRoPE、GLU、マスク付きattention |
| `autograd` | `Module.StateDict`、`LoadStateDict`、`SaveSafeTensors`、`SaveSafeTensorsMetadata`、`LoadSafeTensors` | 名前付きFP32 stateの保存・読み込み（metadataは文字列マップ） |
| `autograd` | `Module.Train`、`SetTraining`、`OnTrainingChange` | 子モジュールへのtrain/evalモード伝播 |
| `autograd` | `TrainingLoop[B]`、`BatchReader[B]`、`PositionSampler`、`UpdateSchedule`、`OneCycle.Updates` | 更新・ログ報告・停止時保存 |
| `autograd` | `GradAccumulator`、`NewGradAccumulator`、`Backward`、`Pending`、`Reset`、`Groups` | micro-batchの勾配蓄積 |
| `autograd` | `MSELoss`、`L1Loss`、`SmoothL1Loss`、`BCEWithLogitsLoss`、`LossOptions`、`Reduction` | mean / sumおよび要素ごとの重み付き損失 |
| `autograd` | `MaskedLoss(pred,target,false)`、`ClipGradNorm`、`NewAdamW`、`NewOneCycle` | NaNフレームをマスクしたL1、勾配クリップ、AdamW、OneCycle |
| `autograd` | `Optimizer`、`CheckpointOptimizer`、`OptimizerSnapshot`、`LearningRate`、`SetLearningRate` | パラメータ更新・学習率・checkpointの共通API |
| `autograd` | `AdamW.State`、`LoadState`、`SaveSafeTensors`、`LoadSafeTensors`、`Close` | AdamWのモーメント・設定・ステップの保存と復元 |
| `autograd` | `NewSGD`、`SGDOptions`、`SGD.State`、`SGDState`、`LoadState`、`SaveSafeTensors`、`LoadSafeTensors` | momentum / Nesterov / decay / ascentとstateの復元 |
| `autograd` | `ParameterGroupSpec`、`PartitionParameters`、`NewSGDGroups`、`NewAdamWGroups`、`OptimizerGroupState` | パラメータグループごとの学習率スケールとdecay設定 |
| `autograd` | `SaveTrainingCheckpoint`、`LoadTrainingCheckpoint`、`TrainingCheckpointMetadata` | model / optimizer / OneCycle / metadataの一括保存 |
| `autograd` | `IndexLoader.State`、`LoadState`、`WindowSampler.Batch`、`State`、`LoadState` | epoch順・PCG windowの再開 |
| `autograd` | `StreamBranch`、`RunBranches` | 独立したCUDA分岐のenqueueとeventによる合流 |
| `cuda` | `SetCurrentStream`、`CurrentStream`、`CurrentStreamHandle`、`WithStream` | 呼び出しスレッドのstream選択 |
| `cuda` | `Event`、`NewEvent`、`Record`、`Stream.WaitEvent`、`Event.WaitOn`、`ElapsedTime` | stream間の順序付けと計時 |
| `cuda` | `MemoryStats`、`ResetAllocationPeak`、`SetPoolCacheLimit`、`ReleasePool` | メモリ割当量・ピーク・キャッシュ上限の管理 |
| `data` | `Dataset[T]`、`SliceDataset`、`NewSubset`、`RandomSplit` | サンプル取得とデータ分割 |
| `data` | `Sampler`、`StatefulSampler`、`NewIndexSampler`、`IndexSamplerState` | index選択、shuffle、途中再開 |
| `data` | `NewLoader[T,B]`、`LoaderOptions`、`LoaderState`、`CollateFunc`、`Close` | バッチ処理と上限付きprefetch |
| `data` | `FloatSample`、`FloatBatch`、`Stack`、`PadSequences`、`PaddingOptions`、`OpenIDX` | ホストバッチ、系列長・マスク、IDX画像データ |
| `data` | `TransformDataset[T]`、`RandomHorizontalFlipFloat32`、`FlipsHorizontallyFloat32`、`RandomVerticalFlipFloat32`、`FlipsVerticallyFloat32`、`AddGaussianNoiseFloat32` | seed / indexに基づく決定的なaugmentation |
| `data` | `ParseNumericCSV`、`ParseOptions`、`NumericTable`、`RegressionSamples`、`RegressionSample`、`RegressionBatch`、`StackRegression` | 数値CSV / TSVと回帰バッチ |
| `data` | `Standardizer`、`FitStandardizer`、`StandardizerState`、`Apply`、`ApplyBatch`、`State`、`LoadState` | 学習データの統計による標準化 |
| `data` | `ReadWAV`、`ReadWAVFile`、`ReadWAVMono`、`WAV`、`Mono`、`WriteWAV`、`WAVEncoding` | 非圧縮RIFF音声の入出力 |
| `metrics` | `Mean`、`Accuracy`、`TopKAccuracy`、`Regression`、`ConfusionMatrix`、各型の`Count` / `Reset` | ホスト側の損失平均・正解率・回帰誤差・混同行列 |

### 要素演算とCUDAコンテキスト・メモリプール

要素演算の主なエッジケースは以下の通りです。
- `Max` と `Min` は同値の場合に勾配を右辺へ流すため、二重計上されません。
- `Clamp` は両端を含む区間で微分可能で、`Sqrt` の微分は0で無限大になります。
- `Softplus` は `max(x,0)+log1p(exp(-|x|))` で計算し、有限入力に対して常に有限値を返します。`Log1p` と `Expm1` は0近傍の桁落ちを回避します。

CUDAのデバイス・暗黙のstream・capture状態はOSスレッド単位で管理されます。処理の間は `NewCUDAContext` を保持してください。`NewCUDAContextOn` はデバイスを検証して選択し、`Close` で以前のデバイスへ復元します。

割当プールはデバイス・stream・サイズクラスで分離されています。`Buffer.Device()` は割当先、`Buffer.Size()` は論理バイト数を返します。コピー・memset・freeはバッファ自身のデバイスを選択したうえで呼び出し側のデバイスを復元します。カーネルとcuBLASは選択中のstreamに従い、`WithStream` はpanic時も以前の選択へ戻します。

`MemoryStats` は `LiveBytes`、`CachedBytes`、`ReservedBytes`、`PeakLiveBytes`、`PeakReservedBytes`、`CachedBuffers`、`CacheLimitBytes` を返します（CUDAライブラリ内部の独自割当は含みません）。キャッシュ上限は既定で2 GiBです。上限を下げると古いキャッシュから解放しますが、使用中およびCUDAグラフが予約した領域は解放しません。`ResetAllocationPeak` はピーク値を現在値へリセットします。

独立した分岐処理は呼び出し側で明示します。`RunBranches` は各streamへ処理をenqueueし、カレントstreamが全分岐を待つeventを記録します。

```go
err := autograd.RunBranches([]autograd.StreamBranch{
    {Stream: left, Run: func() error { ya = branchA(x); return nil }},
    {Stream: right, Run: func() error { yb = branchB(x); return nil }},
})
```

手動で合流させる場合は `Event.Record` と `Stream.WaitEvent` / `Event.WaitOn` を使用します（`Event.WaitOn` には待たせる側のstreamを指定します）。streamはそのstream上のバッファをすべて解放するまで保持してください。

`Capture` には明示的なstreamが必要で、同一スレッドでの入れ子は拒否されます。現在のデバイスを変えずに記録し、記録したデバイス上で再生します。

## 実行コンテキストと勾配の記録

`ExecutionContext` は入力とモデルstateに結び付く記録スコープです。演算・view・`Detach` に継承され、異なる明示的コンテキスト同士を組み合わせる演算はpanicします。ゼロ値では勾配を記録します。

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

入力・モデル・optimizerはそれぞれの所有者が解放します。

- `BindModule` は登録済みのパラメータとバッファを再帰的に検証してbindします。重みのtransposeなどパラメータのみからなる部分式も含め、グラフ構築前に実行してください。既存のグラフや別コンテキストの割当のbindは拒否され、値は複製されません。
- bindしていない定数はbind済みの演算へ参加できます。完全に分離するには、勾配不要の入力やマスクも含めて同じコンテキストへbindしてください。
- `NoGrad` はコンテキスト全体に作用します。入れ子やpanicから正常に復帰し、factoryの `requiresGrad` や既存グラフは変更しません。記録済み損失の `Backward` は `NoGrad` 内でも実行可能ですが、未記録の定数スカラーの `Backward` はエラーになります。
- パッケージ直下の `New` / `Zeros` / `NoGrad` はbindしていないテンソル用の互換APIです。共有の既定スコープを使用するため、明示的コンテキストには影響しません。
- 並行workerには独立したコンテキストと変更可能なモデルstateを用意してください。同じモデルの値・勾配・optimizer state・train/evalモードを並行して変更しないでください。

コンテキストはストレージ、stream、capture、割当プールの寿命を管理しません。CUDAではOSスレッドを固定したうえで `SetCUDA` / `WithCUDA` を使用します。

```go
stream, err := cuda.NewStream()
execution.SetCUDA(0, stream)
err = execution.WithCUDA(func() error {
    // kernel、cuBLAS、コピーをこのstreamで実行
    return nil
})
// streamを使う全バッファの解放後に stream.Destroy() を呼ぶ
```

`WithCUDA` は終了時・panic時に以前のデバイス / streamへ戻します。1回の記録を複数streamへ分割する場合も、分割と合流は呼び出し側で指定します。

### 精度とアテンションの方針

明示的コンテキストの既定値は `ExecutionOptions{BF16Autocast: false, AttentionAlgorithm: "auto"}` で、グローバル設定は継承しません。`Options` はコピーを返します。`SetOptions` は不正なアテンション名を受け取ると状態を変えずにエラーを返し、空文字列は `auto` に正規化します。

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

`Autocast` はBF16フラグだけを一時変更し、入れ子・panic後に復元します。forward時に保存された精度とアテンション方針は、その後の設定変更やretainした `Backward` でも維持されます。CUDAグラフは記録時のカーネルを再生するため、方針を変えるには再記録が必要です。

BF16はCUDA GEMMのオペランドとシャドウに使用し、累積・出力・勾配・master weight・optimizerのモーメントはFP32のままです。CPU、Conv2dなどの未対応演算、tiled flash attentionもFP32で動作します。

| アテンション名 | CUDAでの選択 |
| --- | --- |
| `materialized` | GEMMとsoftmax |
| `flash` | Q/KとVのhead次元128までのtiled経路 |
| `auto` | スコアscratchの見積もりが1 GiB以上ならtiled、それ未満なら実体化 |

CPU、dropout有効時、融合できないレイアウトでは合成経路を使います。オプションはforwardの合間に設定してください（複数演算からなるforward全体が1つのスナップショットになるわけではありません）。

## テンソルのストレージとグラフの寿命

ストレージの参照数とグラフの依存数は個別に管理されます。ベーステンソルを `Close` しても、viewや保存済み計算が必要とする割当は保持されます。ある枝の `ReleaseGraph` が別の枝の先祖を回収することはなく、GoのGCによるグラフ回収も行いません。

| 操作 | 保持・解放するもの |
| --- | --- |
| `New` / `Zeros` | 呼び出し側が所有する入力・パラメータ・バッファを作成 |
| `Close` | ハンドルの公開利用を終了（複数回呼び出し可能） |
| `ReleaseGraph` | 中間値と内部の一時的な末端を回収（永続する末端は `Close` が必要） |
| `RetainData` | 値と共有する割当を保持（履歴は保持しない） |
| `RetainGrad` | 値と中間の勾配を保持（履歴は保持しない） |
| `Detach` | 独立したストレージへコピーして履歴を切断（非連続viewとコンテキストも処理） |
| `BackwardWithOptions(RetainGraph: true)` | 履歴と保存した割当を保持し、再度の `Backward` を可能にする |

`Close` 済みのハンドルに対する `ToHost` / `GradToHost` / `CopyFrom` はエラーになりますが、依存するviewやbackwardは内部の値を参照できます。retainしたハンドルも読み終えたら `Close` してください。

通常の `Backward` は履歴を消費し、CUDAの未使用中間値を回収します。retainしていないCUDA出力は `Backward` 前に読み取ってください（CPUのforward値は `ReleaseGraph` まで残ります）。消費済み履歴の再利用はエラーです。retainした実行では末端の勾配を加算し、中間のadjointは伝播用に再生成します。実行後は `ReleaseGraph` で解放してください。

マスクのtarget、embeddingのindex、正規化の統計、im2colなども必要な履歴が終了するまで保持されます。BF16シャドウはFP32ストレージが変更されると別名も含めて無効化されます。

`CopyFrom` とeagerなoptimizer更新はストレージのversionを進め、`Backward` は勾配書き込み前に保存したオペランドと出力のversionを検証します。CPUの `Data` スライスへの直接書き込みや外部からのCUDA書き込み・再生は追跡されないため、グラフが生きている間は保存値を変更しないでください。

`Data` と `Buffer()` は借用です。スライスの差し替え、Tensor所有バッファの直接Free、ハンドル終了後の借用メモリ利用は避けてください。TensorとIndexBufferは所有状態をポインタで保持するため、共有テンソルの変更・`Close`・`Backward` を並行して行わないでください。

## Conv2dとCNNの学習

`Conv2d(x, weight, bias, options)` はCPU / CUDAのNCHW cross-correlationです。入力は `[batch,channels,height,width]`、重みは `[out_channels,in_channels/groups,kH,kW]`、biasはnilまたは `[out_channels]` を指定します。矩形・偶数カーネル、非連続viewに対応しています。

| `Conv2dOptions` | 条件・既定 |
| --- | --- |
| `Stride` / `Dilation` | height / width順。両方0なら `[1,1]`、それ以外は両方正 |
| `Padding` | 各軸の非負な対称パディング（既定0） |
| `Groups` | 0なら1。入出力両方のチャネル数を割り切る値。`Groups == in_channels` でdepthwise |

出力サイズは各軸で `floor((input + 2*padding - dilation*(kernel-1) - 1)/stride) + 1` です。不正な演算はpanicし、層のコンストラクタはエラーを返します。

```go
conv, err := autograd.NewConv2dLayer(4, 8, [2]int{3, 5},
    autograd.Conv2dOptions{
        Stride: [2]int{1, 2}, Padding: [2]int{1, 2}, Groups: 2,
    }, true, device, rand.New(rand.NewSource(7)))
```

CPU / CUDAともに空間タイルへ展開してGEMMを実行します（im2colと出力タイルの予算は4 MiB、出力位置は最大512、カーネル単体で予算を超える場合は1列を使用）。Backwardは完全なim2colを保持せず再計算します。BF16有効時もFP32で計算し、CUDAの入力勾配はatomic加算のためbit単位の再現性は保証しません。

CNNサンプルはConv2d / BatchNorm / ReLU、max pooling、adaptive global average pooling、分類器、cross-entropy、AdamWを使用し、保存後の再読み込みまで検証します。

```sh
go run ./cmd/cnn-train -device cpu -steps 60 -out out/cnn.safetensors
go run ./cmd/cnn-train -device cuda -steps 60 -out out/cnn-cuda.safetensors
```

## NCHW pooling

`MaxPool2d`、`AvgPool2d`、`AdaptiveAvgPool2d` はCPU / CUDAのFP32 NCHW入力を扱います。非連続viewは自動で実体化され、不正な形状・設定は割当前にpanicします。

- `KernelSize` は必須です。`Stride` がすべて0なら `KernelSize` と同じ値になります。maxの `Dilation` がすべて0なら `[1,1]`、それ以外のstride / dilationは正、Paddingは各軸 `0` 〜 `カーネルサイズの半分` です（averageのdilationは1固定）。
- 出力式はConv2dと同じです。`CeilMode` を有効にするとceilを使い、右端・下端のパディングだけで始まる最後の窓を除外します。
- maxのパディングは負の無限大扱いです。同値はrow-majorで最初の有効要素、NaNは最後のNaNを選びます。dilationした窓に有効要素がない場合は負の無限大となり、入力勾配は生じません（indexテンソルは返しません）。
- averageは既定でパディングも除数に含めます。`ExcludePad: true` にすると有効位置だけを数え、ceilの部分窓でも要求パディングを超える位置は除数から除外します。正の `DivisorOverride` で除数を上書きでき、0は通常動作、負はエラーです。
- adaptiveの正の出力サイズに対し、各binは `floor(position*input/output)` から `ceil((position+1)*input/output)` になります。binの重なりや入力より大きい出力にも対応し、勾配は全binの寄与を加算します。

各Layerは `TensorLayer` を実装し、パラメータ・バッファ・train/eval stateを持ちません。BF16下でもFP32で動作します。CUDAの重複勾配はatomic加算され、capture / 再生に対応します。

## BatchNorm

`BatchNorm(x, weight, bias, runningMean, runningVar, options)` はrank 2〜6のテンソルに対し、チャネル以外の軸を正規化します。チャネル軸は既定で1、`ChannelsLast: true` なら最終軸です。weight / biasはそれぞれ任意の `[channels]` で、省略時は1 / 0として扱います。

- `Eps` は0なら `1e-5`、それ以外は正で有限の値が必要です。
- 学習時はチャネルごとに2サンプル以上が必要です。出力には偏りのあるバッチ分散、running varianceには不偏分散を使用します。
- running mean / varianceは両方nilか、両方とも連続した勾配不要の `[channels]` である必要があります。nilでの学習はstateを更新しません。関数版の評価時は両方必須ですが、1サンプルでも動作します。
- `Momentum` は有限の `[0,1]` で、更新式は `(1-Momentum)*old + Momentum*batch` です。関数版で0を指定するとrunning値を凍結しますが、学習時の正規化には引き続きバッチ統計を使います。変更対象のrunningバッファを他の入力・stateとaliasさせないでください。
- 各forwardはmeanと逆標準偏差を保存するため、その後のrunning更新・`Close`・モード変更によって既存グラフの導関数が変わることはありません。入力とaffineの管理された変更のみversion検証の対象になります。

`BatchNormLayerOptions{}` はaffineとrunning統計有効、epsilon `1e-5`、momentum `0.1`、channels-first、学習モードが既定です。`DisableAffine` で重み・biasを省き、`DisableRunningStats` で学習・評価ともバッチ統計を使うようにできます（この場合、評価時もサンプル数要件が適用されます）。

層の `Momentum` はポインタ型で、nilなら `0.1`、0を指すポインタなら明示的に凍結します。`CumulativeMomentum` を有効にすると `1/num_batches_tracked` でバッチ平均を取ります（サンプル数による重み付け平均ではありません）。オプションは構築時にコピーされます。

Moduleには `weight` / `bias` と `running_mean` / `running_var` / スカラーの `num_batches_tracked` が登録されます（バッファはoptimizer対象外）。全stateがFP32のためカウンタの整数精度は `2^24` までとなり、PyTorchのI64を取り込む際は変換が必要です。

`Train` / `Module.Train` が統計更新を制御し、`NoGrad` は勾配記録だけを制御します（モードと構成はStateDictに保存されません）。層は `TensorLayer` / `Trainer` / `ParameterizedLayer` を実装し、`Close` でaffineとバッファを解放します。

CPU / CUDAとも出力・勾配・保存統計はFP32、モーメントのreductionはfloat64累積です。CUDAのrunning更新とカウンタはデバイス上で完結し、累積平均もcapture / 再生に対応します。

## 音声Transformer向けの演算

ModernBERT系の音声モデル向け演算です。いずれもPyTorch出力との照合テスト済みです。

- `RotaryHalf(x, theta)`：`[batch,heads,sequence,headDim]` に半回転RoPEを適用します（`headDim` は偶数、`theta` は正）。
- `GeGLU(x)`：最終軸を2分割し、前半にGELUを適用して後半と掛けます。
- `SwiGLUProjection(x, w1, w2, w3)`：`SiLU(x*w1^T) * (x*w3^T)` に `w2^T` を掛けます（重みはPyTorchの `[out,in]` 配置）。
- `KeyPaddingAttention(q, k, v, valid, radius)`：真偽値のキーマスクを加法的マスクに変換してattentionを計算します。`radius` が負なら全attention、0以上ならクエリ中心の窓の外をマスクします。全キーがマスクされたクエリはpanicします。
- `RMSNormLast(x, weight, eps)`：最終軸を二乗平均平方根で正規化し、チャネルごとの重みを掛けます。

## 汎用の損失

`MSELoss`、`L1Loss`、`SmoothL1Loss`、`BCEWithLogitsLoss` は同shape・同deviceの予測とtargetからスカラーを返します。`SmoothL1Loss` は `|d|<1` で `0.5*d^2`、それ以外で `|d|-0.5` です。

| 設定 | 仕様 |
| --- | --- |
| `ReductionMean` | ゼロ値の既定 |
| `ReductionSum` | 総和（`none` は未提供） |
| `LossOptions.Weight` | 同shape・同deviceの勾配不要テンソル |
| MSE / L1の重み付きmean | 重みの和で割り、0の重みを平均から除外 |
| BCEの重み付きmean | 要素数で割る（0の重みも除数に含む） |

BCEは `max(x,0) - x*y + log(1+exp(-|x|))` で計算し、targetは `[0,1]` を想定します（確率値を直接渡すBCEは未提供）。NaNフレームのマスクにはrank 3専用の `MaskedLoss` を使用してください。形状・device不一致や未知のreductionはpanicします。

### 学習ループ

`TrainingLoop[B]` はreaderからmicro-batchを取得し、groupの先頭で `ZeroGrad`、完了した更新ごとに `Optimizer.Step` と `Schedule.Step` を1回呼びます。呼び出し側の `Step` がグラフを構築して `GradAccumulator.Backward` へ渡します。

- readerがEOFに達すると次のepochへ進みます。
- `Report` は最初・`LogEvery` 間隔・最後の更新で呼ばれ、その更新における最後のmicro-batchの損失を受け取ります。
- `Save` は予定更新の完了時、またはcontext終了時に1回呼ばれます（reader / `Step` のエラー時は保存しません）。
- `Accumulate >= 2` の場合はSamplerが必要です。途中のキャンセル時は勾配を破棄してSamplerをgroupの先頭へ巻き戻し、そのpayloadを `Save` へ渡します（model / optimizer / scheduleは最後の完了更新状態を維持）。
- 巻き戻されるのはSamplerのみです。forwardが更新するrunning統計や乱数状態は呼び出し側で戻してください。

## 勾配蓄積

`GradAccumulator` は損失を `1/groups` にスケールして `Backward` し、groupの完了状態を返します。蓄積器はループ外で作成し、各micro-batchの損失を渡します。`ZeroGrad`、勾配クリップ、optimizer / scheduleの `Step` は呼び出し側で行い、クリップと更新はgroup完了後に実行してください。

```go
accumulator, err := autograd.NewGradAccumulator(4)
```

各micro-batchでの処理例：

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

端数groupでも `1/groups` のスケールを維持します。途中で更新した場合は `Reset` してください。未完了の勾配はcheckpointに含まれないため、保存はgroup完了後に行います。

`ClipGradNorm` はCPUではクリップ前のノルムを返し、CUDAではホストへの読み戻しを避けるため `NaN` を返します（CUDA上のノルム取得には `GradientNormToHost` を使用）。`maxNorm` が0以下ならクリップしません。

平均されるのはmicro-batchごとの損失です。大きな1バッチと厳密に同じ勾配になるのは有効件数と分母が等しい場合のみなので、短いバッチや異なるマスクがある場合は呼び出し側で重み付けしてください。

## 評価指標

`metrics` はホスト配列を集計します。各型はゼロ値から利用でき、空の値は0、`Count` で観測の有無を判別、`Reset` で初期化します（同一インスタンスへの並行書き込みは非対応）。

- `Mean.Add(value, weight)`：平均値を対象件数で重み付けします（マスク付き損失は有効件数を渡す）。
- `Accuracy.UpdateLogits(scores, labels, classes, ignoreIndex)`：`[len(labels),classes]` のrow-majorスコアを扱います。同点は小さいクラス番号を優先し、`ignoreIndex` は分子・分母から除外します。
- `TopKAccuracy`：`NewTopKAccuracy(k)` で作成します。正解ラベルより高いスコアのクラスがk個未満なら正解とみなすため、境界の同点は正解に含まれます。
- `Regression.Update(prediction, target)`：同長のfloat32配列からMSE / RMSE / MAEを算出します（差分と和はfloat64累積、`Count` と除数は要素数）。`R2` は決定係数で、分散のない目標値や空の集計では0を返します。
- `ConfusionMatrix.Update(predicted, actual, classes, ignoreIndex)`：行が正解、列が予測の件数表を作成します。クラス数は最初の成功した更新で固定され、以降の別クラス数での更新は拒否されます。`Precision`、`Recall`、`F1`、`Support` はクラス単位、`MacroF1` は正解ラベルが存在するクラスのみを平均します。

形状不一致、NaN/Inf、不正な件数、オーバーフローは集計状態を変えずにエラーを返します（無視する行のスコアも有限値が必要です）。

## 信号処理

`dsp` はPyTorchおよびtorchaudioの出力を再現するプリミティブを提供します。FFTはfloat64で累積し、書き出し時に丸めます。

- `NewFFT(n)`：2の冪の長さのみ受け付けます。`Transform(x, inverse)` は実部・虚部を交互に並べた `2*n` 要素をインプレース変換し、逆変換時は `n` で割ります。`RealForward` / `RealInverse` は片側スペクトル `n/2+1` 本を扱います。
- `HannPeriodic(n)`：`torch.hann_window(n)` と同じ周期窓を返します。
- `NewSTFT(filterLength, hopLength)`：`1 <= hopLength <= filterLength` が必要です。`Transform` は入力をフィルタ長の倍数までゼロ埋めし、`center` 分の反射パディングを加えてから周波数優先 `[bins][frames]` の振幅と位相を返します。`FramesFor` はフレーム数を事前計算します。
- `Inverse`：重なり加算した窓の二乗和で割り、`numSamples` に合わせて前後を切り落とします（窓に覆われない位置は0、空スペクトルはpanic）。
- `Resample(input, from, to)`：両レートの最大公約数で約分したポリフェーズHann窓sincフィルタを使用します。レートが等しい場合はコピーを返し、0以下のレートや空入力ではnilを返します。

## PyTorchのcheckpointと乱数

- `torchpth.Read(filename)`：`torch.save` のzip形式を読み込み、名前ごとの `Tensor{Shape, Data}` を返します。入れ子のstate_dictはドット区切りで連結し、単体テンソルは空文字の名前で返します。Float / Half / BFloat16のstorageをfloat32へ変換し、それ以外のdtypeは拒否します。
- `torchrng`：PyTorch CPUの乱数列を再現します。`New(seed)` で生成し、`Uniform` / `Normal` はfloat32、`Uniform64` / `Normal64` はfloat64配列を埋めます。`State` / `SetState` で状態を保存・復元できます（並行利用は非対応）。

## 中断時の保存

MNIST・数値回帰・音声のサンプルは `TrainingLoop.Save` で最後の完了更新を保存します。最初の更新前の中断でも初期checkpointから再開でき、中断時には評価および推論用重みの書き出しを行いません。

勾配蓄積はepochをまたぎ、短いバッチも1 micro-batchとして数えます。未完了のgroupはデータを読み直します。サンプルのモデルは変更されるrunning統計やdropout乱数状態を持ちませんが、これらを持つモデルでは呼び出し側での巻き戻しが必要です。

## OptimizerとSGD

`Optimizer` は `Step()`、`ZeroGrad()`、`LearningRate() float32`、`SetLearningRate(float32)`、`Close()` を公開します。OneCycleはこのAPI経由で学習率を更新します（学習率は有限かつ非負、不正値はpanic）。

`CheckpointOptimizer` は `Snapshot` / `ValidateSnapshot` / `LoadSnapshot` を追加し、`adamw` / `sgd` のタグ付きstateを扱います。独自OptimizerとOneCycleの組み合わせは可能ですが、checkpointエンコーダは組み込みの2種類のみ対応しています。

`NewSGD(params, options)` に渡すパラメータは、同一デバイス上で名前が一意な連続FP32末端テンソル（要素数正）である必要があります。スライスとオプションは内部でコピーされるため、パラメータの同一性・順序を保ってください。

| `SGDOptions` | 仕様 |
| --- | --- |
| `LR` / `Momentum` / `Dampening` / `WeightDecay` | 有限・非負。`LR=0` でもmomentumは更新 |
| `Nesterov` | `Momentum > 0` かつ `Dampening == 0` が必要 |
| `Maximize` | 生の勾配のみ符号反転（decayは値を縮める方向のまま） |

`d = grad`（Maximize時は `-grad`）に `WeightDecay * parameter` を加えます。momentumの初回バッファはdampeningなしの `d`、2回目以降は `buffer = Momentum*buffer + (1-Dampening)*d` です。更新量は通常 `buffer`、Nesterovでは `d + Momentum*buffer` を使い、`LR * update` を引きます（`Momentum=0` の場合は通常のSGDとなりDampeningは作用しません）。

勾配を持たないパラメータはdecayとstate更新をスキップし、0の勾配は更新対象とします。momentumの遅延初期化はパラメータごとに行われ、`StepCount` は全パラメータに勾配がない `Step` もカウントします。eagerな `ZeroGrad` は勾配をnil状態に戻しますが、CUDAグラフ用に準備された割当は保持して0埋めします。

stateは名前・形状・全オプション・ステップ・初期化フラグ・初期化済みmomentumを保存し、`LoadState` はすべてを検証してから置換します。単体ファイルのmetadataは `gograd-sgd`、統合形式は `gograd-training-1` です。種類タグのない旧学習checkpointはAdamWとして扱い、種類の不一致はmodel変更前に拒否します。

CPU / CUDAのeager更新はストレージのversionを進め、BF16シャドウを無効化します。SGDはFP32で動作し、CUDA上の更新とstateはデバイス上に留まります。`Close` が解放するのはoptimizer stateのみで、パラメータ自体は呼び出し側の所有です。

CUDAグラフ利用時は `PrepareGraph` / `GraphStepCount` / `SyncGraphStepCount` を使用します。state書き出しとeager復帰時にはデバイス上のステップ数を同期してください。state復元や割当変更は古いグラフを破棄してから行ってください（外部再生による書き込みはホスト側versionの追跡対象外です）。

### パラメータグループ

`ParameterGroupSpec{Name, Params, LRScale, WeightDecay}` を `NewSGDGroups` / `NewAdamWGroups` へ渡します（平坦なコンストラクタは暗黙の `default` グループに相当）。`PartitionParameters` で名前による分割が可能です。

- specは全パラメータを過不足なく1回ずつ含み、グループ名は一意かつ非空、各グループも空でない必要があります。
- `LRScale` は基準学習率への倍率（有限・非負、0は基準学習率をそのまま使う指定）、`WeightDecay` はグループごとの絶対値です。
- `NewSGDGroups` では `options.WeightDecay` を0にする必要があります。momentum / dampening / Nesterov / Maximizeは全グループで共有されます。
- stateは順序付きのパラメータ名とグループ構成を保存し、復元時はmembership・スケール・decayを変更前に検証します。
- グループのない旧stateはスケールを持たない1グループのoptimizerでのみ受け入れ、保存済みの基準decayを使います（グループ化stateと平坦なoptimizerの相互読み込みは拒否）。
- 公開される基準 `WeightDecay` は先頭グループの値を反映します。CUDAグラフには記録時の基準学習率とスケールが固定されるため、後のスケジューラ更新は自動反映されません。

## 汎用のホストデータとバッチ処理

`Dataset[T]` は `Len() int` と `Get(context.Context, index) (T, error)` を公開します。`SliceDataset` はスライスをラップし、`NewSubset` はindexのみコピーします（`Indices` はコピーを返します）。`RandomSplit(size, validationCount, seed)` は全件を覆う重複なしの分割を作成します。反復・再開中はデータの長さ・内容・indexの同一性を維持してください。

```go
sampler, err := data.NewIndexSampler(dataset.Len(), data.SamplerOptions{
    Shuffle: true, Seed: 7,
})
loader, err := data.NewLoader[data.FloatSample, data.FloatBatch](
    dataset, sampler, data.Stack, data.LoaderOptions{BatchSize: 2},
)
defer loader.Close()
```

`IndexSampler` は既定で昇順です。shuffleには専用のseed付きPCGを使い、`Peek` は状態を進めずに次のindexコピーを返し、`Advance` で確定します。`NextEpoch` は残りを破棄して次の順序を開始します（空データセットはEOF）。

`Loader.Next` は取得・collate・キャンセル確認がすべて成功した後にのみカーソルを進めます。読み込みやcollateの失敗は再試行可能で、callbackのEOFは `UnexpectedEOF` へ変換されます。`BatchSize` は正とし、末尾の端数バッチは既定で小さいバッチとして返されます（`DropLast` 指定時は消費してEOFを返却）。

`Stack` は同shapeのサンプルをコピーしてバッチ軸を付与します。`PadSequences` は末尾次元が一致する系列に対し、時間軸を最大長または指定 `Length` まで `PaddingOptions.Value` で埋めます（短い `Length` への切り詰めには `Truncate` が必要）。`Lengths` は元の長さ、`Mask` は `[batch,time]` のbool配列（有効位置のみtrue）で、空系列にも対応します。

組み込みcollateは所有権を持つホストスライスを返します。独自collateでは借用サンプルを変更しないでください。ラベル・マスクのテンソル化やデバイス転送は呼び出し側で行います。Loaderと変更可能なSamplerは逐次使用し、独立workerには別インスタンスを用意してください。

### 数値のCSVテーブルと回帰のsample

- `ParseNumericCSV`：コメント・ヘッダ付きの矩形CSV / TSVを読み込みます。空白を除いた各値は有限である必要があり、不正な行は1始まりの行番号付きでエラーを報告します。`RegressionSamples` は特徴量とスカラーtargetへ分離し（`targetColumn=-1` で最終列）、`StackRegression` は同長の特徴量を `[batch,features]` へまとめます。
- `FitStandardizer`：学習splitのみで平均・母分散をfloat64累積し、定数特徴量のscaleは1にします。`Apply` はコピー、`ApplyBatch` はインプレース変換を行います。`State` / `LoadState` で統計を保存し、検証・推論に再利用できます。
- `cmd/tabular-train`：`-generate` またはCSVを入力とし、standardizer・loader順・fingerprintをoptimizer / OneCycleとともに保存します。再開時にデータ・分割・batch・steps・seed・hidden・optimizerの変更を拒否します（タイムスタンプは比較しません）。

### 再現可能なaugmentation

`TransformDataset[T]` は `Base.Get` の結果を `Apply(index, seed, sample)` へ渡します（`index` と `Seed` の自動合成はせず、`Apply=nil` ならそのまま返します）。サンプルはコピーされないため、変更を加える場合はcallback内でコピーし、並列worker使用時は `Base.Get` と `Apply` をスレッドセーフにしてください。先読みの破棄や再取得が発生するため、共有乱数や呼び出し回数に依存させてはいけません。

- 水平・垂直反転はCHW順、正の次元、長さ一致、有限の確率 `[0,1]`、非負indexが必要です（確率0でも検証を実行し、失敗時は入力を変更しません）。`FlipsHorizontallyFloat32` / `FlipsVerticallyFloat32` は常に反転します。
- 新規コードでは `uint64` のseedを受け取る反転APIを使用してください（旧 `int` seed版の `FlipEarly` は32-bit環境でseedが切り詰められます）。
- `AddGaussianNoiseFloat32` は非負index、有限・非負のstd、有限入力が必要です（`std=0` と空配列は許容、検証失敗やfloat32オーバーフロー時は無変更、出力のclipはなし）。

組み込み変換は `seed` と `index` のみから乱数を生成するため、epochをまたいでも同じサンプルには同じ変換が適用されます。元indexを維持したい場合は変換後のdatasetを `Subset` でラップしてください（逆順でラップするとsubset内の位置がindexになります）。元データ・分割・変換種別・version・設定・seedはアプリ側のmetadataで管理してください。

CNNのIDX経路では `-augment` 指定時のみ、学習画像へ横反転0.5・縦反転0.3・ノイズstd 0.02を適用します（検証画像は無変更）。

### WAV音声とaugmentation

- `ReadWAV`：非圧縮RIFFの8/16/24/32ビット整数PCM、32/64ビットIEEE float、`WAVE_FORMAT_EXTENSIBLE` に対応します。整数は `2^(bits-1)` で正規化し（8ビットは符号なし）、floatはそのまま読み込みます。`Samples` はインターリーブ、`Mono` はチャネル平均を返します。
- `WriteWAV`：整数は表現範囲にclampして最近接丸めを行い、floatの非有限値は拒否します。往復誤差は整数PCMで最小段の1.5倍以内、floatでは誤差なしです。
- `cmd/audio-train`：mono音声をwindowへ分割し、`(seed, window index)` の専用PCGでノイズを生成することで、worker数・prefetch・再開位置に関わらず同一入力を再現します。MLP denoiserをMSEで学習し、SNR（dB）と各種fingerprint・stateを保存します。

### データのcheckpoint

- `IndexSampler.State`：version・サイズ・seed・shuffle・epoch・位置・全順序・PCG stateを保存し、復元前に構成・順列・乱数payloadを検証します。
- `Loader.State`：サイズ・`BatchSize`・`DropLast` と `StatefulSampler` のpayloadを保持し、`LoadState` で構成検証後に復元します。データ内容やsplit、独自collateの同一性はアプリ側のmetadataで管理してください。

### 並列readerと上限付きprefetch

`Workers=0` で同期実行、正の値で並列ホストreaderを使用します。`Prefetch` は処理中バッチを含む未処理バッチ数の上限で、並列時の0は既定値2、同期時は0になります（reader数はサンプル枠数に制限されます）。

- workerが実行するのは `Dataset.Get` のみで、collate、テンソル生成、CUDAスレッド管理は `Next` の呼び出し元スレッドで直列に実行されます。
- 結果は完了順ではなくsampler順で返され、複数エラー時はsampler順で最初のエラーを報告します。
- 読み込み・collate失敗時は未処理の先読みを破棄してreaderをjoinし、確定済みカーソルを維持します（worker内のpanicは読み込みエラー扱い、collateのpanicは後片付け後に再panic）。
- contextキャンセル時はreaderの完了を待機します（`Get` はキャンセルに協調し、collate完了まで有効なサンプルを返す必要があります）。有効なcontextで再度 `Next` を呼ぶと未消費バッチから再試行します。
- `Close` は複数回呼び出し可能で、readerをキャンセル・joinします（EOF・最終バッチ・`NextEpoch` でも古いreaderを解放）。`Close` / `State` / `LoadState` を含む同一Loaderへのメソッド呼び出しは逐次で行ってください。
- `State` と `LoadState` はreaderと先読みを停止し、配達済み位置のみを扱います。`Workers` と `Prefetch` はstateの同一性に含まれないため、変更して再開可能です。

反復中のSamplerはLoaderが所有するものとみなし、外部から直接進めたり戻したりしないでください。`Get` の副作用、共有乱数、独自collateの内部stateはLoaderでは巻き戻せません。

### raw/gzip IDXとMNISTの例

`OpenIDX` はmagic 2051/2049の画像・ラベルを読み込み、拡張子に関わらずraw / gzip形式に対応します。内部ではuint8で保持し、`Get` 時に `[0,1]` のfloat32 CHW画像と整数ラベルを返します（Fingerprintは復号済み両ファイルのSHA-256）。

MNISTサンプルは `784→64→10` のMLP、cross-entropy、AdamW / SGD、OneCycleを使用します（学習はshuffle、評価は昇順）。`-download` は不足ファイルのみ取得し、ローカルIDXも利用可能です。

```sh
go run ./cmd/mnist-train -download -device cpu -steps 100 -stop-after 40 -train-limit 2000 -valid-limit 1000
go run ./cmd/mnist-train -device cpu -steps 100 -train-limit 2000 -valid-limit 1000 -resume out/mnist.safetensors.training.safetensors
go run ./cmd/mnist-train -device cuda -steps 100 -train-limit 2000 -valid-limit 1000 -out out/mnist-cuda.safetensors
```

既定値はデータ `out/mnist`、出力 `out/mnist.safetensors`、batch 64、seed 7です。`TrainLimit` / `ValidLimit` は0で全件、正の値で公式splitの先頭N件を使用します。`StopAfter` は完了更新の絶対数、`Steps` は再開時も固定するスケジュール全体の更新数です。学習checkpointは既定で出力パスに `.training.safetensors` を付与して保存します。

- `-workers` / `-prefetch`：学習と評価の両方に適用（既定0）。
- `-accum N`：N micro-batchごとに更新（既定1）。
- `-optimizer adamw|sgd`：optimizer選択。`-lr` はOneCycleのピーク学習率（0ならAdamW `0.005` / SGD `0.1`）。SGDのmomentum既定値は `0.9`、`-nesterov` で有効化、decayは `0.0001`。
- `-no-decay-suffix`：既定 `bias` で、一致するパラメータのdecayを無効化（空文字でグループ分け無効、グループ導入前のcheckpoint再開時は空文字を指定）。

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

名前・順序・形状を検証してからmodel / optimizerへコピーします。学習率・decay・beta・epsilon・OneCycleは復元されますが、モデルのtrain/evalモード、dropout seed、データとsamplerの同一性は呼び出し側で管理してください（推論用重みやoptimizer単体のファイルからは学習全体を再開できません）。

CUDAグラフ利用時は未完了launchを同期してから保存し、state復元後に再記録してください。optimizerスナップショットはデバイス上のステップ数を同期して読み取ります。

## 互換性の注意

- autograd、CUDA graph、BF16、割当プール、opレジストリは実験的な位置づけです。
- 1つの実行コンテキストで複数デバイスを混在させることは未対応です。
- safetensorsの読み込み中、デバイス転送に失敗すると途中で中断する場合があります。
- GoとPyTorchではサンプリング・初期化・dropoutの乱数ストリームが異なります。演算単体の比較では重みとマスクを固定し、学習全体の比較では収束品質で評価してください。