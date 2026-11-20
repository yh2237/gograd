# APIリファレンス

CPU / CUDAの学習・推論および共通ユーティリティのAPI仕様です。Vulkan / DirectML固有の対応範囲と制約は [GPUバックエンド](GPU_BACKENDS.md) を参照してください。

## パッケージ別API一覧

| パッケージ | 主なAPI | 用途 |
| --- | --- | --- |
| `tensor` | `Device`（`CPU`, `CUDA`, `Vulkan`, `DirectML`） | デバイス指定 |
| `autograd` | `New`, `Zeros`, `ToHost`, `CopyFrom`, `Detach`, `Backward`, `BackwardWithOptions`, `ReleaseGraph`, `Close` | テンソル生成・転送・逆伝播・寿命管理 |
| `autograd` | `Add`, `Sub`, `Mul`, `Div`, `Max`, `Min`, `AddScalar`, `SubScalar`, `MulScalar`, `DivScalar`, `PowScalar`, `Exp`, `Log`, `Sqrt`, `Sigmoid`, `Tanh`, `Sin`, `Cos`, `Softplus`, `Erf`, `Log1p`, `Expm1`, `Abs`, `ReLU`, `GELU`, `Clamp` | 要素演算・broadcast |
| `autograd` | `Reshape`, `Unsqueeze`, `Squeeze`, `Flatten`, `Transpose`, `Permute`, `Slice`, `Expand`, `Contiguous`, `Concat`, `Stack`, `Sum`, `Mean`, `MatMul` | 形状・view・reduction・行列積 |
| `autograd` | `Embedding`, `EmbeddingWithPadding`, `Softmax`, `LogSoftmax`, `Dropout`, `ScaledDotProductAttention`, `CrossEntropy` | Transformer・分類演算 |
| `autograd` | `Conv1d`, `Conv1dGEMM`, `Conv1dGEMMStride`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`, `BatchNorm`, `LayerNorm`, `GroupNorm`, `RMSNormLast` | 畳み込み・pooling・正規化 |
| `autograd` | `RotaryHalf`, `GeGLU`, `SwiGLUProjection`, `KeyPaddingAttention` | RoPE・GLU・マスク付きattention |
| `autograd` | `LinearLayer`, `EmbeddingLayer`, `Conv1dLayer`, `Conv2dLayer`, `MaxPool2dLayer`, `AvgPool2dLayer`, `AdaptiveAvgPool2dLayer`, `BatchNormLayer`, `LayerNormLayer`, `GroupNormLayer`, `RMSNormLayer`, `DropoutLayer`, `GELULayer`, `FlattenLayer`, `MultiheadAttention`, `TransformerEncoderLayer`, `Sequential` | 層・モジュール |
| `autograd` | `MSELoss`, `L1Loss`, `SmoothL1Loss`, `BCEWithLogitsLoss`, `MaskedLoss` | 損失関数 |
| `autograd` | `NewExecutionContext`, `Bind`, `BindModule`, `NoGrad`, `Autocast`, `SetOptions`, `SetCUDA`, `WithCUDA`, `NewCUDAContext`, `NewCUDAContextOn`, `RunBranches` | 実行コンテキスト・CUDA制御 |
| `autograd` | `Module`, `StateDict`, `LoadStateDict`, `SaveSafeTensors`, `SaveSafeTensorsMetadata`, `LoadSafeTensors`, `OpenSafeTensorFile`, `LoadSafeTensorsStream` | state管理・safetensors I/O |
| `autograd` | `Optimizer`, `CheckpointOptimizer`, `NewAdamW`, `NewAdamWGroups`, `NewSGD`, `NewSGDGroups`, `PartitionParameters`, `ClipGradNorm`, `NewOneCycle`, `GradAccumulator`, `TrainingLoop`, `SaveTrainingCheckpoint`, `LoadTrainingCheckpoint` | 最適化・学習ループ・checkpoint |
| `cuda` | `SetCurrentStream`, `CurrentStream`, `WithStream`, `NewEvent`, `Capture`, `MemoryStats`, `ResetAllocationPeak`, `SetPoolCacheLimit`, `ReleasePool` | stream / event / graph / プール制御 |
| `data` | `Dataset`, `SliceDataset`, `NewSubset`, `RandomSplit`, `NewIndexSampler`, `NewLoader`, `Stack`, `PadSequences`, `OpenIDX`, `TransformDataset`, `ParseNumericCSV`, `FitStandardizer`, `ReadWAV`, `WriteWAV` | ホストデータ処理・augmentation・prefetch |
| `metrics` | `Mean`, `Accuracy`, `TopKAccuracy`, `Regression`, `ConfusionMatrix` | ホスト評価指標 |
| `dsp` | `NewFFT`, `HannPeriodic`, `NewSTFT`, `Resample` | FFT / STFT / リサンプリング |
| `torchpth` | `Read` | PyTorch zip checkpointのFP32読み込み |
| `torchrng` | `New` | PyTorch CPU乱数列の再現 |

## 実行コンテキストとCUDA管理

- `ExecutionContext` は勾配記録・`ExecutionOptions`・CUDAデバイス/stream設定をスコープ分離する。異なる明示的コンテキスト間の演算はpanicする（未bindの定数は参加可能）。
- モデル構築後はforward前に `BindModule` を呼ぶ。既存グラフや別コンテキストの割当に対するbindはエラー。
- `NoGrad` と `Autocast` はコンテキスト全体に作用し、入れ子・panic時に元の状態を復元する。
- `ExecutionOptions` の既定値は `BF16Autocast: false, AttentionAlgorithm: "auto"`。`AttentionAlgorithm` は `"materialized"` / `"flash"` / `"auto"`（空文字は `"auto"`、不正値はエラー）。
  - `auto` はCUDA上で実体化スコアscratchが1 GiB以上かつhead次元128以下の場合のみtiled（flash）経路を選択する。
  - BF16はCUDA GEMMのオペランドとシャドウにのみ使用し、累積・勾配・master weight・optimizer state・Conv2d・flash attentionはFP32。
- パッケージ直下の `New` / `Zeros` / `NoGrad` / `AttentionAlgorithm` は未bindテンソル向けのグローバル既定スコープを使用する。並行workerは独立した `ExecutionContext` とモデルstateを持つこと。
- CUDAのデバイス・stream・capture状態はOSスレッド単位で管理される。CUDA処理中は `NewCUDAContext` / `NewCUDAContextOn` でOSスレッドを固定する。`SetCUDA` の設定は `WithCUDA` で一時適用される。
- `RunBranches` は各分岐を別streamへenqueueし、カレントstreamへeventで合流させる。`Capture` は明示的stream上でのみ実行でき、同一スレッドでの入れ子は拒否される。
- CUDAメモリプールは既定上限2 GiB（`SetPoolCacheLimit` / `ReleasePool` / `MemoryStats` で管理）。使用中およびCUDAグラフ予約中の割当は解放しない。

## ストレージとグラフの寿命

| 操作 | 仕様 |
| --- | --- |
| `Close` | ハンドルの公開利用を終了（冪等）。依存中のviewやグラフが参照する割当は内部保持される |
| `ReleaseGraph` | 中間ノードと内部の一時末端を回収する（呼び出し側所有の入力・パラメータは `Close` が必要） |
| `RetainData` | `ReleaseGraph` / `Backward` 後も値と割当を保持（履歴は保持しない。不要時に `Close`） |
| `RetainGrad` | 中間テンソルの勾配を保持（不要時に `Close`） |
| `Detach` | 独立ストレージへコピーして履歴を切断 |
| `Backward` / `BackwardWithOptions` | 既定では履歴を消費し未使用CUDA中間値を回収する。`RetainGraph: true` で再実行可能 |

- `CopyFrom` とeager optimizer更新はストレージのversionを進める。`Backward` は勾配書き込み前に保存した入力・出力のversionを検証する（`Data` スライスへの直接書き込みや外部CUDA書き込み・グラフ再生は追跡対象外）。
- `Data` と `Buffer()` は借用参照。`Tensor` と `IndexBuffer` はポインタで扱い値コピーしない。

## 演算仕様

### 要素演算・形状・Transformer演算

- `Max` / `Min` は同値時に勾配を右辺へ送る。`Clamp` は境界を含めて勾配を通す。`Sqrt` の0での微分は無限大。
- `Softplus` は `max(x,0)+log1p(exp(-|x|))`、`Log1p` / `Expm1` は0近傍の桁落ちを回避する。
- `Flatten(a, start, end)` は両端を含む（負軸可、スカラーは `[1]`）。
- `Dropout` は `seed` と要素indexのhashでマスクを生成する（PyTorch乱数列とは異なる）。
- `RotaryHalf(x, theta)`：`[batch,heads,seq,headDim]`（偶数 `headDim`、正の `theta`）に半回転RoPEを適用。
- `GeGLU(x)`：最終軸を2分割し `GELU(前半, false) * 後半`。
- `SwiGLUProjection(x, w1, w2, w3)`：`SiLU(x*w1^T) * (x*w3^T)` に `w2^T` を掛ける（重みは `[out,in]`）。
- `KeyPaddingAttention(q, k, v, valid, radius)`：真偽値キーマスクを加法的マスクへ変換。`radius < 0` は全attention、0以上は窓外をマスク。全キーがマスクされたクエリはpanic。
- `RMSNormLast(x, weight, eps)`：最終軸のRMS正規化とチャネル重み乗算。

### Conv2d / Pooling / BatchNorm

- **Conv2d**：NCHW入力、重み `[out,in/groups,kH,kW]`、任意bias `[out]` のcross-correlation。`Stride` / `Dilation` の全0は `[1,1]`、`Groups` の0は `1`、`Padding` は各軸非負の対称パディング。出力長は `floor((in + 2*pad - dil*(k-1) - 1)/stride) + 1`。
- **MaxPool2d / AvgPool2d / AdaptiveAvgPool2d**：NCHW入力。`Stride` の全0は `KernelSize`、`Dilation` の全0は `[1,1]`、`Padding <= kernel/2`。
  - `MaxPool2d`：パディングは `-Inf` 扱い。同値はrow-major先頭、NaNは末尾を選択。
  - `AvgPool2d`：既定でパディングを除数に含む（`ExcludePad: true` で有効要素のみ、`DivisorOverride > 0` で除数上書き、`CeilMode` の超過位置は常に除外）。
  - `AdaptiveAvgPool2d`：各binは `[floor(pos*in/out), ceil((pos+1)*in/out))`。
- **BatchNorm**：rank 2〜6（既定チャネル軸1、`ChannelsLast: true` で最終軸）。`Eps` の0は `1e-5`、`Momentum` は `[0,1]`。
  - 学習時はチャネルあたり2サンプル以上が必要（出力は偏り分散、running更新は不偏分散）。`weight` / `bias` は各々任意、`runningMean` / `runningVar` は両方nilまたは連続・勾配不要の `[channels]`（評価時は必須、入力・stateとalias不可）。
  - `BatchNormLayerOptions` の既定はaffine有効・running有効・`Eps=1e-5`・`Momentum=0.1`（`Momentum` はポインタ型でnilが `0.1`、`&0` が凍結）。`CumulativeMomentum` は `1/num_batches_tracked` を使用。`num_batches_tracked` はFP32保持のため整数精度は `2^24` まで。
  - `Train` / `Module.Train` は統計更新とDropoutを制御し、`NoGrad` は勾配記録のみを制御する。

### 損失関数

- `MSELoss`、`L1Loss`、`SmoothL1Loss`（`|d|<1` で `0.5*d^2`、以外で `|d|-0.5`）、`BCEWithLogitsLoss`（`max(x,0)-x*y+log(1+exp(-|x|))`、targetは `[0,1]`）はスカラーを返す。
- `LossOptions.Reduction` は `ReductionMean`（既定）または `ReductionSum`。`Weight` 指定時、MSE / L1のmeanは重み和で除算（0重み除外）、BCEのmeanは要素数で除算。
- `MaskedLoss` はrank 3専用。NaNフレームを除外し、有効フレーム数 × channelsで平均する（有効0件時は損失0）。

## Optimizer・勾配蓄積・学習再開

- `Optimizer` は `Step` / `ZeroGrad` / `LearningRate` / `SetLearningRate` / `Close` を定義する。`CheckpointOptimizer` は `adamw` / `sgd` タグ付きスナップショットを提供する。
- **AdamW**：decoupled weight decay、FP32モーメント、単一ステップカウンタ。
- **SGD**：coupled weight decay、`Momentum`、`Dampening`、`Nesterov`（`Momentum > 0 && Dampening == 0` 必須）、`Maximize`（生勾配のみ反転）。初回momentumはdampeningなし。勾配nilのパラメータは更新スキップ、0勾配は更新対象。
- **パラメータグループ**：`NewSGDGroups` / `NewAdamWGroups` は全パラメータを重複なく1回ずつ含む `ParameterGroupSpec{Name, Params, LRScale, WeightDecay}` を受け取る（`LRScale == 0` は基準学習率1倍、`NewSGDGroups` では `options.WeightDecay == 0` 必須）。
- **CUDAグラフ連携**：`PrepareGraph` / `GraphStepCount` / `SyncGraphStepCount` を使用する。`ZeroGrad` はCUDAグラフ用勾配バッファを解放せず0埋めする。state保存前・eager復帰時はステップ数を同期し、構成変更時は古いグラフを破棄する。
- `GradAccumulator.Backward` は `1/groups` を初期勾配として逆伝播する。端数のグループにも同じ係数を使い、逆伝播後も損失の値は読み取れる。
- `ClipGradNorm` はCPUではクリップ前ノルム、CUDAではホスト同期を避けるため `NaN` を返す（CUDA上のノルム取得は `GradientNormToHost`）。
- **TrainingLoop**：group先頭で `ZeroGrad`、更新完了時に `Optimizer.Step` と `Schedule.Step` を1回呼ぶ。`Accumulate >= 2` で途中キャンセル時は未完了勾配を破棄し、Samplerをgroup先頭へ戻して `Save` を呼ぶ（running統計・外部乱数状態は呼び出し側管理）。
- **checkpoint**：`SaveTrainingCheckpoint` / `LoadTrainingCheckpoint` は `gograd-training-1` 形式のsafetensorsでmodel / optimizer / OneCycle / metadataをアトミック保存・検証復元する（`optimizer_kind` 省略時はAdamW互換）。train/evalモード・dropout seed・データ同一性は呼び出し側で管理する。

## ホストデータ処理・評価指標・ユーティリティ

- **`data`**：
  - `IndexSampler` はseed付きPCGでshuffleし、`Peek` / `Advance` / `State` / `LoadState` を提供する。
  - `Loader` は `Dataset.Get`・collate・キャンセル確認がすべて成功した後にのみカーソルを進める（失敗時は再試行可能、callbackのEOFは `UnexpectedEOF`）。`Workers > 0` で `Dataset.Get` のみを並列化し、`Prefetch` 上限内でsampler順へ整列して返す（collateは `Next` 呼び出しスレッドで直列実行）。`State` は先読み分を破棄して配達済み位置のみ保存し、`Workers` / `Prefetch` を変更して再開可能。
  - `PadSequences` は末尾次元一致の系列を最大長または指定 `Length`（切り詰めは `Truncate: true`）まで埋め、`Lengths` と `[batch,time]` のbool `Mask` を返す。
  - `TransformDataset` の組み込み変換（`RandomHorizontalFlipFloat32` / `RandomVerticalFlipFloat32` / `AddGaussianNoiseFloat32`）は `(seed, index)` のみから乱数を生成するためworker数・読み込み順に依存しない（新規コードは `uint64` seed版を使用）。
  - `ParseNumericCSV`（有限値検証・行番号付きエラー）、`FitStandardizer`（float64累積、定数特徴量のscale=1）、`OpenIDX`（raw / gzip自動判別、uint8保持、`[0,1]` float32出力）、`ReadWAV` / `WriteWAV`（非圧縮PCM 8/16/24/32-bit、IEEE float 32/64-bit、`WAVE_FORMAT_EXTENSIBLE`）。
- **`metrics`**：ゼロ値から利用可能（並行書き込み不可）。`Mean`、`Accuracy`（同点は小クラス優先、`ignoreIndex` 除外）、`TopKAccuracy`（境界同点は正解）、`Regression`（MSE / RMSE / MAE / R2、float64累積）、`ConfusionMatrix`（初回更新でクラス数固定、`MacroF1` はsupport > 0クラス平均）。非有限値・形状不一致は状態を変更せずエラー。
- **`dsp`**：`NewFFT`（2の冪長、float64累積）、`HannPeriodic`、`NewSTFT`（`1 <= hop <= filter`、反射パディング、`[bins][frames]` 振幅・位相）、`Resample`（ポリフェーズHann窓sinc）。
- **`torchpth` / `torchrng`**：`torchpth.Read` はFloat / Half / BFloat16をfloat32へ変換して読み込み（他dtypeはエラー）。`torchrng.New(seed)` はPyTorch CPU乱数列を再現（`State` / `SetState` 対応、非スレッドセーフ）。
