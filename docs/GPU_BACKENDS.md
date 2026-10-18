# GPUバックエンド実装の進捗

ローカルの`notes/gpu-backend-plan.md`に沿って、M0の共通APIとCUDAアダプタ、
M1のWindows/amd64向けVulkan基盤、M2のGEMMとview、M3のトランスフォーマー向け演算を実装した。LinuxのローダーとDirectMLは未実装。

| デバイス | 実行環境 | 現在の範囲 |
| --- | --- | --- |
| CPU / CUDA | 従来どおり | 既存の推論・学習。CUDAの新アダプタも利用可 |
| Vulkan | Windows/amd64、Vulkan 1.1以上の計算GPU | 基本演算、GEMM、view、正規化、Conv1d、構成演算による自己注意。推論のみ |
| Vulkan | Linux、その他のOS/アーキテクチャ | スタブ。`Available() == false` |
| DirectML | すべて | 未登録。`Available() == false` |

## 実装した範囲

- `tensor.Vulkan` / `tensor.DirectML`を追加。
- `gpu.Backend`：FP32の割当、転送、解放と、計画の基本forward演算。
- `gpu.Registry`：並行アクセス可能な登録と検索。登録の置き換えは禁止。
- `gpu.Buffer`：割当元のバックエンドインスタンスを記録し、取り違えと解放後の使用を検出する。
  `Close` / `Backend.Release`による重複解放は安全。
- サイズと積のoverflowを、割当・CUDAのint32への変換より前に検証する。
- `autograd/gpu_cuda_backend.go`：既存のCUDAカーネルとcuBLASを呼ぶ推論アダプタ。
  MatMulの転置4通り、Add、Mul、MulScalar、近似GELU、行単位Softmax、LayerNorm、
  GroupNorm、stride/padding/dilation付きConv1dを実装。
- 非CPU/CUDAデバイスでのbackward、AdamW、SGD、勾配クリップを明示的に拒否する。
- 未接続のデバイスの演算を、CPUやCUDAとして誤って実行しない。
- Vulkanのヘッドレスinstance/device/compute queueを実装。discrete GPUとcompute-only queueを優先し、CPU型のソフトウェアデバイスは選ばない。
- device-localバッファとhost-visible/coherentなステージングバッファ、転送とバリア、フェンスを実装。
- 埋め込みSPIR-VでAdd / Mul / MulScalarを実装。中間結果はGPU上に留まる。
- `autograd.Tensor`のNew / Zeros / ToHost / CopyFrom / Detach / Reshapeを共通バッファへ接続。
  既存のleaseでClose / ReleaseGraph / RetainDataとviewの寿命を管理する。
- サイズクラスのバッファプールは既定で256 MiBまでキャッシュする。`MemoryStats` / `SetPoolCacheLimit` / `ReleasePool`を公開。
- M2のGEMMは8×16の出力タイル、K方向16、128 invocations、共有メモリ1536 Bで計算する。
  転置の4パターンをspecialization constantで切り替え、パイプラインを保持する。
- 2次元だけでなくrank 6までのバッチGEMM、バッチ軸のブロードキャスト、offsetや非連続stride付きの入力に対応する。
  GEMMは入力viewを直接読むため、転置やバッチの展開行列を作らない。
- `gpu.View`と、追加のバックエンド能力を表す`gpu.ViewBackend` / `gpu.ViewMatMuler`を追加。Gather / ScatterはGPU上で実行する。
  Permute / Transpose / Slice / Expandは共有ストレージのメタデータを組み合わせる。連続な範囲のReshapeもコピーしない。
- M3のGELU（tanh近似とerf式）、Softmax、LayerNorm、GroupNormを埋め込みシェーダで実装。
  Softmaxと正規化は128 invocationsのworkgroupで行ごとに集約し、正規化の分散は平均からの差で求める。
- Conv1dはGPU上のim2colとGEMM、任意のbias加算で計算する。stride / padding / dilationに対応する。
  im2colの一時バッファは処理が成功・失敗した後にプールへ返す。デバイスのstorage buffer上限を超える場合は明示的に拒否する。
- Concatは入力ごとに出力viewへScatterする。非連続入力と空の入力片にも対応し、呼び出し元の入力スライスは変更しない。
- `ScaledDotProductAttention`はGEMM → スケール・加算mask → Softmax → GEMMで構成する。
  専用の融合カーネルではなく、中間のscoreとweightもGPU上に置く。評価時のDropoutは入力をそのまま返す。

## 現時点の仕様と制約

- `gpu`自体はドライバを読み込まない。`autograd`のimportでCUDAアダプタとVulkanが登録され、
  実際のドライバの読み込みは利用時まで遅延する。
- Windows/amd64では`vulkan-1.dll`をシステムディレクトリから読み、利用可能なGPUを選ぶ。
  `vulkan.DeviceInfo()`で選択結果を、`vulkan.Err()`で利用できない理由を取得できる。
  初期化は一度だけ行い、失敗を記憶する。ドライバを後から導入した場合はプロセスを再起動する。
- Vulkanが使えない環境や未登録のDirectMLでは、テンソル生成は`gpu.ErrUnavailable`。
  転送やsubmitでデバイスが失敗した場合も、以後`Available() == false`となる。
- `gpu.Get`は利用できないデバイスから別のデバイスへ自動で切り替えない。
  必要ならアプリがテンソルやモデルを生成する前にCPUを選ぶ。
- 共通APIの単位はfloat32の**要素数**。演算の形はバッファ全体と一致させ、出力は入力と別の割当を使う。
  `Download`だけは先頭の一部を読み戻せる。
- Conv1dは`[batch,time,in_channels]`、重みは`[out_channels,in_channels,kernel]`。
  biasはnilにできる。stride/dilationは正の値を明示する。
- 共通の`Backend.GELU`はtanh近似。任意の`gpu.ExactGELUBackend`はerf式のGELUを表し、Vulkanが実装する。
  `autograd.GELU(x, false)`をtanh近似へ読み替えることはない。erf自体は誤差上限1.5e-7の多項式近似を使う。
  CUDAアダプタのGEMMはFP32で、既存のBF16設定を参照しない。
- 新GPUの入力やパラメータの`RequiresGrad`は保持する。trueの入力を使うforwardには、
  その入力の`ExecutionContext.NoGrad`が必要。記録を要求した場合は出力を割り当てる前に拒否する。
  BatchNormの統計更新とoptimizerは、NoGrad中でも拒否する。
- Vulkanの対応演算はAdd / Mul / MulScalar、MatMul、GELU、Softmax、LayerNorm、GroupNorm、Conv1d、Concat、
  構成演算による自己注意、GPU間コピーとrank 6までのview。`LinearLayer`の転置付き投影とbiasのブロードキャストも使える。
  Softmaxは任意の軸を受け付け、非最終軸ならGPU上で軸を移して計算し、逆順に並べ替えたviewを返す。
  BiasGELU / BiasResidualも基本演算の組み合わせで実行し、既存の`TransformerEncoderLayer`を評価モードとNoGradで使える。
  LogSoftmax、FlashAttentionなどは引き続き`gpu.ErrUnsupported`。
  明示的に`AttentionAlgorithm="flash"`を選んだ場合も拒否する。学習時のDropoutはNoGrad中でも`gpu.ErrInferenceOnly`。
  旧`tensor.Tensor` / `nn`経路はCPU/CUDAのままで、新GPUは`autograd`から使う。
- viewのstride/offsetは要素数単位で、負のstrideは未対応。
  非連続なToHost / Contiguous / Detachや、要素演算のブロードキャストはGatherで実体化するが、CPUへは戻さない。
  非連続なCopyFromはScatterで共有ストレージを更新し、全別名のversionを進める。
  Expanded viewなど、書き込み先が重複するviewは更新前に拒否する。
- Vulkanはバックエンド内のmutexでqueue・command pool・descriptorを同期するため、CUDAのようなOSスレッド固定は不要。
  ただし同じTensorやモデルの変更を複数goroutineから行う際の同期は、引き続き呼び出し側の責任。
- 現在はsubmitごとにフェンスを待つ。CPUより速いことやM6の2倍という性能目標は、まだ主張しない。
  command bufferのまとめと性能調整は後続の作業。
- `vulkan.NewBackend()`は独立したdevice/queueを作る。独立したバックエンドの`Close()`は全バッファの解放後に行う。
  `autograd`が使う登録済みの既定バックエンドはプロセス寿命で、`Close()`できない。キャッシュは`ReleasePool()`で返せる。
- CUDAの演算には従来どおり`autograd.NewCUDAContext` / `NewCUDAContextOn`によるOSスレッドの固定が必要。
  演算の全バッファは同じdevice/streamで確保する。異なる選択で実行すると`gpu.ErrBackendMismatch`。
- CUDAアダプタの転送・zero・解放は割当元のdevice/streamを使い、呼び出し元の選択を復元する。
  読み戻しは計算streamの完了を待ち、解放した割当は元のstreamのプールへ返す。
  streamは、そのstreamで確保した全バッファを解放するまで生かしておく。
- 既存のCPU/CUDAのTensor演算・学習・BF16・graph captureの経路は置き換えていない。

## 使用例

```go
execution := autograd.NewExecutionContext()
x, err := execution.New([]float32{1, 2, 3}, []int{3}, tensor.Vulkan, true)
if err != nil { panic(err) } // 必要ならモデル構築前にCPUを選び直す
defer x.Close()
var y *autograd.Tensor
execution.NoGrad(func() {
    y = autograd.MulScalar(autograd.Add(x, x), 0.5)
})
defer y.ReleaseGraph()
values, err := y.ToHost() // 中間結果の読み戻しは不要
```

既存の`TransformerEncoderLayer`を使う場合は、`layer.Training = false`と
`layer.Attention.Training = false`を設定し、`execution.BindModule(&layer.Module)`で
全パラメータを同じcontextに結び付けてから、`execution.NoGrad`内でForwardする。
学習フラグが残っていれば、中間バッファを割り当てる前に拒否する。

実行できる小さな例も用意した。起動時にGPU名・APIバージョン・compute queueを表示し、出力をCPUと照合する。
CPUへフォールバックした場合は、実際のデバイスをCPUと表示する。GPU上での失敗を暗黙にCPUでやり直すことはない。

```text
go run ./cmd/vulkan-infer -n 100003
go run ./cmd/vulkan-infer -linear
go run ./cmd/vulkan-infer -cpu-fallback
```

## シェーダの再生成（開発者向け）

通常の`go build` / `go test`は、同梱の`vulkan/shaders/{pointwise,view,gemm,reduce,conv}.spv`を使う。SDKやコンパイラは不要。
ソースは隣の`.comp`ファイルにある。現在のバイナリはKhronos glslang 16.6.0で生成し、Vulkan 1.1を対象にしている。
ソースを変えた場合だけ、`glslang`をPATHに置いて次を実行する。

```text
go generate ./vulkan
```

## テスト

```text
go test ./gpu
go test ./autograd -run 'Test(GPUDeviceRolloutGuards|CUDAInferenceBackend)$' -v
go test ./vulkan -v
go test ./autograd -run 'Test(InferenceGPU|VulkanTensorInference)' -v
go test ./...
go vet ./...
```

Windows/amd64では`CGO_ENABLED=0`で全パッケージのテストが通り、`go vet ./...`も成功した。
CUDAアダプタとVulkanの一致テストも実機で実行して成功している。
VulkanはRTX 3060 Ti、API 1.4.341、compute queue 2で検証した。
`cmd/vulkan-infer -n 100003`の最大絶対誤差は0だった。
`cmd/vulkan-infer -linear`の`[2,768,74] @ [74,768] + bias`もCPUとの最大絶対誤差は0だった。
Vulkanのテストは`-gcflags=all=-d=checkptr=2`でも成功した。
GPU実機テストをスキップさせずに検証する場合は、`GOGRAD_REQUIRE_VULKAN=1`を設定する。
Linux/amd64とmacOS/arm64向けにも、cgoなしで`autograd`のテストバイナリをクロスコンパイルした。
これらはスタブ経路のビルド確認で、GPUの実行確認ではない。
Windows/386の`autograd`は、既存のCUDAパッケージの2 GiB定数がintでoverflowするためビルドできない。
今回のVulkanローダーもWindows/amd64のみを対象にしている。

共通層はドライバなしで登録、並行検索、所有権、二重解放、サイズ検証をテストする。
CUDAが利用できる場合は全アダプタ演算をCPUと照合する。ゼロサイズ、GEMMの転置、
部分タイル、Conv1dのbias有無、streamの切替・復元、解放先のプール、使用中メモリの増加も確認する。
CUDAがない場合はアダプタの実機テストだけをスキップする。
Vulkanも利用できない環境では実機テストをスキップするが、共通ストレージの寿命管理、失敗時の出力回収、
NoGradの復帰、割当・読み戻し回数の確認、ドライバ失敗後の回収、SPIR-Vの形式検査はドライバなしで実行する。
Vulkan実機では0要素・部分workgroup・2次元dispatch・複数goroutine・プールの上限と解放も確認する。
GEMMは768×768×74と768³を含め転置4通りをCPU SGEMMと照合し、`atol=1e-4`、`rtol=1e-4`で検証する。
バッチ軸のブロードキャスト、rank 6、部分タイル、K=0、slice/transpose/expandを通した読込、
viewへの書き込み、RetainData、親の早期Close、失敗時の一時バッファ回収も確認する。
M3では3072列までのSoftmax / LayerNorm、GroupNorm、Conv1dのbias有無とstride / padding / dilation、
非連続入力、任意軸のSoftmaxとConcatをCPUと照合する。
`TestInferenceGPUTransformerNetwork`はConv1d → GroupNorm → 2-head causal自己注意 → LayerNorm → FFNを
3回実行し、`atol=1e-4`、`rtol=1e-4`とコサイン0.999以上を検証する。RTX 3060 Tiではコサイン1.0だった。
既存の`TransformerEncoderLayer`でも、評価モードのforwardを3回繰り返してCPUと照合し、
各回のReleaseGraph後にGPUの使用中メモリが元へ戻ることを確認する。
ドライバなしでも、NoGradの伝播、暗黙のdownloadがないこと、失敗時に呼び出し元の入力graphを壊さないことを確認する。
Conv1dはim2col・GEMM・biasの各段階で失敗を注入し、一時バッファを含めて回収されることを確認する。

## 残る作業

1. M1のLinuxローダー。`dlopen`と関数ポインタ呼び出しを、cgoなしの呼出規約を含めて検証する。
2. M4/M5のDirectML実装。
3. M6の記録のまとめ、キャッシュ調整と実測ベンチマーク。
