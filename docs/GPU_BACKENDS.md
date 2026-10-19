# GPUバックエンド実装の進捗

ローカルの`notes/gpu-backend-plan.md`に沿って、M0の共通APIとCUDAアダプタ、
M1のWindows/amd64向けVulkan基盤、M2のGEMMとview、M3のトランスフォーマー向け演算、
M4のWindows/amd64向けDirectML基盤、M5のDirectML向けトランスフォーマー演算を実装した。
Linux/amd64にもcgoなしのVulkanローダーを実装した。ただしLinuxでの実GPU計算は未検証。条件と検証範囲は[Linux版Vulkan](GPU_LINUX.md)を参照。
M6では共通の推論ベンチマーク、DirectMLのviewコピー削減と複合演算内のcommand記録のまとめ、Vulkanのbias融合演算を追加した。計測条件と実測は[性能検証](GPU_BACKEND_PERFORMANCE.md)にまとめている。

| デバイス | 実行環境 | 現在の範囲 |
| --- | --- | --- |
| CPU / CUDA | 従来どおり | 既存の推論・学習。CUDAの新アダプタも利用可 |
| Vulkan | Windows/amd64、Vulkan 1.1以上の計算GPU | 基本演算、GEMM、view、正規化、Conv1d、構成演算による自己注意。推論のみ |
| Vulkan | Linux/amd64 | ローダーと演算経路を実装。Windowsと共有する推論機能。実GPU計算は未検証 |
| Vulkan | その他のOS/アーキテクチャ（Linux/arm64、Androidを含む） | スタブ。`Available() == false` |
| DirectML | Windows/amd64、DirectMLとD3D12対応の計算GPU | 基本演算、GEMM、view、正規化、Conv1d、構成演算による自己注意。推論のみ。erf式GELUはfeature level 5.1以上 |
| DirectML | その他のOS/アーキテクチャ | スタブ。`Available() == false` |

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
- Windows/Linux amd64でABI型・デバイス選択・バッファ・command記録・シェーダを共有する。
  Windowsの呼出しは標準の`syscall`、Linuxは`purego v0.11.2`の`dlopen` / SysV関数呼出しを使う。
  LinuxでもビルドにcgoやCコンパイラは不要。ただし標準のlibc / libdlとVulkanローダー・ICDが実行時に必要。
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
- M4のDirectMLは、純GoのCOM呼び出しでDXGI / D3D12 / DirectMLを使う。
  ハードウェアアダプタだけを列挙し、D3D12とDirectMLが作れるデバイスのうち専用ビデオメモリの多いものを優先する。
  WARPなどのソフトウェアアダプタは選ばない。
- DirectMLのテンソルはD3D12 DEFAULTヒープに置き、UPLOAD / READBACKヒープ経由で転送する。
  リソースを256バイト単位のサイズクラスでプールし、キャッシュ上限は既定で256 MiB。
  小さなバッファでもD3D12の配置単位のメモリを使うため、上限と使用量はWidthではなく`GetResourceAllocationInfo`の確保サイズで数える。
- DirectMLのAdd / Mulは要素演算子、MulScalarはIdentity演算子のScaleBias、GEMMはDML_GEMMを使う。
  初回は演算子をコンパイルしてGPU上で初期化し、形・転置・スカラー値ごとに再利用する。
  キャッシュは最大64件かつtemporary / persistentリソースの合計256 MiBまで。`ReleaseOperators`で明示的に破棄できる。
- M5ではDirectMLにも`gpu.ViewBackend` / `gpu.ViewMatMuler`を実装した。
  offsetが16バイト境界に揃い、rank 4以下のGEMMは、転置やゼロstrideのブロードキャストをテンソル記述子で直接読む。
  それ以外のviewはGPU上で連続化し、rank 6までのバッチ軸をまとめてバッチGEMMを実行する。
- DirectMLのGatherは、16バイト境界かつrank 4以下の非連続viewなら、ScaleBiasを設定しないIdentity演算子で一括コピーする。
  それ以外のGatherとScatterはD3D12のバッファ間コピーを使い、末尾の連続な範囲をまとめて記録する。
  offsetが16バイト境界に揃わなくても扱える。Scatterは指定した要素だけを書き、viewの隙間にある別名の値を壊さない。
  Concatも共通のScatter経路で実行する。CPUが扱うのはコピー位置のメタデータだけで、テンソル値は読み戻さない。
- 追加能力の`gpu.ViewBinaryBackend`をDirectMLに実装した。対応するviewのAdd / Mulは、
  biasなどのゼロstrideも記述子で表して直接計算し、展開用バッファを作らない。
  未対応のrankやalignmentなら、共通層が従来のGPU上のGather経路を使う。CPU計算へのフォールバックではない。
- DirectMLのLayerNorm / GroupNormはMeanVarianceNormalizationで正規化した後、GPU上でチャネルごとの重みとbiasを適用する。
  GroupNormは`[batch,groups,frames,channels_per_group]`のstride付き記述子で各グループを独立に正規化する。
  Softmaxは行単位のDML_ACTIVATION_SOFTMAXを使う。
- DirectMLのConv1dはDML_CONVOLUTIONのcross-correlationを使う。
  入出力の`[batch,time,channels]`をstride付きNCHW記述子で表し、ホストでの並べ替えは行わない。
- DirectMLのGELUには2つの経路がある。tanh近似はGPU上の積・スケール・加算・tanhで構成する。
  erf式はfeature level 5.1以上のDML_ACTIVATION_GELUを使い、対応しない環境では`gpu.ErrUnsupported`を返す。

## 現時点の仕様と制約

- `gpu`自体はドライバを読み込まない。`autograd`のimportでCUDAアダプタ・Vulkan・DirectMLが登録され、
  実際のドライバの読み込みは利用時まで遅延する。
- Windows/amd64では`vulkan-1.dll`をシステムディレクトリから読み、利用可能なGPUを選ぶ。
  `vulkan.DeviceInfo()`で選択結果を、`vulkan.Err()`で利用できない理由を取得できる。
  初期化は一度だけ行い、失敗を記憶する。ドライバを後から導入した場合はプロセスを再起動する。
- 各バックエンドが使えない環境では、テンソル生成は`gpu.ErrUnavailable`。
  転送やsubmitでデバイスが失敗した場合も、以後`Available() == false`となる。
- `gpu.Get`は利用できないデバイスから別のデバイスへ自動で切り替えない。
  必要ならアプリがテンソルやモデルを生成する前にCPUを選ぶ。
- 共通APIの単位はfloat32の**要素数**。演算の形はバッファ全体と一致させ、出力は入力と別の割当を使う。
  `Download`だけは先頭の一部を読み戻せる。
- Conv1dは`[batch,time,in_channels]`、重みは`[out_channels,in_channels,kernel]`。
  biasはnilにできる。stride/dilationは正の値を明示する。
- 共通の`Backend.GELU`はtanh近似。追加能力の`gpu.ExactGELUBackend`はerf式のGELUを表し、VulkanとDirectMLが実装する。
  `autograd.GELU(x, false)`をtanh近似へ読み替えることはない。Vulkanのerf自体は誤差上限1.5e-7の多項式近似を使う。
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
  非連続なToHost / Contiguous / Detachや、必要なブロードキャストはGPU上のGatherで実体化する。
  対応するDirectMLの要素演算はviewを直接読み、不要な実体化を避ける。中間値をCPUへ戻すことはない。
  非連続なCopyFromはScatterで共有ストレージを更新し、全別名のversionを進める。
  Expanded viewなど、書き込み先が重複するviewは更新前に拒否する。
- Vulkanはバックエンド内のmutexでqueue・command pool・descriptorを同期するため、CUDAのようなOSスレッド固定は不要。
  ただし同じTensorやモデルの変更を複数goroutineから行う際の同期は、引き続き呼び出し側の責任。
- DirectMLもバックエンドのmutexでCOMオブジェクトとqueueを同期し、OSスレッド固定は不要。
  DLLはシステムディレクトリの`dxgi.dll` / `d3d12.dll` / `DirectML.dll`だけを読み込む。
  各submitでリソース状態の遷移、UAVバリア、フェンス待ちを行い、完了前にバッファ・記述子を再利用しない。
  `directml.DeviceInfo()`でアダプタとfeature levelを、`directml.Err()`で利用できない理由を取得できる。
- DirectMLでもautogradの非連続view、バッチGEMM、ブロードキャスト、正規化、GELU、Softmax、Conv1d、Concatを使える。
  既存の`LinearLayer`、評価モードの`TransformerEncoderLayer`にも接続した。
  Encoderはerf式GELUを使うためfeature level 5.1以上が必要。未対応でも近似式やCPU計算へ暗黙には切り替えない。
  Vulkanと同様に、LogSoftmaxとFlashAttentionは未対応、学習・統計更新・optimizerは明示的に拒否する。
- `directml.MemoryStats()`はテンソルとステージングのリソースを数え、演算子内部のヒープは含めない。
  キャッシュ済み演算子の件数とtemporary / persistentリソースは`directml.CachedOperatorStats()`で別に確認できる。
  未キャッシュの一時的な演算子とドライバ内部の管理領域を含む、プロセス全体のVRAM量ではない。
- 現在はsubmitごとにフェンスを待つ。`gpu.Synchronizer`をVulkan / DirectMLに実装し、読み戻しなしで完了を確認できる。
  既存の各演算がmutex内でフェンスを待つため、Synchronizeは進行中の演算が終わるのを待ち、新しい空submitは増やさない。
  実測した常駐データの条件ではCPUより高速だったが、全GPU・全サイズでの速度や初回実行の高速化を保証しない。
  DirectMLのtanh近似GELUは9 dispatch、LayerNorm / GroupNormは3 dispatchを1つのcommand listへ記録する。
  warmup後はそれぞれ1回のsubmit / フェンス待ちになり、演算間の状態遷移とUAVバリアは維持する。
  初回の演算子初期化は別submit。バッチ内で繰り返す演算は独立したdescriptorスロットを使い、通常のcache上限に数える。
  バッチ準備中の演算子は退避せず、cacheに入らない演算子もフェンス完了まで保持する。
  forward全体のcommand記録のまとめ、融合演算、複雑なviewのcopy削減は引き続き改善対象。
- Vulkanは追加能力の`gpu.FusedBiasBackend`を実装し、`BiasGELU`（erf式）と`BiasResidual`を
  それぞれ1 dispatch / submitで実行する。biasの展開バッファと合成演算の中間出力を作らない。
  biasは最後の軸に対応する1次元、residualは入力と同じshapeの場合に適用する。
  非連続入力は必要なものだけGPU上で連続化する。それ以外のブロードキャストや能力未実装のバックエンドは従来のGPU合成経路を使う。
  `ErrUnsupported`だけが合成経路への切替条件で、実行エラーは伝搬する。CPUへの切替ではない。
  残差加算は`(x + bias) + residual`の順序を保ち、erf式をtanh式へ置き換えない。
- `vulkan.NewBackend()`は独立したdevice/queueを作る。独立したバックエンドの`Close()`は全バッファの解放後に行う。
  `autograd`が使う登録済みの既定バックエンドはプロセス寿命で、`Close()`できない。キャッシュは`ReleasePool()`で返せる。
- DirectMLにも独立した`NewBackend` / `Close`と、既定バックエンドの`ReleasePool` / `ReleaseOperators`を用意した。
  独立したバックエンドは、全テンソルを解放してからCloseする。
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
go run ./cmd/directml-infer
go run ./cmd/directml-infer -encoder
go run ./cmd/directml-infer -m 7 -n 11 -k 13 -cpu-fallback
go run ./cmd/gpu-infer-bench -warmup 2 -iters 7 -json
go run ./cmd/gpu-infer-bench -workloads gelu,layernorm,encoder -warmup 2 -iters 7 -json
go run ./cmd/gpu-infer-bench -workloads bias_gelu,bias_residual,ffn,encoder -warmup 2 -iters 7 -json
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
go test ./directml -v
go test ./autograd -run TestDirectMLTensorInference -v
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
DirectMLは同じRTX 3060 Ti、feature level 6.4で検証した。
利用不可を理由にした実機テストのスキップを禁止するには`GOGRAD_REQUIRE_DIRECTML=1`を設定する。
共通のM5一致テストはerf式GELUまで含むため、feature level 5.1未満ならそのテスト群をスキップする。
`cmd/directml-infer`の`[74,768] @ [768,768]`と要素演算のCPUとの差は0だった。
DirectMLとautogradのDirectML経路は`-gcflags=all=-d=checkptr=2`でも検証した。
`cmd/directml-infer -encoder`の`[2,17,16]`、4 heads、FFN幅48は最大絶対誤差3.58e-7、コサイン1.0だった。
macOS/arm64とLinux/arm64では、cgoなしのスタブ経路のテストバイナリをクロスコンパイルした。
Linux/amd64はWSL2 Debian、Go 1.25.0、glibc 2.41で`CGO_ENABLED=0`の全テストと`go vet ./...`が成功した。
Linuxの関数呼出しは10引数・64 bit値・スタックalignment・並行呼出し・GCを専用のnative ABIテストで検証した。
checkptr検査、CコンパイラのあるWSL上での選択したテストの`-race`検査も成功した。
WSLで列挙できたのはllvmpipe（CPU型、Vulkan 1.4.305）だけで、GPUとして選ばず`Available() == false`を返す。
したがってLinuxのGPU一致テストはスキップしており、Linux上でGPU計算が成功したという検証ではない。
Windows/386ではDirectML単独のスタブとドライバ不要テストも実行した。
Windows/386の`autograd`は、既存のCUDAパッケージの2 GiB定数がintでoverflowするためビルドできない。
同じCUDA依存を含むVulkanテストもWindows/386ではビルドできないが、`go build ./gpu ./vulkan`のスタブはビルドできる。
Vulkanのnativeローダーの対象はWindows/amd64とLinux/amd64。Linux/arm64などは引き続きスタブ。

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
DirectMLのM4では、0要素・部分サイズの要素演算、GEMMの転置4通りとK=0、8 goroutineでの並行実行、
バッファプールと演算子キャッシュの上限・明示的な解放を確認する。CPU一致は`atol=1e-5`、`rtol=1e-5`。
autogradではGEMMからAdd / MulScalar / Reshapeをつなぎ、親を早期Closeしてから読み戻せることを確認する。
所有権、二重解放、ドライバ失敗後の回収、未対応環境のスタブはドライバなしでも検証する。
M5の一致テストはDirectMLでも実行する。3072列までのSoftmax / LayerNorm、GroupNorm、Conv1dの各オプション、
rank 6のバッチbroadcastとviewの合成、非連続な部分書き込みを確認する。
M3と同じConv1d＋自己注意＋FFNのコサインはDirectMLでも1.0で、既存Encoder層の反復実行後に使用中メモリが元へ戻ることも確認した。
ドライバなしのテストでは、view実体化・近似GELU・正規化の途中失敗で一時バッファが残らないこと、
erf式GELUのfeature制約がGPU dispatch前に判定され、バックエンドを利用不可にしないことを確認する。
M6のテストでは、aligned offset付きのGEMM / Add / Mulが入力を実体化しないこと、
非対応レイアウトがGPU上の実体化へ戻ること、Gatherが負のゼロとNaN payloadを保つことも確認する。
推論ベンチマークにはCPUだけで動くコマンド・集計・一致判定のテストを用意した。
DirectMLのバッチ化は、cold時の初期化とwarm時のsubmit回数、descriptorスロットの再利用、
満杯のcacheからの退避、cache上限を超えるバッチを実機で検証する。
ドライバ不要テストでは複合演算の各段に失敗を注入し、エラー伝搬とscratch回収を確認する。
Vulkanのbias融合は単一dispatch・scratchなしの呼出し、0要素、部分workgroup、2次元dispatch、入力のalias、
出力のalias拒否、加算順序、CPU一致を検証する。autogradでも能力なし・`ErrUnsupported`時のGPU合成、
転置・offset・Expand・rank 6、実行エラー時の出力と連続化scratchの回収を確認する。

## 残る作業

1. Linux/amd64の実GPUで、転送・全演算・Encoder一致と性能を検証する。ローダーと呼出規約のテストは実装済み。
2. M6の残り：forward全体のcommand記録のまとめ、融合演算・cacheの追加調整とAMD / Intelでの実測。
   ベンチマーク、DirectMLの主要なviewコピー削減、近似GELUと正規化内部のバッチ化は実装済み。
   Vulkanのerf式BiasGELU / BiasResidualの融合も実装済み。
   READMEは既存の未コミット変更を維持している。
