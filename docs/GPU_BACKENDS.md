# GPUバックエンド実装の進捗

VulkanとDirectMLは`autograd`から使うFP32推論バックエンド。学習はCPU/CUDAを使う。旧`tensor.Tensor` / `nn`はCPU/CUDA専用。

| バックエンド | 対象環境 | 検証状況 |
| --- | --- | --- |
| Vulkan | Windows/amd64、Vulkan 1.1以上の計算GPU | RTX 3060 Tiで演算・Encoder・性能を検証 |
| Vulkan | Linux/amd64 | ローダー・ABIを検証。実GPU計算は未検証 |
| DirectML | Windows/amd64、DirectML / D3D12対応の計算GPU | RTX 3060 Ti、feature level 6.4で検証 |

対象外のOS・アーキテクチャはスタブで`Available() == false`。[Linuxの依存と検証範囲](GPU_LINUX.md)、[性能測定](GPU_BACKEND_PERFORMANCE.md)を参照。

## 実装した範囲

| 演算・機能 | 対応範囲 |
| --- | --- |
| 基本演算 | Add、Mul、MulScalar、GPU間コピー |
| MatMul | 転置4通り、rank 6までのバッチ軸のブロードキャスト |
| view | Reshape、Permute、Transpose、Slice、Expand、Gather / Scatter |
| Transformer向け演算 | tanh近似 / erf式GELU、任意軸Softmax、LayerNorm、GroupNorm、Concat |
| Conv1d | `[batch,time,in_channels]`、重み`[out_channels,in_channels,kernel]`、任意のbias、stride / padding / dilation |
| Attention | GEMM・スケール・加算mask・Softmax・GEMMの合成 |
| モデル | `LinearLayer`、評価モードの`TransformerEncoderLayer` |

`gpu.Backend`はFP32バッファの割当・転送・演算を定義する。バッファは割当元を記録し、別バックエンドでの利用と解放後の使用を拒否する。登録済みバックエンドの置き換えは禁止。

Vulkanは埋め込みSPIR-V、device-localバッファ、host-visible/coherentなステージングを使う。GEMMは8×16の出力タイル、K方向16、128 invocations、共有メモリ1536 Bで計算し、転置4通りのパイプラインを保持する。viewの入力はstrideとoffsetで直接読む。Conv1dはGPU上のim2colとGEMMに展開する。`gpu.ViewBinaryBackend`のAdd / Mulは2つのviewのstrideとoffsetから添字を計算し、1 dispatchで評価する。

DirectMLは純GoのCOM呼出しでD3D12 DEFAULT / UPLOAD / READBACKリソースを使う。Conv1dのtime-majorデータはstride付きNCHW記述子で表す。LayerNorm / GroupNormは正規化とチャネルごとのaffine演算で構成する。

DirectMLのrank 4以下・16バイト境界に揃うGEMM入力とAdd / Mulは転置やゼロstrideを記述子で直接読む。追加能力の`gpu.ViewMatMuler` / `gpu.ViewBinaryBackend`がこの経路を表す。対応する非連続GatherはScaleBiasなしのIdentityを使い、それ以外はGPUコピーで実体化する。rank 6まで対応し、16バイト境界に揃わないoffsetも扱える。Scatterは指定要素だけを書き、viewの隙間の値を変更しない。

Vulkanの`gpu.FusedBiasBackend`は1次元biasの`BiasGELU`と、同shapeのresidualを足す`BiasResidual`を1 dispatchへ融合する。DirectMLの近似GELUは9 dispatch、LayerNorm / GroupNormは3 dispatchを1 command listへまとめる。各submitでフェンスを待ち、完了前にバッファやdescriptorを再利用しない。`gpu.Synchronizer`で読み戻さず完了を確認できる。

## 現時点の仕様と制約

- ドライバは初回利用時に読み込む。初期化の失敗は保持するため、導入後はプロセスの再起動が必要。WindowsはシステムディレクトリのDLL、Linuxは`libvulkan.so.1`を使う。
- `Available()`がfalseなら`Err()`で理由を確認できる。テンソル生成は`gpu.ErrUnavailable`を返し、`gpu.Get`は別デバイスへ切り替えない。転送やsubmitが失敗したバックエンドも以後は利用不可となる。`vulkan.DeviceInfo()` / `directml.DeviceInfo()`で選択したGPUとバージョンを取得できる。
- 記録が有効な`RequiresGrad=true`の入力は出力を割り当てる前に拒否する。`ExecutionContext.NoGrad`内でforwardを行う。backward、optimizer、勾配クリップ、BatchNormの統計更新、学習時のDropoutはNoGrad中でも拒否し、`gpu.ErrInferenceOnly`を返すかpanicする。
- LogSoftmaxとFlashAttentionは`gpu.ErrUnsupported`。`AttentionAlgorithm="flash"`の明示指定も拒否する。
- `Backend.GELU`はtanh近似、`gpu.ExactGELUBackend`はerf式。`GELU(x, false)`をtanh式へ置き換えない。DirectMLのerf式とEncoderにはfeature level 5.1以上が必要。Vulkanのerfは誤差上限1.5e-7の多項式近似。
- 共通APIのサイズはバイト数ではなくfloat32の要素数。形はバッファ全体と一致させ、出力は入力と別に割り当てる。`Download`は先頭の一部も読み戻せる。Conv1dのstride / dilationは正の値を明示する。
- stride / offsetの単位も要素数。負のstrideは未対応。非連続入力は必要に応じてGPU上で実体化し、CPUへ戻さない。`CopyFrom`はScatterで別名のversionも更新し、Expandなど書込先が重複するviewは更新前に拒否する。
- Vulkan / DirectMLはバックエンド内のmutexで同期し、OSスレッド固定は不要。同じTensorやモデルに対する並行の変更は呼出し側で同期する。VulkanはCPU型デバイス、DirectMLはWARPを選ばない。
- CUDAは`NewCUDAContext` / `NewCUDAContextOn`でOSスレッドを固定する。推論アダプタのバッファは同じdevice / streamで確保し、streamは全バッファの解放まで保持する。このアダプタはFP32で、既存のBF16設定を参照しない。

追加能力が未実装か`ErrUnsupported`を返す場合は既存のGPU演算へ合成する。実行エラーは伝搬し、CPUでやり直すことはない。融合した残差加算も`(x + bias) + residual`の順序を保つ。

Vulkan / DirectMLのバッファプールは既定で256 MiB。`MemoryStats`で使用量を確認し、`SetPoolCacheLimit`で上限変更、`ReleasePool`でキャッシュを解放できる。DirectMLはリソースのWidthではなく`GetResourceAllocationInfo`の確保量で数える。Vulkanの`MemoryStats`は、完了を確認できなかった割当を`QuarantinedBytes` / `QuarantinedBuffers`として報告する。

DirectMLの演算子キャッシュは最大64件、temporary / persistentリソース合計256 MiB。形・転置・スカラー値で再利用し、`ReleaseOperators`で解放する。バッチ内のdescriptorスロットも上限に数え、準備中の演算子は退避しない。キャッシュしない演算子もフェンス完了まで保持する。

`directml.MemoryStats`は演算子内部のヒープを含まない。`CachedOperatorStats`はキャッシュ済みの演算子だけを数え、いずれもプロセス全体のVRAM量ではない。

`NewBackend()`で作った独立バックエンドは全テンソルを解放してから`Close()`する。`autograd`が使う登録済みの既定バックエンドはプロセス寿命で、`Close()`は使えない。バッファの重複解放は安全。submit失敗後に完了を確認できないVulkanバックエンドは割当を隔離し、`Close()`がエラーを返す。デバイスロストまたは待機成功を確認できるまで、そのバックエンドを再利用しない。

## 使用例

```go
execution := autograd.NewExecutionContext()
x, err := execution.New([]float32{1, 2, 3}, []int{3}, tensor.Vulkan, true)
if err != nil { panic(err) }
defer x.Close()
var y *autograd.Tensor
execution.NoGrad(func() {
    y = autograd.MulScalar(autograd.Add(x, x), 0.5)
})
defer y.ReleaseGraph()
values, err := y.ToHost()
```

Encoderは`layer.Training = false`と`layer.Attention.Training = false`を設定し、`execution.BindModule(&layer.Module)`の後、`execution.NoGrad`内でForwardする。

```sh
go run ./cmd/vulkan-infer -n 100003
go run ./cmd/vulkan-infer -linear
go run ./cmd/directml-infer
go run ./cmd/directml-infer -encoder
go run ./cmd/gpu-infer-bench -warmup 2 -iters 7 -json
```

両推論例の`-cpu-fallback`はGPUを利用できない場合に構築前にCPUを選ぶ。GPUで実行中に失敗した処理の再試行には使わない。

## シェーダの再生成

通常のビルドは同梱の`vulkan/shaders/{pointwise,view,binary,gemm,reduce,conv}.spv`を使う。隣の`.comp`を変更した場合だけ、`glslang`をPATHに置いて再生成する。現在はglslang 16.6.0、Vulkan 1.1を対象にしている。

```sh
go generate ./vulkan
```

## テスト

```sh
go test ./...
go vet ./...
```

GPUが利用できない環境では実機テストをスキップする。スキップを禁止するには`GOGRAD_REQUIRE_VULKAN=1` / `GOGRAD_REQUIRE_DIRECTML=1`を設定する。DirectMLのerf式GELUを含む一致テストはfeature level 5.1未満でスキップする。

Windows/amd64、`CGO_ENABLED=0`、RTX 3060 Tiで全テストとcheckptr検査が成功した。Vulkan APIは1.4.341、compute queueは2、DirectML feature levelは6.4。Conv1d・自己注意・FFNの構成とEncoderをCPUと照合し、コサイン0.999以上と、反復後の使用中GPUメモリの回復を確認している。

推論例のCPUとの最大絶対誤差は次のとおり。

| 例 | 入力 | 最大絶対誤差 |
| --- | --- | ---: |
| `vulkan-infer -n 100003` | 100003要素 | 0 |
| `vulkan-infer -linear` | `[2,768,74] @ [74,768] + bias` | 0 |
| `directml-infer` | `[74,768] @ [768,768]`と要素演算 | 0 |
| `directml-infer -encoder` | `[2,17,16]`、4 heads、FFN幅48 | 3.58e-7、コサイン1.0 |

Linuxはローダー・ABI・失敗経路のテスト、checkptr、選択したテストのrace検査が成功したが、実GPU計算は未検証。[検証環境とコマンド](GPU_LINUX.md#検証した範囲)を参照。

macOS/arm64とLinux/arm64はスタブのクロスコンパイルを確認した。Windows/386は既存のCUDAの2 GiB定数がintでoverflowするため、autogradとVulkanのテストをビルドできない。`go build ./gpu ./vulkan`とDirectML単独のスタブテストは成功した。

## 残る作業

1. Linux実GPUでの転送・演算・Encoder一致・性能検証。
2. AMD / Intel GPUでの実機検証。forward全体のcommand記録のまとめや追加の融合・cache調整は今後の最適化対象。
