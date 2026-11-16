# GPUバックエンド

VulkanとDirectMLは `autograd` 経由で利用するFP32推論専用バックエンドである（学習はCPU / CUDAを使用）。

| バックエンド | 対象環境 | 検証状況 |
| --- | --- | --- |
| Vulkan | Windows/amd64、Vulkan 1.1以上の計算GPU | RTX 3060 Tiで演算・Encoder・性能を検証 |
| Vulkan | Linux/amd64 | ローダー・ABIを検証（実GPU計算は未検証） |
| DirectML | Windows/amd64、DirectML / D3D12対応の計算GPU | RTX 3060 Ti（Feature Level 6.4）で検証 |

対象外のOS・アーキテクチャではスタブが組み込まれ、`Available() == false` を返す。AMD / Intel GPUでの実機検証は未実施。

## 実装した範囲

| 演算・機能 | 対応範囲 |
| --- | --- |
| 基本演算 | `Add`、`Mul`、`MulScalar`、GPU間コピー（`Sub`、`Div`、`Max`、`Min` は未対応で `ErrUnsupported`） |
| MatMul | 転置4通り、rank 6までのバッチ軸ブロードキャスト |
| view | `Reshape`、`Permute`、`Transpose`、`Slice`、`Expand`、`Gather` / `Scatter` |
| Transformer向け演算 | tanh近似 / erf式GELU、任意軸Softmax、LayerNorm、GroupNorm、Concat |
| Conv1d | `[batch,time,in_channels]`、重み `[out_channels,in_channels,kernel]`、任意bias、stride / padding / dilation |
| Attention | GEMM・スケール・加算mask・Softmax・GEMMの合成経路 |
| モデル | `LinearLayer`、評価モードの `TransformerEncoderLayer` |

`gpu.Backend` はFP32バッファの割当・転送・演算を定義する。バッファは割当元を記録し、異なるバックエンドでの利用や解放後のアクセスを拒否する（登録済みバックエンドの差し替えは不可）。

- **Vulkan**：埋め込みSPIR-V、device-localバッファ、host-visible/coherentステージングを使用する。GEMMは8×16出力タイル（K方向16、128 invocations、共有メモリ1536 B）で転置4通りのパイプラインを保持する。Conv1dはGPU上のim2colとGEMMで計算し、`gpu.ViewBinaryBackend` の `Add` / `Mul` および `gpu.FusedBiasBackend`（`BiasGELU` / `BiasResidual`）は1 dispatchで評価する。
- **DirectML**：純GoのCOM呼び出しでD3D12 `DEFAULT` / `UPLOAD` / `READBACK` リソースを使用する。rank 4以下かつ16バイト境界に揃うGEMM入力と `Add` / `Mul` は記述子で直接読み取り（`gpu.ViewMatMuler` / `gpu.ViewBinaryBackend`）、対応する非連続 `Gather` は `ScaleBias` なしの `Identity` 演算子を使用する（非対応配置はGPUコピーで実体化、rank 6および非16バイト境界offsetに対応）。近似GELUは9 dispatch、LayerNorm / GroupNormは3 dispatchを1 command listへまとめてsubmitする。

いずれのバックエンドも各submitでフェンスを待機し、完了前にバッファやdescriptorを再利用しない（`gpu.Synchronizer` で読み戻しなしに完了待機可能）。

## 現時点の仕様と制約

- **ドライバ読み込み**：初回利用時に動的ロードする（WindowsはシステムディレクトリのDLL、Linuxは `libvulkan.so.1`）。初期化失敗はキャッシュされるため、ドライバ導入後はプロセスの再起動が必要。
- **利用不可時の挙動**：`Available()` がfalseの場合は `Err()` で理由を取得でき、テンソル生成は `gpu.ErrUnavailable` を返す（別デバイスへの自動フォールバックはしない）。転送やsubmitに失敗したバックエンドも以降は利用不可となる。`vulkan.DeviceInfo()` / `directml.DeviceInfo()` でデバイス名とバージョンを取得可能。
- **推論専用制約**：記録有効かつ `RequiresGrad=true` の入力は出力割当前に拒否されるため、`ExecutionContext.NoGrad` 内でforwardを実行する必要がある。backward、optimizer、勾配クリップ、BatchNorm統計更新、学習時のDropoutは `NoGrad` 中でも拒否される（`gpu.ErrInferenceOnly` またはpanic）。
- **未対応演算**：`LogSoftmax` と `FlashAttention`（`AttentionAlgorithm="flash"` の明示指定を含む）は `gpu.ErrUnsupported` を返す。
- **DirectML Conv1dの制限**：`dilation > 1` かつ出力長が1、かつ入力長がカーネル長と等しい場合、DirectMLが畳み込み結果ではなく0を出力する不具合を回避するため `gpu.ErrUnsupported` を返す。
- **GELU精度**：`Backend.GELU` はtanh近似、`gpu.ExactGELUBackend` はerf式を使用する。DirectMLのerf式とEncoderにはFeature Level 5.1以上が必要。Vulkanのerfは誤差上限 `1.5e-7` の多項式近似。
- **サイズ・stride単位**：共通APIのサイズ・stride・offsetはすべてバイト数ではなくfloat32の要素数単位で扱う（負のstrideは未対応）。非連続入力はGPU上で実体化し、`CopyFrom` で書込先が重複するview（`Expand` 等）は更新前に拒否する。
- **スレッド安全性**：Vulkan / DirectMLはバックエンド内部のmutexで同期するためOSスレッド固定は不要（CPU型デバイスやWARPは自動選択対象から除外）。ただし同一テンソルやモデルへの並行変更は呼び出し側で同期すること。
- **CUDA推論アダプタ**：`NewCUDAContext` / `NewCUDAContextOn` でOSスレッドを固定し、同一device / streamでバッファを確保する。このアダプタはFP32専用でBF16設定を参照しない。

追加能力が未実装または `ErrUnsupported` を返す場合は既存のGPU演算へフォールバックして合成する（実行時エラーはそのまま伝搬し、CPUでの再実行は行わない）。

### メモリ・演算子キャッシュと寿命管理

- **バッファプール**：既定上限は256 MiB。`MemoryStats` で使用量確認、`SetPoolCacheLimit` で上限変更、`ReleasePool` でキャッシュ解放が可能。DirectMLはリソース幅ではなく `GetResourceAllocationInfo` の実確保量でカウントする。Vulkanでは完了確認できなかった割当を `QuarantinedBytes` / `QuarantinedBuffers` として隔離報告する。
- **DirectML演算子キャッシュ**：最大64件・合計256 MiBまで保持し、形状・転置・スカラー値で再利用する（`ReleaseOperators` で解放）。内部ヒープは `directml.MemoryStats` に含まれず、プロセス全体のVRAM量とは異なる。
- **ライフサイクル**：`NewBackend()` で生成した独立バックエンドは全テンソル解放後に `Close()` する（`autograd` 登録済みの既定バックエンドはプロセス寿命のため `Close()` 不可）。submit失敗後に完了確認できないVulkanバックエンドは割当を隔離し、`Close()` でエラーを返す。

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

Encoderを推論する場合は `layer.Training = false` および `layer.Attention.Training = false` を設定し、`execution.BindModule(&layer.Module)` の後に `execution.NoGrad` 内で `Forward` を呼び出す。

```sh
go run ./cmd/vulkan-infer -n 100003
go run ./cmd/vulkan-infer -linear
go run ./cmd/directml-infer
go run ./cmd/directml-infer -encoder
go run ./cmd/gpu-infer-bench -warmup 2 -iters 7 -json
```

`-cpu-fallback` はGPU利用不可時に構築前からCPUを選ぶフラグであり、GPU実行中のエラー再試行には使用されない。

## シェーダの再生成

シェーダはコンパイル済みの `vulkan/shaders/{pointwise,view,binary,gemm,reduce,conv}.spv` を同梱している。`.comp` ファイルを変更した場合のみ、`glslang`（現在 glslang 16.6.0 / Vulkan 1.1 対象）を `PATH` に通して再生成する。

```sh
go generate ./vulkan
```

## テスト

```sh
go test ./...
go vet ./...
```

- GPUがない環境では実機テストをスキップする（`GOGRAD_REQUIRE_VULKAN=1` / `GOGRAD_REQUIRE_DIRECTML=1` でスキップ禁止）。DirectMLのerf式GELUテストはFeature Level 5.1未満でスキップされる。
- `GOGRAD_DIRECTML_SOFTWARE=1` を指定するとソフトウェアラスタライザ（WARP）を使用し、GPUなしのCI環境と同じ実装で検証できる（WARPで1要素グループの正規化がNaNを返す問題はbiasのみを返す経路で回避）。
- Windows/amd64（`CGO_ENABLED=0`、RTX 3060 Ti、Vulkan 1.4.341、DirectML FL 6.4）で全テストおよび `checkptr` 検査に合格し、コサイン類似度0.999以上とメモリリークなしを確認済み。

推論サンプルにおけるCPU出力との最大絶対誤差：

| 例 | 入力 | 最大絶対誤差 |
| --- | --- | ---: |
| `vulkan-infer -n 100003` | 100003要素 | 0 |
| `vulkan-infer -linear` | `[2,768,74] @ [74,768] + bias` | 0 |
| `directml-infer` | `[74,768] @ [768,768]` と要素演算 | 0 |
| `directml-infer -encoder` | `[2,17,16]`、4 heads、FFN幅48 | `3.58e-7`（コサイン 1.0） |

- Linuxはローダー・ABI・失敗経路・`checkptr`・`-race` 検査に合格済みだが実GPU計算は未検証（[Linux版の検証範囲](GPU_LINUX.md#検証した範囲) 参照）。
- macOS/arm64とLinux/arm64はスタブのクロスコンパイルを確認済み（Windows/386は既存CUDAの2 GiB定数が `int` でオーバーフローするため `autograd` / `vulkan` のテストビルド不可）。

## 残る作業

1. Linux実GPUでの転送・演算・Encoder一致・性能検証
2. AMD / Intel GPUでの実機検証、およびforward全体のコマンド記録統合やさらなる融合・キャッシュ最適化