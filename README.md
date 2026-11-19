# gograd

Go製のfloat32テンソル・自動微分ライブラリです。CPU / CUDAでの学習・推論と、Vulkan / DirectMLでの推論に対応しています（cgo・Cコンパイラ・Python不要）。

## 動作環境

Go 1.25.0 以上が必要です。

| バックエンド | 環境 | 用途と検証状況 |
| --- | --- | --- |
| CPU | CPU | 学習・推論 |
| CUDA | Windows、NVIDIA GPU、CUDAランタイム / cuBLAS / NVRTC | 学習・推論 |
| Vulkan | Windows/amd64、Vulkan 1.1以上 | FP32推論（RTX 3060 Tiで検証） |
| Vulkan | Linux/amd64 | FP32推論（ローダー・ABI検証済み、実GPU計算は未検証） |
| DirectML | Windows/amd64、DirectML / D3D12対応GPU | FP32推論（RTX 3060 Ti / WARPで検証） |

- Vulkan / DirectMLは `autograd` 経由で利用します。AMD / Intel GPUでの実機検証は未実施です。
- Linux版Vulkanの実行要件は [Linux版Vulkan](docs/GPU_LINUX.md) を参照してください。
- WindowsでCUDAを使う場合は `$env:CUDA_PATH\bin\x64` を `PATH` に追加してください。

## 自動微分の例

```go
package main

import (
    "fmt"

    "github.com/yh2237/gograd/autograd"
    "github.com/yh2237/gograd/tensor"
)

func main() {
    execution := autograd.NewExecutionContext()
    x, err := execution.New([]float32{2, 3}, []int{2}, tensor.CPU, true)
    if err != nil { panic(err) }
    defer x.Close()

    loss := autograd.Sum(autograd.Mul(x, x), 0)
    defer loss.ReleaseGraph()
    if err := loss.Backward(); err != nil { panic(err) }
    grad, err := x.GradToHost()
    if err != nil { panic(err) }
    fmt.Println(grad) // [4 6]
}
```

入力・パラメータは `Close`、中間グラフは `ReleaseGraph` で解放します。CUDA利用時は `autograd.NewCUDAContext` でOSスレッドを固定し、Vulkan / DirectMLで `RequiresGrad=true` の入力をforwardする場合は対象コンテキストの `NoGrad` 内で実行します。GPUが利用できない場合、自動的なCPUフォールバックは行いません。

## パッケージとドキュメント

| パッケージ | 用途 |
| --- | --- |
| `autograd` | define-by-run自動微分、層、Transformer、CNN、損失、AdamW / SGD、checkpoint |
| `gpu`、`vulkan`、`directml` | FP32推論バックエンド、バッファ、view |
| `cuda`、`kernels` | CUDAバインディング、stream / event / graph、デバイスカーネル |
| `tensor` | CPU / CUDAテンソル、SGEMM |
| `data`、`metrics` | Dataset / Loader、CSV / IDX / WAV、augmentation、評価指標 |
| `torchpth`、`dsp`、`torchrng` | PyTorch checkpoint読み込み、FFT / STFT / リサンプリング、PyTorch CPU乱数再現 |

- [APIリファレンス](docs/API.md)
- [設計と計測値](docs/DESIGN.md)
- [GPUバックエンド](docs/GPU_BACKENDS.md)
- [GPU推論の性能](docs/GPU_BACKEND_PERFORMANCE.md)
- [Linux版Vulkan](docs/GPU_LINUX.md)

## テストとベンチマーク

```sh
go test ./...
go test -short ./...
go vet ./...
go run ./cmd/gpu-infer-bench -warmup 2 -iters 7
go run ./cmd/transformer-bench -device cuda -warmup 2 -bf16
go run ./cmd/attention-bench -seq 4096 -algorithm flash
```

`-short` は系列長4096のFlashAttention照合テストをスキップします。GPU非搭載環境では実機テストを自動スキップします（禁止フラグは [GPUバックエンド](docs/GPU_BACKENDS.md#テスト) を参照）。

## 制限

- 公開API・CUDA graph・BF16は実験的な位置づけです。
- 値・勾配・master weightはFP32保持です。CUDAのBF16オプションはGEMMオペランドとシャドウにのみ使用します（FP16とloss scalingは未対応）。
- CUDAのtiled flash attentionはFP32専用、head次元128まで対応します。
- Vulkan / DirectMLは推論専用で、学習・backward・FlashAttentionは未対応です。演算別の対応範囲は [GPUバックエンド](docs/GPU_BACKENDS.md#対応演算) を参照してください。
- パッケージ直下の互換API（`NoGrad`、BF16・アテンション設定）はグローバル状態を持ちます。並行実行時は独立した `ExecutionContext` とモデルstateを使用してください。

## ライセンス

MIT License ([LICENSE](LICENSE))
