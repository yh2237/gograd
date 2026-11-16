# gograd

Go製のfloat32テンソル・自動微分ライブラリです。CPU / CUDAでの学習・推論と、Vulkan / DirectMLでの推論に対応しています。cgo・Cコンパイラ・Pythonは不要です。

## 動作環境

Go 1.25.0 以上が必要です。

| バックエンド | 環境 | 用途と検証状況 |
| --- | --- | --- |
| CPU | CPU | 学習・推論 |
| CUDA | Windows、NVIDIA GPU、CUDAランタイム / cuBLAS / NVRTC | 学習・推論 |
| Vulkan | Windows/amd64、Vulkan 1.1以上 | FP32推論（RTX 3060 Tiで検証） |
| Vulkan | Linux/amd64 | FP32推論（ローダー・ABI検証済み、実GPU計算は未検証） |
| DirectML | Windows/amd64、DirectML / D3D12対応GPU | FP32推論（RTX 3060 Tiで検証） |

- Vulkan / DirectMLは `autograd` 経由で利用します。DirectMLのerf版GELUおよびEncoderにはFeature Level 5.1以上が必要です。AMD / Intel GPUでの実機検証は未実施です。
- LinuxのVulkan呼び出しには純Goの `purego v0.11.2` を使用しているためcgoは不要ですが、実行時に `libc` / `libdl`、`libvulkan.so.1`、対応ICDが必要になり、完全静的リンクにはなりません。詳細は [Linuxの条件と検証範囲](docs/GPU_LINUX.md) を参照してください。
- WindowsでCUDAを使う場合は `bin\x64` を `PATH` に追加してください。

```powershell
$env:PATH = "$env:CUDA_PATH\bin\x64;$env:PATH"
```

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

入力やパラメータは `Close`、演算の中間グラフは `ReleaseGraph` で解放します。CUDA利用時は処理の間 `autograd.NewCUDAContext` を保持してOSスレッドを固定してください。Vulkan / DirectMLで `RequiresGrad=true` の入力をforwardする場合は、対象コンテキストの `NoGrad` ブロック内で実行します。

## 学習・推論の実行例

```sh
go run ./cmd/cnn-train -device cpu -steps 60 -out out/cnn.safetensors
go run ./cmd/mnist-train -download -device cpu -steps 100 -train-limit 2000 -valid-limit 1000
go run ./cmd/tabular-train -generate -device cpu -steps 400 -out out/tabular.safetensors
go run ./cmd/audio-train -generate -device cpu -steps 300 -out out/denoise.safetensors
go run ./cmd/vulkan-infer -linear
go run ./cmd/directml-infer -encoder
```

MNISTはローカルのraw / gzip形式のIDXファイルも利用できます。学習の再開やoptimizer、prefetch、勾配蓄積の指定方法は [学習API](docs/API.md) を参照してください。

GPUが利用できない場合、ライブラリは自動でCPUへフォールバックしません。推論例の `-cpu-fallback` は、テンソルやモデルの構築前にCPUを選択するためのフラグです。

## パッケージとドキュメント

| パッケージ | 用途 |
| --- | --- |
| `autograd` | define-by-runの自動微分、層、Transformer、損失、AdamW / SGD、学習checkpoint |
| `gpu`、`vulkan`、`directml` | FP32推論バックエンド、バッファ、view |
| `cuda`、`kernels` | CUDAバインディング、stream / event / graph、デバイスカーネル |
| `tensor` | CPU / CUDAテンソルとSGEMM |
| `data`、`metrics` | Dataset / Loader、CSV / IDX / WAV、augmentation、評価指標 |
| `torchpth` | PyTorch zip形式のtensor / state_dict読み込み |
| `dsp`、`torchrng` | FFT / STFT / リサンプリング、PyTorch CPU乱数の再現 |

- [学習API](docs/API.md)：実行コンテキスト、寿命管理、CNN、optimizer、データ処理、学習の再開
- [設計](docs/DESIGN.md)：グラフ、ストレージ、CPU / CUDA実装、学習の計測値
- [GPUバックエンド](docs/GPU_BACKENDS.md)：Vulkan / DirectMLの対応演算、制約、使用例
- [GPU性能測定](docs/GPU_BACKEND_PERFORMANCE.md)：測定条件、実測値、最適化前後の比較
- [Linux版Vulkan](docs/GPU_LINUX.md)：依存ライブラリ、ローダー、検証範囲

## テストとベンチマーク

```sh
go test ./...
go test -short ./...
go vet ./...
go run ./cmd/gpu-infer-bench -warmup 2 -iters 7 -json
go run ./cmd/transformer-bench -device cuda -warmup 1
go run ./cmd/transformer-bench -device cuda -warmup 2 -graph -bf16
go run ./cmd/attention-bench -seq 4096 -algorithm materialized
go run ./cmd/attention-bench -seq 4096 -algorithm flash
```

- `-short` を付けると、実行時間の大半を占める系列長4096のFlashAttention照合テストをスキップします。
- 演算と勾配はPyTorchのfixtureや有限差分で検証しています（PyTorchはfixtureの再生成用であり、通常の利用には不要です）。
- GPUがない環境では実機テストを自動でスキップします。スキップを禁止する環境変数は [GPUバックエンド](docs/GPU_BACKENDS.md#テスト) を参照してください。

## 制限

- 公開APIおよびCUDA graph / BF16などの実装は実験的な位置づけです。
- 値・勾配・master weightはFP32で保持し、CUDAのBF16オプションはGEMMのオペランドとシャドウにのみ使用します（FP16とloss scalingは未対応）。
- CUDAのtiled attentionはhead次元128まで対応していますが、計測した系列長では実体化（materialized）経路より低速です。
- Vulkan / DirectMLは学習、LogSoftmax、FlashAttentionに未対応です。
- パッケージ直下の互換API（`NoGrad` やBF16・アテンション設定）はグローバルな共有状態を持ちます。並行実行時は独立した `ExecutionContext` とモデルstateを使用し、共有テンソルの変更は呼び出し側で同期してください。

## ライセンス

MIT License ([LICENSE](LICENSE))