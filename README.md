# gograd

Goで書いたfloat32のテンソル・自動微分ライブラリです。CPU/CUDAで学習と推論、Vulkan/DirectMLで推論を行います。通常の利用にcgo・Cコンパイラ・Pythonは不要です。

## 動作環境

Go 1.25.0以上を使います。

| バックエンド | 環境 | 用途と検証状況 |
| --- | --- | --- |
| CPU | CPU | 学習・推論 |
| CUDA | Windows、NVIDIA GPU、CUDAランタイム / cuBLAS / NVRTC | 学習・推論 |
| Vulkan | Windows/amd64、Vulkan 1.1以上 | FP32推論。RTX 3060 Tiで検証 |
| Vulkan | Linux/amd64 | FP32推論。ローダー・ABIを検証済み、実GPU計算は未検証 |
| DirectML | Windows/amd64、DirectML / D3D12対応GPU | FP32推論。RTX 3060 Tiで検証 |

Vulkan/DirectMLは`autograd`から使います。旧`tensor.Tensor` / `nn`はCPU/CUDA専用です。DirectMLのerf式GELUとEncoderにはfeature level 5.1以上が必要です。AMD / Intel GPUでの実機検証は未実施です。

LinuxのVulkan呼出しには純Goの`purego v0.11.2`を使い、実行時にlibc / libdl、`libvulkan.so.1`、対応ICDが必要です。cgoなしでも完全静的リンクにはなりません。[Linuxの条件と検証範囲](docs/GPU_LINUX.md)を参照してください。

WindowsのCUDAは`bin\x64`をPATHに追加します。

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

入力・パラメータは`Close`、演算の中間値は`ReleaseGraph`で解放します。CUDAでは処理の間`autograd.NewCUDAContext`を保持してOSスレッドを固定します。Vulkan/DirectMLで`RequiresGrad=true`の入力を使うforwardは、その実行コンテキストの`NoGrad`内で行います。

## 学習・推論の実行例

```sh
go run ./cmd/cnn-train -device cpu -steps 60 -out out/cnn.safetensors
go run ./cmd/mnist-train -download -device cpu -steps 100 -train-limit 2000 -valid-limit 1000
go run ./cmd/tabular-train -generate -device cpu -steps 400 -out out/tabular.safetensors
go run ./cmd/audio-train -generate -device cpu -steps 300 -out out/denoise.safetensors
go run ./cmd/vulkan-infer -linear
go run ./cmd/directml-infer -encoder
```

CNNの出力先ディレクトリは先に作成してください。MNISTはローカルのraw/gzip IDXも使えます。学習の再開、optimizer、prefetch、勾配蓄積の指定は[学習API](docs/API.md)にあります。

GPUが利用できない場合、ライブラリは自動でCPUへ切り替えません。推論例の`-cpu-fallback`はテンソルとモデルの構築前にCPUを選ぶためのフラグです。

## パッケージとドキュメント

| パッケージ | 用途 |
| --- | --- |
| `autograd` | define-by-runの自動微分、層、Transformer、損失、AdamW / SGD、学習checkpoint |
| `gpu`、`vulkan`、`directml` | FP32推論バックエンド、バッファとview |
| `cuda`、`kernels` | CUDAバインディング、stream / event / graph、デバイスカーネル |
| `tensor`、`nn` | 既存のCPU/CUDAテンソル・モジュール・明示的backward・JSON checkpoint |
| `data`、`metrics` | Dataset / Loader、CSV / IDX / WAV、augmentation、評価指標 |
| `torchpth` | PyTorch zip形式のtensor / state_dict読込み |
| `dsp`、`torchrng` | FFT / STFT / リサンプリング、PyTorch CPU乱数の再現 |

- [学習API](docs/API.md)：実行コンテキスト、寿命管理、CNN、optimizer、データ処理、再開
- [設計](docs/DESIGN.md)：グラフ、ストレージ、CPU/CUDA実装、学習の計測値
- [GPUバックエンド](docs/GPU_BACKENDS.md)：Vulkan/DirectMLの対応演算、制限、使用例
- [GPU性能測定](docs/GPU_BACKEND_PERFORMANCE.md)：条件、実測値、最適化前後の比較
- [Linux版Vulkan](docs/GPU_LINUX.md)：依存、ローダー、検証範囲

## テストとベンチマーク

```sh
go test ./...
go vet ./...
go run ./cmd/gpu-infer-bench -warmup 2 -iters 7 -json
go run ./cmd/transformer-bench -device cuda -warmup 1
go run ./cmd/transformer-bench -device cuda -warmup 2 -graph -bf16
go run ./cmd/attention-bench -seq 4096 -algorithm materialized
go run ./cmd/attention-bench -seq 4096 -algorithm flash
```

演算と勾配はPyTorchのfixtureや有限差分で照合します。PyTorchはfixtureの再生成・比較に使い、通常の利用には必要ありません。GPUを利用できない環境では実機テストをスキップします。Vulkan/DirectMLのスキップを禁止する環境変数は[GPUのテスト](docs/GPU_BACKENDS.md#テスト)を参照してください。

## 制限

公開APIとCUDA graph / BF16などの実装は実験的な位置づけです。基本の値・勾配・master weightはFP32で、任意のCUDA BF16はGEMMのオペランドとシャドウに使います。FP16とloss scalingは未対応です。

CUDAのtiled attentionはhead次元128までで、計測した系列長では実体化経路より遅くなります。Vulkan/DirectMLは学習、LogSoftmax、FlashAttentionに対応していません。

互換用のNoGrad・BF16・アテンション設定は共有状態です。並行実行には独立した`ExecutionContext`とモデルstateを使い、共有テンソルの変更は呼出し側で同期してください。

## ライセンス

MIT。詳細は[LICENSE](LICENSE)を参照してください。
