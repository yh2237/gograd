# GPU推論の性能

## 測定条件

`cmd/gpu-infer-bench` を用い、デバイス常駐のFP32 / `NoGrad` 推論（view実体化・割当・`gpu.Synchronizer` 完了待機・`ReleaseGraph` を含む）を計測した。初期化・重み転送・初回forward・ホスト読み戻しは中央値に含めない。全反復後にCPU出力と照合し（`atol=1e-4`、`rtol=1e-4`、コサイン類似度0.999以上）メモリリークがないことを確認している。

- **環境**：Windows/amd64、Go 1.27.0（`CGO_ENABLED=0`）、Intel Core i5-12400（12論理コア、`GOMAXPROCS=12`）、NVIDIA GeForce RTX 3060 Ti、Vulkan 1.4.341、DirectML Feature Level 6.4
- **コマンド**：`go run ./cmd/gpu-infer-bench -warmup 2 -iters 7`
- **ワークロード**（batch=1、seq=74、model=768、FFN=3072、heads=12、Encoder=12層）：
  - `gemm`：`[768,768] @ [768,768]`
  - `projection`：`[74,768] @ [768,768]`
  - `ffn`：`[1,74,768]` → Linear(3072) → bias + erf式GELU → Linear(768) + bias
  - `encoder`：time-major Conv1d（kernel=3, stride=1, padding=1）→ 12層 pre-norm `TransformerEncoderLayer`

## 実測値

単位はms。倍率は同一workloadのCPU中央値に対する比。

| workload | CPU median / p95 | Vulkan median / p95 | Vulkan倍率 | DirectML median / p95 | DirectML倍率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gemm | 1.836 / 2.309 | 0.273 / 0.310 | 6.72x | 0.223 / 0.293 | 8.24x |
| projection | 0.427 / 0.668 | 0.154 / 0.295 | 2.77x | 0.123 / 0.181 | 3.46x |
| ffn | 4.487 / 5.405 | 0.676 / 0.948 | 6.64x | 0.361 / 0.426 | 12.44x |
| encoder | 111.995 / 121.688 | 11.736 / 11.794 | 9.54x | 7.241 / 7.382 | 15.47x |

EncoderのCPU対比最大絶対誤差はVulkan `6.20e-6`、DirectML `8.11e-6`（コサイン類似度 1.0）。

## 測定の前提

- 常駐データのsteady stateに対する測定であり、ホスト転送込みの速度や他のGPU・形状での性能は保証しない。
- Linux実GPU計算およびAMD / Intel GPUは未検証。
