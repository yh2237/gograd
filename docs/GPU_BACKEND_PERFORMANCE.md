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
| gemm | 2.941 / 4.205 | 0.820 / 0.922 | 3.59x | 0.267 / 0.585 | 11.00x |
| projection | 0.614 / 2.185 | 0.178 / 0.273 | 3.44x | 0.152 / 0.286 | 4.04x |
| ffn | 5.697 / 6.768 | 1.147 / 1.570 | 4.97x | 0.537 / 1.484 | 10.60x |
| encoder | 120.210 / 153.970 | 17.244 / 20.777 | 6.97x | 6.023 / 6.375 | 19.96x |

EncoderのCPU対比最大絶対誤差はVulkan `5.72e-6`、DirectML `8.11e-6`（コサイン類似度 1.0）。

## 測定の前提

- 常駐データのsteady stateに対する測定であり、ホスト転送込みの速度や他のGPU・形状での性能は保証しない。
- Linux実GPU計算およびAMD / Intel GPUは未検証。
