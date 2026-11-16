# GPU推論の性能

## 対象と測定方法

`cmd/gpu-infer-bench` を用い、CPU・Vulkan・DirectMLのFP32 / NoGrad推論性能を比較する。入力と重みはデバイス上に常駐させ、viewの実体化・メモリ割当・グラフ管理を含む `autograd` API全体の実行時間を計測する（カーネル単体の時間ではない）。

```sh
go run ./cmd/gpu-infer-bench -warmup 2 -iters 7 -json
go run ./cmd/gpu-infer-bench -size 128 -hidden 512 -heads 4 -layers 1 -seq 74
```

- 各サンプルはforward開始からGPU完了同期（`gpu.Synchronizer`）および `ReleaseGraph` 完了までの時間とする（初期化・重み転送・初回実行・ホスト読み戻し・一致検証はmedianに含めない）。
- 初回出力をCPUと照合し、warmup後に `runtime.GC` を1回実行してから計測する。全反復終了後も `atol=1e-4`、`rtol=1e-4`、コサイン類似度0.999以上、およびメモリリークがないことを確認する。
- 計時はWindowsでは `QueryPerformanceCounter`、その他ではGoのmonotonic clockを使用する。
- JSON出力には全サンプル、median、nearest-rankのp95が含まれる。初期化時間等は別項目（`init_ms`、`setup_ms`、`first_forward_ms`）として記録する。
- 利用不可なデバイスは理由を表示してスキップし、CPUへ自動フォールバックしない（`-skip-unavailable=false` でスキップ禁止、`-devices` で対象指定可能）。測定は他プロセスと競合しない環境で順次実行した。

## 測定環境とワークロード

- **環境**：Windows/amd64、Go 1.27.0（`CGO_ENABLED=0`）、Intel Core i5-12400（12論理コア、`GOMAXPROCS=12`）、NVIDIA GeForce RTX 3060 Ti、Vulkan API 1.4.341、DirectML Feature Level 6.4
- **共通形状**：batch=1、sequence=74、model=768、FFN=3072、heads=12、Encoder=12層、Conv1d入力channels=74（初回forward後、warmup 2回・測定7回）

| workload | 内容 |
| --- | --- |
| gemm | `[768,768] @ [768,768]` |
| projection | `[74,768] @ [768,768]` |
| ffn | `[1,74,768]` → Linear(3072) → bias + erf式GELU → Linear(768) + bias |
| encoder | time-major Conv1d（kernel=3, stride=1, padding=1）→ 12層のpre-norm TransformerEncoderLayer |

## 最適化後のベンチマーク結果（viewコピー削減後）

単位はms。倍率は同一workloadのCPU中央値に対する高速化比を示す。

| workload | CPU median / p95 | Vulkan median / p95 | Vulkan倍率 | DirectML median / p95 | DirectML倍率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gemm | 2.563 / 2.672 | 0.789 / 0.903 | 3.25x | 0.245 / 0.278 | 10.46x |
| projection | 0.705 / 0.816 | 0.149 / 0.176 | 4.72x | 0.182 / 0.240 | 3.87x |
| ffn | 25.021 / 27.319 | 1.373 / 1.991 | 18.22x | 0.634 / 1.497 | 39.50x |
| encoder | 553.505 / 612.442 | 36.174 / 37.709 | 15.30x | 33.830 / 42.922 | 16.36x |

Encoderの最大絶対誤差はVulkan `5.72e-6`、DirectML `8.11e-6` で、コサイン類似度はいずれも0.999以上であった。

Encoderの生サンプル（ms）：

```text
CPU:      547.8621, 612.4417, 586.6704, 553.5045, 535.0929, 565.9672, 545.8879
Vulkan:    37.7085,  37.4588,  36.1738,  35.2626,  34.3905,  36.2057,  35.3225
DirectML:  33.0404,  32.9921,  32.2440,  34.4442,  33.8297,  42.9216,  41.3080
```

同一実行におけるセットアップ・初回forward・読み戻し時間（medianには含まない）：

| backend | パラメータ生成・構築・転送 | 初回forward | 初回出力の読み戻し |
| --- | ---: | ---: | ---: |
| CPU | 484.461 ms | 634.354 ms | 0.020 ms |
| Vulkan | 478.523 ms | 84.947 ms | 0.489 ms |
| DirectML | 501.270 ms | 138.646 ms | 0.291 ms |

## DirectMLのviewコピー削減前後

旧実装ではpacked QKVのK / Vにおけるオフセット付き転置重みを、各768×768行列につき1要素ずつコピーしていたため、12層で計14,155,776個のcopy regionが発生していた。

これを改善するため、rank 4以下かつ16バイト境界に揃うGEMMの直接バインド、`Add` / `Mul` のゼロstrideブロードキャスト、および `Identity` 演算子を用いた非連続Gather（`ScaleBias` を使わず負のゼロやNaN payloadを保持）を実装した。また、offsetをcache keyから除外して同形状のQ / K / Vで演算子を再利用するようにした。

別実行（`-workloads encoder -warmup 1 -iters 3`）による中央値比較：

| backend | 変更前 | 変更後 |
| --- | ---: | ---: |
| CPU | 578.200 ms | 506.109 ms |
| Vulkan | 39.667 ms | 35.939 ms |
| DirectML | 7,070.639 ms | 32.930 ms |

数秒を要していたコピー処理のボトルネックは解消された（※3回測定の参考値）。

## DirectML複合演算内のバッチ化

近似GELUの9 dispatch、およびLayerNorm / GroupNormの3 dispatchをそれぞれ1つのcommand listへ統合し、warmup後のsubmitとフェンス待ちを9→1回、3→1回へ削減した（演算間バリアは維持し、forward全体の非同期化は行っていない）。同一演算の反復時に未実行descriptorを上書きしないよう独立スロットを割り当て、実行完了まで演算子をpinする。

```sh
go run ./cmd/gpu-infer-bench -workloads gelu,layernorm,encoder -warmup 2 -iters 7 -json
```

別プロセスで7回ずつ測定した結果（`gelu` は `[1,74,3072]` のtanh近似、`layernorm` は `[1,74,768]` の最終軸）：

| DirectML workload | 変更前 median / p95 | 変更後 median / p95 | 前後の中央値比 |
| --- | ---: | ---: | ---: |
| gelu | 0.8369 / 1.3339 ms | 0.1851 / 0.9337 ms | 4.52x |
| layernorm | 0.3169 / 0.4068 ms | 0.2495 / 0.2770 ms | 1.27x |
| encoder | 34.7472 / 36.8871 ms | 35.2579 / 37.8502 ms | 0.99x |

```text
gelu before:      0.8590,  0.8369,  0.9156,  1.3339,  0.7268,  0.8280,  0.7786
gelu after:       0.3587,  0.9337,  0.2049,  0.1851,  0.1637,  0.1748,  0.1712
layernorm before: 0.4068,  0.3923,  0.3529,  0.3169,  0.2999,  0.2647,  0.2534
layernorm after:  0.2106,  0.2661,  0.2622,  0.2770,  0.2495,  0.2137,  0.2360
encoder before:  33.5330, 34.7472, 31.4275, 36.8871, 34.2618, 34.8019, 35.7983
encoder after:   34.6284, 36.9622, 35.2579, 35.2272, 35.2707, 33.6217, 37.8502
```

最大絶対誤差（GELU `4.77e-7`、LayerNorm `8.34e-7`、Encoder `8.11e-6`）とメモリ消費量に変化はなかった。Encoder全体で改善が見られなかったのは、Encoderが近似GELUではなくerf式GELUを使用しているためである（変更後のLayerNorm単体はCPUの0.1475 msより遅い）。

## Vulkanのbias融合演算

`gpu.FusedBiasBackend` により、1次元biasの `BiasGELU` と同shapeのresidualを加算する `BiasResidual` をそれぞれ1 dispatchへ融合し、連続入力時のdispatch数と `Alloc` 要求を3→1回に削減した。GELUはerf近似、残差加算は `(x + bias) + residual` の評価順を維持する。

```sh
go run ./cmd/gpu-infer-bench -workloads bias_gelu,bias_residual,ffn,encoder -warmup 2 -iters 7 -json
```

別プロセスで7回ずつ測定した結果（`bias_gelu` は `[1,74,3072]`、`bias_residual` は `[1,74,768]`）：

| Vulkan workload | 変更前 median / p95 | 変更後 median / p95 | 前後の中央値比 |
| --- | ---: | ---: | ---: |
| bias_gelu | 0.1850 / 0.2801 ms | 0.0771 / 0.1727 ms | 2.40x |
| bias_residual | 0.2475 / 0.3471 ms | 0.2473 / 0.3459 ms | 1.00x |
| ffn | 1.3367 / 1.6453 ms | 1.4333 / 2.3510 ms | 0.93x |
| encoder | 39.8714 / 43.8014 ms | 36.1015 / 37.5938 ms | 1.10x |

```text
bias_gelu before:      0.2801,  0.1700,  0.1850,  0.1660,  0.1692,  0.2332,  0.2670
bias_gelu after:       0.1727,  0.0906,  0.0682,  0.0609,  0.0634,  0.0771,  0.1290
bias_residual before:  0.3471,  0.2061,  0.2475,  0.2288,  0.2179,  0.2506,  0.2825
bias_residual after:   0.2639,  0.2473,  0.3459,  0.2578,  0.0600,  0.0547,  0.0573
ffn before:           1.4126,  1.3547,  1.6453,  1.2359,  1.2323,  1.3367,  1.2515
ffn after:            1.3905,  1.4333,  2.3510,  1.3340,  1.6692,  1.8738,  1.3195
encoder before:      40.2656, 43.8014, 41.6598, 37.9244, 38.6950, 39.4888, 39.8714
encoder after:       37.1714, 37.5938, 36.8934, 35.3774, 34.4226, 33.8998, 36.1015
```

最大絶対誤差（BiasGELU `1.79e-7`、BiasResidual `0`、FFN `2.68e-7`、Encoder `5.72e-6`）は変わらず、Encoderのgraph解放後のpool cacheは 138,072,064 → 122,343,424 bytes に減少した。別プロセス実行による環境変動も含まれるため、単純な前後差は参考値とする。

## 残る制約

- 本測定は常駐データのsteady stateに対するものであり、初期化・データ転送込みの速度や他のGPU・形状での性能は保証しない。
- 現在は各submitでフェンスを待機しており、forward全体のコマンド一括記録やさらなるカーネル融合に改善の余地がある。
- メモリ使用量はgograd管理分のバッファのみを対象としており、プロセス全体のVRAM消費量ではない。
- Linuxの実GPU計算およびAMD / Intel GPUは未検証である（詳細は [Linux版の検証範囲](GPU_LINUX.md#検証した範囲) を参照）。