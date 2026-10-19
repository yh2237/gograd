# GPU推論の性能検証（M6）

## 対象と測定方法

`cmd/gpu-infer-bench`でCPU・Vulkan・DirectMLのFP32 / NoGrad推論を比較する。入力と重みはデバイスに常駐させ、同じ`autograd` APIと固定seedを使う。viewの実体化、割当、graph管理も含むAPI全体の時間で、カーネル単体ではない。

```sh
go run ./cmd/gpu-infer-bench -warmup 2 -iters 7 -json
go run ./cmd/gpu-infer-bench -size 128 -hidden 512 -heads 4 -layers 1 -seq 74
```

各サンプルはforward開始からGPU完了待ちとReleaseGraph終了まで。初期化・重み転送・初回実行・読み戻し・一致判定はmedianに含めない。backward、optimizer、BF16、CUDA graph captureも対象外。

初回出力をCPUと照合し、warmup後にruntime.GCを1回実行してから測定する。計測後も`atol=1e-4`、`rtol=1e-4`、コサイン0.999以上を検証し、反復後の使用中バッファ数・バイト数の増加を確認する。完了待ちは`gpu.Synchronizer`、時計はWindowsがQueryPerformanceCounter、その他はGoのmonotonic clock。

JSONには全サンプル、median、nearest-rankのp95を出す。7サンプルのp95は最大値。`setup_ms`は初期値生成・構築・転送、`first_forward_ms`は初回forward完了まででgraph解放前。ドライバ初期化は別の`init_ms`。

利用できないデバイスは理由を表示してスキップし、CPUに置き換えない。スキップ禁止は`-skip-unavailable=false`、対象指定は`-devices cpu,vulkan`。対応しない演算・feature levelはエラー。以下の測定はGPUテストや他のベンチマークと並行せず、1プロセス内で順次実行した。

## 測定環境と形

Windows/amd64、Go 1.27.0、CGO_ENABLED=0。CPUはIntel Core i5-12400、12 logical CPUs、GOMAXPROCS=12。GPUはRTX 3060 Ti、Vulkan API 1.4.341、DirectML feature level 6.4。

共通の形はbatch=1、sequence=74、model=768、FFN=3072、heads=12、Encoder=12層、Conv1d入力channels=74。初回forwardの後にwarmup 2回、測定7回。

| workload | 内容 |
| --- | --- |
| gemm | `[768,768] @ [768,768]` |
| projection | `[74,768] @ [768,768]` |
| ffn | `[1,74,768]` → Linear(3072) → bias + erf式GELU → Linear(768) + bias |
| encoder | time-major Conv1d（kernel=3、stride=1、padding=1）→ 12層のpre-norm TransformerEncoderLayer |

## viewコピー削減後の測定

単位はms。倍率は同じworkloadのCPU中央値に対する値。

| workload | CPU median / p95 | Vulkan median / p95 | Vulkan倍率 | DirectML median / p95 | DirectML倍率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gemm | 2.563 / 2.672 | 0.789 / 0.903 | 3.25x | 0.245 / 0.278 | 10.46x |
| projection | 0.705 / 0.816 | 0.149 / 0.176 | 4.72x | 0.182 / 0.240 | 3.87x |
| ffn | 25.021 / 27.319 | 1.373 / 1.991 | 18.22x | 0.634 / 1.497 | 39.50x |
| encoder | 553.505 / 612.442 | 36.174 / 37.709 | 15.30x | 33.830 / 42.922 | 16.36x |

Encoderの最大絶対誤差はVulkan `5.72e-6`、DirectML `8.11e-6`、コサインはいずれも0.999以上。この条件では両バックエンドがCPU比2倍という目安を超えた。

Encoderの生サンプルは次のとおり。

```text
CPU:      547.8621, 612.4417, 586.6704, 553.5045, 535.0929, 565.9672, 545.8879
Vulkan:    37.7085,  37.4588,  36.1738,  35.2626,  34.3905,  36.2057,  35.3225
DirectML:  33.0404,  32.9921,  32.2440,  34.4442,  33.8297,  42.9216,  41.3080
```

同じ実行の構築・初回forward・読み戻し時間。上のmedianには含めない。

| backend | パラメータ生成・構築・転送 | 初回forward | 初回出力の読み戻し |
| --- | ---: | ---: | ---: |
| CPU | 484.461 ms | 634.354 ms | 0.020 ms |
| Vulkan | 478.523 ms | 84.947 ms | 0.489 ms |
| DirectML | 501.270 ms | 138.646 ms | 0.291 ms |

## DirectMLの変更前後

旧経路はpacked QKVのK / Vのオフセット付き転置重みを、各768×768行列につき1要素ずつコピーしていた。12層では14,155,776個のcopy regionになる。

rank 4以下・16バイト境界のGEMMを直接バインドし、Add / Mulはゼロstrideでbroadcast、対応するGatherはIdentityで処理するよう変更した。offsetをcache keyから外し、同じ形のQ / K / Vで演算子を再利用する。非対応の配置はGPU上で実体化し、部分書込のScatterはbuffer copyを保つ。IdentityはScaleBiasを使わず、負のゼロとNaN payloadを保つ。

別実行の`-workloads encoder -warmup 1 -iters 3`による中央値。

| backend | 変更前 | 変更後 |
| --- | ---: | ---: |
| CPU | 578.200 ms | 506.109 ms |
| Vulkan | 39.667 ms | 35.939 ms |
| DirectML | 7,070.639 ms | 32.930 ms |

数秒規模のコピー記録は解消した。ただし3回測定のため、細かな差の評価には使わない。上の7回測定とは別の実行。

## DirectML複合演算内のバッチ化

近似GELUの9 dispatch、LayerNorm / GroupNormの3 dispatchを、それぞれ1 command listへまとめた。warmup後のsubmit / フェンス待ちは9→1回、3→1回。計算式と演算間バリアは変えず、forward全体の非同期化は行っていない。

cold時は全演算子を初期化してから実行用listを開く。同じ演算を繰り返す場合は独立したdescriptorスロットを使い、未実行のdescriptorを上書きしない。準備中の演算子はpinし、未キャッシュのものもフェンス完了まで保持する。近似GELUのwarm cacheは7演算子から9スロットとなり、通常のcache上限に数える。

```sh
go run ./cmd/gpu-infer-bench -workloads gelu,layernorm,encoder -warmup 2 -iters 7 -json
```

前後を別プロセスで7回ずつ測定した。geluは`[1,74,3072]`のtanh近似、layernormは`[1,74,768]`の最終軸。

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

最大絶対誤差はGELU `4.77e-7`、LayerNorm `8.34e-7`、Encoder `8.11e-6`で前後とも同じ。コサイン0.999以上、使用中バッファの増加はなかった。

Encoder全体の改善は確認できなかった。Encoderはerf式GELUを使うため近似GELUの変更は適用されず、正規化のsubmitだけが減る。同じ実行でCPUも532.0→552.7 ms、変更していないVulkanも35.9→40.6 msと変動した。変更後のLayerNorm単体はCPUの0.1475 msより遅い。

## Vulkanのbias融合演算

`gpu.FusedBiasBackend`でBiasGELU / BiasResidualを1 dispatchへ融合した。1次元biasと同shapeのresidualが対象。連続入力のdispatchとAlloc要求はそれぞれ3→1回となる。物理割当数はpool再利用による。

GELUは従来と同じerf近似、残差加算は`(x + bias) + residual`の順序を保つ。非連続入力は必要なものだけGPUで連続化し、対象外のbroadcast、能力未実装、ErrUnsupportedは既存のGPU合成経路を使う。

```sh
go run ./cmd/gpu-infer-bench -workloads bias_gelu,bias_residual,ffn,encoder -warmup 2 -iters 7 -json
```

bias_geluは`[1,74,3072]`、bias_residualは`[1,74,768]`。前後を別プロセスで7回ずつ測定した。

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

最大絶対誤差はBiasGELU `1.79e-7`、BiasResidual 0、FFN `2.68e-7`、Encoder `5.72e-6`で前後とも同じ。コサイン0.999以上、使用中バッファの増加はなかった。Encoderのgraph解放後のpool cacheは138,072,064→122,343,424 bytesに減った。

BiasGELU単体は短縮したが、BiasResidualとFFNの中央値では改善を確認できなかった。Encoderは約9.5%短縮したものの、CPUも573.8→547.0 ms、変更していないDirectMLも33.8→43.3 msと変動した。

## 残る制約

測定は常駐データのsteady stateで、初期化・重み転送込みの速度や、他のGPU・shapeでの速度は保証しない。前後の差を最適化だけの因果効果とは断定しない。

現在は各submitでフェンスを待ち、forward全体の記録のまとめは未実装。カーネル融合にも改善余地がある。メモリ値は入力と重みが常駐した状態のgograd管理分であり、ピークやプロセス全体のVRAM量ではない。

Linuxの実GPU計算・性能とAMD / Intel GPUは未検証。[Linux版の検証範囲](GPU_LINUX.md#検証した範囲)を参照。
