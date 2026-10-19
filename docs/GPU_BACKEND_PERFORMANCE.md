# GPU推論の性能検証（M6 初期最適化）

## 対象と測定方法

`cmd/gpu-infer-bench`でCPU・Vulkan・DirectMLのFP32推論を同じ`autograd` APIから実行する。
学習用の`cmd/transformer-bench`とは別のコマンドで、backward、optimizer、BF16、CUDA graph captureは含まない。
CPU側のview実体化や、各バックエンドのバッファ割当・graph管理も含むAPI全体の比較であり、カーネル単体の時間ではない。

```powershell
$env:CGO_ENABLED='0'
go run ./cmd/gpu-infer-bench -warmup 2 -iters 7 -json
```

小さい形で確認する場合:

```text
go run ./cmd/gpu-infer-bench -size 128 -hidden 512 -heads 4 -layers 1 -seq 74
```

- 入力とパラメータは各デバイスに常駐させ、NoGradで実行する。データ・初期値は固定seed。
- ドライバ初期化、パラメータ構築と転送、最初のforwardは別々に報告する。
  `setup_ms`はCPU側の初期値生成も含む。`first_forward_ms`は初回の演算子コンパイルや割当の影響を受ける。
- 初回の出力をCPUと照合した後、指定回数のwarmupを追加し、`runtime.GC`を一度行ってから計測する。
- 各サンプルはforward開始からGPU完了待ちと`ReleaseGraph`が終わるまで。
  medianには初期化・転送・読み戻し・一致判定を含めない。`first_forward_ms`はgraph解放前のforward完了まで。
- GPU完了の確認には`gpu.Synchronizer`を使う。現在のVulkan / DirectMLは各演算がフェンスを待つため、
  Synchronizeで追加の空submitは行わない。結果の読み戻しで同期時間を代用しない。
- Windowsでは`QueryPerformanceCounter`、その他のOSではGoのmonotonic clockで測る。
  このWindows環境では`time.Now`を使った短いサンプルが0 msになったため、QPCへ変更した。
- median、nearest-rankのp95、全サンプルをJSONへ出力する。7サンプルのp95はその中の最大値になる。
- `atol=1e-4`、`rtol=1e-4`とコサイン類似度0.999以上を、初回と計測後の出力で検証する。
  各反復のgraph解放後に、使用中GPUバッファの数とバイト数が増えていないことも確認する。
- 未対応デバイスは理由を表示してスキップするだけで、CPUに置き換えない。
  スキップを許可しない場合は`-skip-unavailable=false`、対象を絞る場合は`-devices cpu,vulkan`などを指定する。
  対応しない演算やfeature levelはエラーにする。

ベンチマーク同士やGPUテストを同時に実行すると結果が変わる。以下の最終測定は回帰テスト完了後、1プロセス内で順番に実行した。

## 測定環境と形

- Windows/amd64、Go 1.27.0、`CGO_ENABLED=0`
- Intel Core i5-12400、12 logical CPUs、`GOMAXPROCS=12`
- NVIDIA GeForce RTX 3060 Ti
- Vulkan API 1.4.341 / DirectML feature level 6.4
- batch=1、sequence=74、model=768、FFN幅=3072、heads=12、encoder layers=12、Conv1d入力channels=74
- warmup=2（別途報告する初回forwardの後）、測定7回

| workload | 内容 |
| --- | --- |
| gemm | `[768,768] @ [768,768]` |
| projection | `[74,768] @ [768,768]` |
| ffn | `[1,74,768]` → Linear(3072) → bias + erf式GELU → Linear(768) + bias |
| encoder | time-major Conv1d（kernel=3、stride=1、padding=1）→ 12層のpre-norm TransformerEncoderLayer |

## 最終測定

時間はms、倍率は同じworkloadのCPU中央値に対する値。

| workload | CPU median / p95 | Vulkan median / p95 | Vulkan倍率 | DirectML median / p95 | DirectML倍率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gemm | 2.563 / 2.672 | 0.789 / 0.903 | 3.25x | 0.245 / 0.278 | 10.46x |
| projection | 0.705 / 0.816 | 0.149 / 0.176 | 4.72x | 0.182 / 0.240 | 3.87x |
| ffn | 25.021 / 27.319 | 1.373 / 1.991 | 18.22x | 0.634 / 1.497 | 39.50x |
| encoder | 553.505 / 612.442 | 36.174 / 37.709 | 15.30x | 33.830 / 42.922 | 16.36x |

Encoderの最大絶対誤差はVulkanで`5.72e-6`、DirectMLで`8.11e-6`。コサイン類似度はいずれも0.999以上だった。
この環境・形・常駐データの条件では、M6のEncoder 2倍という目安を両バックエンドが超えた。
AMD / Intel、別の形、初回実行まで含めた速度は保証しない。クロック固定や統計的な信頼区間の算出は行っていない。

Encoderの生サンプル（ms）:

```text
CPU:      547.8621, 612.4417, 586.6704, 553.5045, 535.0929, 565.9672, 545.8879
Vulkan:    37.7085,  37.4588,  36.1738,  35.2626,  34.3905,  36.2057,  35.3225
DirectML:  33.0404,  32.9921,  32.2440,  34.4442,  33.8297,  42.9216,  41.3080
```

同じ最終実行のEncoder構築と初回の参考値。以下は上のmedianには含めない。

| backend | パラメータ生成・構築・転送 | 初回forward | 初回出力の読み戻し |
| --- | ---: | ---: | ---: |
| CPU | 484.461 ms | 634.354 ms | 0.020 ms |
| Vulkan | 478.523 ms | 84.947 ms | 0.489 ms |
| DirectML | 501.270 ms | 138.646 ms | 0.291 ms |

ドライバ初期化はさらに別の`init_ms`として出力される。steady-stateの倍率を、
この構築・初回forward・読み戻しまで含めた倍率として解釈しない。

## DirectMLの変更前後

主なボトルネックは、packed QKVのオフセット付き転置重みをGPU上で連続化するときのコピー記録だった。
旧経路はKとVそれぞれの768×768行列を1要素ずつコピーし、12層ではその部分だけで14,155,776個のcopy regionを記録していた。

今回の変更:

1. 16バイト境界に揃うrank 4以下のGEMM入力は、オフセットを含めてDirectMLへ直接バインドする。
   オフセットは演算子のcache keyに含めず、同じ形のQ / K / Vで演算子を再利用する。
2. `gpu.ViewBinaryBackend`を追加し、対応するAdd / Mulはゼロstrideでbiasなどのブロードキャストを表す。
   展開用のGPUバッファと、そのGatherを省く。
3. 対応する非連続GatherはIdentity演算子でまとめて実行する。
   ScaleBiasを設定せず、負のゼロとNaN payloadを保つことを実機テストで確認する。
4. 非対応のrankやalignmentは、従来どおりGPU上で実体化する。Scatterの部分書き込みはbuffer copyのままにし、
   tensorのpadding領域への書き込みで別名の値を壊すことを避ける。

同じEncoder形で、変更前後をそれぞれ`-workloads encoder -warmup 1 -iters 3`で測った別実行の中央値:

| backend | 変更前 | 変更後 |
| --- | ---: | ---: |
| CPU | 578.200 ms | 506.109 ms |
| Vulkan | 39.667 ms | 35.939 ms |
| DirectML | 7,070.639 ms | 32.930 ms |

3回の測定による比較なので、細かな差の厳密な評価には使わない。
DirectMLの数秒規模のコピー記録が消えたことは、入力の実体化を省くドライバ不要テストでも確認している。
上の最終測定表は別途7回測定したもので、比較用の3回測定から都合のよいサンプルを選んだものではない。

## 残る制約

- 上の速度は常駐データのsteady state。初期化・重み転送・演算子コンパイルを含む初回の高速化は主張しない。
- まだ各演算のsubmitごとにフェンスを待つ。複数演算の記録をまとめる処理は未実装。
- DirectMLのtanh近似GELUは複数dispatch。融合演算やcacheの調整余地も残る。
- GPUのmemory統計はgraph解放後でも入力と重みが常駐した状態を表す。プロセス全体のVRAM量ではない。
- LinuxのVulkanローダー、AMD / Intelでの実機検証、READMEへの統合は別作業。
  READMEにある既存の未コミット変更は、この性能検証のコミットへ混ぜていない。
