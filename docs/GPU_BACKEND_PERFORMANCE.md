# GPU推論の性能検証（M6）

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

## viewコピー削減後の測定

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

## DirectML複合演算内のバッチ化

viewコピー削減に続き、tanh近似GELUの9段とLayerNorm / GroupNormの3段を、それぞれ1つのcommand listへ記録するようにした。
計算式とdispatch数は変えず、段ごとの状態遷移とUAVバリアを維持する。
warmup後のsubmitとフェンス待ちはGELUが9回→1回、正規化が3回→1回になる。
これは各API呼び出しの内部だけをまとめる処理で、forward全体を非同期実行するAPIではない。

- cold時は全段の演算子コンパイルと初期化を完了してから実行用listを開く。初期化のsubmitは別途必要。
- 同じ演算を繰り返す場合は、独立したdescriptor heap / binding tableを持つスロットを使う。
  一つのheapを再バインドして、未実行のdispatchが参照するdescriptorを上書きしない。
- スロットも通常のcache上限（64演算子、temporary / persistent合計256 MiB）に数える。
  近似GELUのwarm cacheは従来の7演算子から9スロットになる。
- バッチ準備中の演算子はpinして退避を防ぐ。上限を超えるものはcacheせず、実行フェンス完了後に解放する。
  外部からの解放・別スレッドからの実行は、従来どおりbackendのmutexで直列化する。

局所的な効果を測るため、`gelu`と`layernorm`をベンチマークの選択肢に追加した。既定の4 workloadは変更していない。

```text
go run ./cmd/gpu-infer-bench -workloads gelu,layernorm,encoder -warmup 2 -iters 7 -json
```

前述と同じ環境・shape・同期方法で、バッチ化前後を別プロセスで7回ずつ測定した。
`gelu`は`[1,74,3072]`のtanh近似、`layernorm`は`[1,74,768]`の最後の軸を正規化する。

| DirectML workload | 変更前 median / p95 | 変更後 median / p95 | 前後の中央値比 |
| --- | ---: | ---: | ---: |
| gelu | 0.8369 / 1.3339 ms | 0.1851 / 0.9337 ms | 4.52x |
| layernorm | 0.3169 / 0.4068 ms | 0.2495 / 0.2770 ms | 1.27x |
| encoder | 34.7472 / 36.8871 ms | 35.2579 / 37.8502 ms | 0.99x |

DirectMLの生サンプル（ms、実行順）:

```text
gelu before:      0.8590,  0.8369,  0.9156,  1.3339,  0.7268,  0.8280,  0.7786
gelu after:       0.3587,  0.9337,  0.2049,  0.1851,  0.1637,  0.1748,  0.1712
layernorm before: 0.4068,  0.3923,  0.3529,  0.3169,  0.2999,  0.2647,  0.2534
layernorm after:  0.2106,  0.2661,  0.2622,  0.2770,  0.2495,  0.2137,  0.2360
encoder before:  33.5330, 34.7472, 31.4275, 36.8871, 34.2618, 34.8019, 35.7983
encoder after:   34.6284, 36.9622, 35.2579, 35.2272, 35.2707, 33.6217, 37.8502
```

最大絶対誤差は近似GELUが`4.77e-7`、LayerNormが`8.34e-7`、Encoderが`8.11e-6`で、変更前後で同じだった。
コサイン類似度もすべて0.999以上。計測中の使用中バッファ数・バイト数の増加はなかった。

**Encoder全体の改善は、この測定では確認できなかった。** Encoderはerf式GELUを使うため、近似GELUのバッチ化は適用されない。
正規化のsubmit削減は適用されるが、この実行ではCPUのEncoder中央値も532.0→552.7 ms、
変更していないVulkanも35.9→40.6 msと変動した。単一演算の中央値比をEncoderの改善率として扱わない。
LayerNorm単体は変更後もCPU（0.1475 ms）より遅く、すべての演算がCPUより速いとは主張しない。

検証にはwarm時のsubmitが1回であること、cold時の初期化回数、cache満杯時のpin、64スロットを超えるバッチ、
繰り返し実行とcache解放後の再構築を確認する実機テストを追加した。
ドライバ不要テストは近似GELUの全9段と正規化の全3段への失敗注入を含み、エラー伝搬とscratch回収を確認する。
全リポジトリのstrict GPUテスト、`go vet ./...`、checkptr検査も成功した。

## Vulkanのbias融合演算

`gpu.FusedBiasBackend`を追加し、Vulkanの`BiasGELU`と`BiasResidual`を1 dispatch / submitへ融合した。
いずれもbiasは1次元、入力の最後の軸に対応する。residualは入力と同じshapeが対象。
それ以外の一般ブロードキャスト、能力未実装のバックエンド、`ErrUnsupported`は従来のGPU演算を合成する。
実行エラーを隠したり、CPUへ計算を移したりしない。

- 連続入力なら、従来の「bias展開のGather → Add → GELU / Add」の3 dispatchを1回に減らす。
  `Alloc`の要求も、bias展開と中間値を含めた3回から最終出力の1回になる（実際の物理割当はpool再利用に依存する）。
- 既存pointwise SPIR-Vに分岐を追加した。推論時のシェーダコンパイルや新しい外部依存はない。
  residual用の4番目のdescriptorは、分岐を使わない演算にも有効なdummyを設定する。
- GELUは従来と同じerf近似を使い、tanh近似へ変えない。残差加算の順序は`(x + bias) + residual`を保つ。
- 転置やoffset付き入力は必要なものだけGPU上で連続化し、そのscratchを呼出し内で回収する。
  CPU/CUDAの学習・BF16・既存の融合演算経路は変更していない。

局所ベンチマークに`bias_gelu`（`[1,74,3072]`）と`bias_residual`（`[1,74,768]`）を追加した。
既定の4 workloadは変更していない。

```text
go run ./cmd/gpu-infer-bench -workloads bias_gelu,bias_residual,ffn,encoder -warmup 2 -iters 7 -json
```

同じ測定環境・同期方法で、融合前後を別プロセスで7回ずつ測った。回帰テストと並行には実行していない。

| Vulkan workload | 変更前 median / p95 | 変更後 median / p95 | 前後の中央値比 |
| --- | ---: | ---: | ---: |
| bias_gelu | 0.1850 / 0.2801 ms | 0.0771 / 0.1727 ms | 2.40x |
| bias_residual | 0.2475 / 0.3471 ms | 0.2473 / 0.3459 ms | 1.00x |
| ffn | 1.3367 / 1.6453 ms | 1.4333 / 2.3510 ms | 0.93x |
| encoder | 39.8714 / 43.8014 ms | 36.1015 / 37.5938 ms | 1.10x |

Vulkanの生サンプル（ms、実行順）:

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

最大絶対誤差はBiasGELUが`1.79e-7`、BiasResidualが0、FFNが`2.68e-7`、Encoderが`5.72e-6`で、変更前後で同じだった。
コサイン類似度はすべて0.999以上。計測中に使用中バッファ数・バイト数が増えないことも確認した。
Encoderのgraph解放後のpool cacheは`138,072,064`→`122,343,424` bytesとなった。
これはその時点の再利用用cacheであり、ピークVRAMやプロセス全体のメモリ使用量ではない。

**BiasGELU単体は短縮したが、BiasResidualとFFNの改善は今回の中央値からは確認できなかった。**
Encoderの中央値は約9.5%短縮したものの、CPUも573.8→547.0 msと変動し、
変更していないDirectMLも33.8→43.3 msと変動した。厳密な因果効果や、全環境での10%改善を主張するものではない。

実機テストは0要素、端数workgroup、2次元dispatch、入力alias、出力alias拒否、加算順序、CPU一致を含む。
ドライバ不要テストは能力による選択、`ErrUnsupported`での合成経路への復帰、単一dispatch・scratchなしの呼出し、
実行失敗時の出力・連続化scratch回収を確認する。strict GPUテスト、`go vet ./...`、Vulkanを含むcheckptr検査も成功した。

## 残る制約

- 上の速度は常駐データのsteady state。初期化・重み転送・演算子コンパイルを含む初回の高速化は主張しない。
- まだ各submitでフェンスを待つ。DirectMLの近似GELUと正規化内部はバッチ化したが、forward全体の記録のまとめは未実装。
- DirectMLのtanh近似GELUは1つのcommand list内で複数dispatch。カーネル融合やcacheの調整余地も残る。
- GPUのmemory統計はgraph解放後でも入力と重みが常駐した状態を表す。プロセス全体のVRAM量ではない。
- LinuxのVulkanローダー、AMD / Intelでの実機検証、READMEへの統合は別作業。
  READMEにある既存の未コミット変更は、この性能検証のコミットへ混ぜていない。
