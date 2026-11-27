# 設計と計測値

## テンソル・グラフ・メモリ管理

- `Tensor` は `Shape`・`Strides`・`DType`・`Device` と共有 `tensorStorage` を持つ。CPUは固定サイズバケットの `sync.Pool` 付きGoスライス、CUDAはサイズクラス別デバイスプール、Vulkan / DirectML推論は `gpu.Buffer` を使用する。
- ストレージはlease参照カウント、グラフは子ノード依存カウントで管理する。`Close` は公開ハンドルのみ終了し、依存グラフが残る間はストレージを保持する。
- 勾配記録中は親ノード・VJPクロージャ・ストレージversionを保存し、`Backward` でトポロジカル逆順に伝播する。`NoGrad` 時は補助バッファを即時解放する。
- 推論専用デバイス（Vulkan / DirectML）は `ReleaseGraph` 用の親リンクのみ保持し、導関数・学習処理は拒否する（詳細は [GPUバックエンド](GPU_BACKENDS.md)）。

## バックエンド実装

- **CPU**：AVX2/FMAの6×16 SGEMMマイクロカーネルを実行時選択（`GOGRAD_SGEMM_4X16=0` で4×8、非対応時は純Go）。要素演算・reduction・`LayerNorm`・`GroupNorm`・`MultiheadAttention` は並列化し、AVX2/FMA環境では `GELU`（erf版）・`Softmax` の指数計算もベクトル化する。
- **CUDA**：
  - 行列積はcuBLAS（`transB` 直接指定、strided-batched GEMM、ブロードキャスト重み勾配の平坦化）を使用する。BF16有効時はGEMMオペランドに2D BF16シャドウキャッシュ（`cuda.CaptureID()` でCUDAグラフ記録中も安全に更新）を使い、累積・master weight・勾配・optimizer stateはFP32で保持する。
  - rank 6までのview・broadcast・reduction・要素演算・正規化・`SmoothL1Loss`・`CrossEntropy`・`Dropout` 等はNVRTCカーネルで実装する。`MultiheadAttention` のself-attentionはQKV一括投影とhead分割/統合カーネルを使用し、初回勾配書き込み時は `beta=0` 経路で事前のゼロ埋めを省略する。
  - **Attention**：実体化経路（cuBLAS + 融合mask/scale/softmax）と、2Dレジスタタイル化したFP32オンラインsoftmaxのFlashAttention経路（head次元128まで、Backwardはタイル再計算）を持つ。`auto` は実体化スコアscratchの見積もりが1 GiB以上の場合にFlashAttentionを選択する。
  - **Conv2d / Pooling / BatchNorm**：Conv2dは1バッチ・1グループ単位の空間タイルim2col + GEMM（予算4 MiB、出力位置最大512、Backwardでタイル再計算）。Poolingはウィンドウ幾何共通化・CUDAはatomic scatter。BatchNormはfloat64累積の統計計算とデバイス上カウンタ更新でCUDAグラフ再生に対応する。
  - **CUDAメモリプール**：デバイス・stream・サイズクラス別（256 B / 4 KiB / 64 KiB / 1 MiB刻み）、既定上限2 GiB。`cudaMalloc` 失敗時はキャッシュ解放後に1回再試行する。
- **Vulkan / DirectML**：遅延コマンドバッチ記録（最大512 op）と完了後回収（`pendingRelease`）を採用する。DirectMLは単一の拡張可能shader-visible descriptor heapと再利用 `IDMLBindingTable` を用いる。

## 計測値

環境：Windows/amd64、Go 1.27.0（`CGO_ENABLED=0`）、Intel Core i5-12400（12論理コア）、NVIDIA GeForce RTX 3060 Ti。

### Transformer 1学習ステップ（`cmd/transformer-bench`）

4層、model 256、heads 4、FFN 1024、batch 16、系列長256（forward + backward + 勾配クリップ + AdamW）。

| 実行条件 | ステップ時間 |
| --- | ---: |
| CPU FP32 | 431.9–470.3 ms |
| CUDA FP32 eager | 18.2 ms |
| CUDA FP32 graph replay | 26.7 ms |
| CUDA BF16 eager | 12.6–14.6 ms |
| CUDA BF16 graph replay | 15.2 ms |

### CUDA Attention（`cmd/attention-bench`）

batch 1、heads 4、head次元64、CUDA event計測。

| 系列長 | 実体化 forward / step | Flash forward / step | 実体化ピーク増分 | Flashピーク増分 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.077 / 0.215 ms | 0.071 / 0.250 ms | 3.50 MiB | 1.01 MiB |
| 1024 | 0.373 / 1.069 ms | 0.524 / 2.884 ms | 50.00 MiB | 4.03 MiB |
| 4096 | 5.515 / 15.315 ms | 7.446 / 28.576 ms | 776.00 MiB | 16.12 MiB |

Vulkan / DirectMLの推論計測は [GPU推論の性能](GPU_BACKEND_PERFORMANCE.md) を参照してください。
