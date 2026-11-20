# GPUバックエンド

VulkanとDirectMLは `autograd` 経由で利用するFP32推論専用バックエンドです（学習はCPU / CUDAを使用）。対象外のOS・アーキテクチャではスタブが組み込まれ、`Available() == false` を返します。

## 対応演算

DirectML / Vulkan ともに以下の推論演算に対応します（未対応演算は CPU へ自動フォールバックせず `gpu.ErrUnsupported` を返します。`Log1p`, `Expm1`, `Embedding`, `FlashAttention` は両バックエンド未対応）。

| カテゴリ | 対応演算 |
| --- | --- |
| 二項演算（broadcast対応） | `Add`, `Sub`, `Mul`, `Div`, `Max`, `Min` |
| スカラー演算 | `AddScalar`, `SubScalar`, `MulScalar`, `DivScalar`, `PowScalar` |
| 単項演算 | `Exp`, `Log`, `Abs`, `Tanh`, `ReLU`, `Sin`, `Cos`, `Sqrt`, `Sigmoid`, `Softplus`, `Erf`（DirectML は FL 2.0+）, `Clamp`, `GELU`（tanh近似 / erf、DirectML の erf は FL 5.1+） |
| Reduction / 正規化 / 確率 | `Sum`, `Mean`（rank 0〜6）, `Softmax`, `LogSoftmax`, `LayerNorm`, `GroupNorm`, `BatchNorm`（評価のみ）, `RMSNormLast` |
| 畳み込み・Pooling | `Conv1d`, `Conv2d`, `MaxPool2d`（DirectML の dilation は FL 2.1+）, `AvgPool2d`, `AdaptiveAvgPool2d`（割り切れるサイズ） |
| 行列積・view・形状 | `MatMul`（転置4通り、rank 6まで）, `Reshape`, `Unsqueeze`, `Squeeze`, `Flatten`, `Permute`, `Transpose`, `Slice`, `Expand`, `Concat`, `Stack`, `Gather` / `Scatter` |
| Attention・ブロック・モデル | `ScaledDotProductAttention`（実体化経路）, `KeyPaddingAttention`, `RotaryHalf`, `GeGLU`, `SwiGLUProjection`, `LinearLayer`, `Conv1dLayer`, `Conv2dLayer`, `MaxPool2dLayer`, `AvgPool2dLayer`, `AdaptiveAvgPool2dLayer`, `BatchNormLayer`（評価）, `TransformerEncoderLayer`（評価） |

## 実装方式

- **Vulkan**：埋め込みSPIR-V（`vulkan/shaders/*.spv`、計8シェーダ）、device-localバッファ、host-visible/coherentステージングを使用。演算・コピーは1コマンドバッファへ遅延記録し（上限512 descriptor set / 512 op）、`Download` / `CopyFrom` / `Synchronize` / `ReleasePool` / `Close` またはプール圧力時に一括submitしてフェンス待機する。`gpu.ViewBinaryBackend` / `gpu.ViewBinaryOpBackend` および `gpu.FusedBiasBackend`（`BiasGELU` / `BiasResidual`）は1 dispatchで評価する。`Conv2d` はGPU上im2col＋バッチGEMM（＋任意bias）、`BatchNorm` 評価は正規化＋任意affineで評価する。
- **DirectML**：純GoのCOM呼び出しでD3D12 `DEFAULT` / `UPLOAD` / `READBACK` リソースを使用。コマンドキューへ遅延蓄積し（上限512 op / 1,048,576 copy region）、2フェーズflush（1: 演算子のコンパイル・キャッシュpin、2: チャンク全体のリソース状態遷移追跡と単一の拡張可能shader-visible descriptor heap〔初期4096記述子〕・再利用 `IDMLBindingTable` による記録・submit・フェンス待機）を実行する。rank 4以下かつ16バイト境界に揃うGEMM・二項演算は記述子で直接読み取り、対応する非連続 `Gather` は `ScaleBias` なしの `Identity` 演算子を使用する。

両バックエンドとも、バッチ実行中に解放されたバッファは `pendingRelease` へ退避し、flush完了（フェンス通過）後にプールへ回収する。

## 制約

- **ドライバ読み込み**：初回利用時に動的ロードする（WindowsはシステムディレクトリのDLL、Linuxは `libvulkan.so.1`）。初期化失敗やsubmit失敗はキャッシュされ以降は `gpu.ErrUnavailable` となる。CPU型デバイス・WARPは自動選択から除外される。
- **推論専用制約**：記録有効かつ `RequiresGrad=true` の入力は出力割当前に拒否されるため、`ExecutionContext.NoGrad` 内で実行する。backward、optimizer、勾配クリップ、`BatchNorm` の学習モード・running統計なしモード、学習時 `Dropout` は `NoGrad` 内でも `gpu.ErrInferenceOnly` となる。
- **DirectML固有の制限**：
  - Feature Level要件：`Erf` はFL 2.0以上、dilation付き `MaxPool2d` はFL 2.1以上、erf版 `GELU` はFL 5.1以上。
  - `Conv1d` で `dilation > 1 && outLen == 1 && inLen == kernel` の条件、およびWARP上の `Conv2d` で単一出力位置のdilatedカーネルが軸長と一致する条件は、DirectMLが0を返す不具合回避のため `gpu.ErrUnsupported` とする。
  - `Clamp` は入力の `NaN` を保持し、境界との比較では `-0` を `+0` より小さい値として扱う。`lo > hi` または境界値が `NaN` の場合は `gpu.ErrUnsupported`。
  - `ScalarDiv` のsubnormal除数、`AvgPool2d` の `DivisorOverride != 0` および `CeilMode` でパディング拡張かつ `!ExcludePad` の組み合わせ、`AdaptiveAvgPool2d` の割り切れないサイズはいずれも `gpu.ErrUnsupported`。
- **Vulkan固有の制限**：`Erf` および erf版 `GELU` は最大誤差 `1.5e-7` の多項式近似。`Conv1d` / `Conv2d` の im2col バッファが `MaxStorageBufferBytes` を超える場合は `gpu.ErrUnsupported`。`AvgPool2d` / `AdaptiveAvgPool2d` の制約は DirectML と同一。
- **サイズ単位・スレッド安全性**：`gpu` 共通APIのサイズ・stride・offsetはfloat32要素数単位（負のstride不可）。バックエンド内部はmutex同期のためOSスレッド固定不要。
- **プール・キャッシュ・寿命**：バッファプール既定上限は256 MiB（`MemoryStats` / `SetPoolCacheLimit` / `ReleasePool`）。DirectML演算子キャッシュは最大64件・合計256 MiB（`OperatorStats` / `ReleaseOperators`）。`NewBackend()` の独立バックエンドは全バッファ解放後に `Close()` する（登録済み既定バックエンドは `Close()` 不可）。Vulkanで完了確認できない割当は `QuarantinedBytes` / `QuarantinedBuffers` に隔離報告される。

## シェーダの再生成

Vulkanの `.comp` ファイルを変更した場合は、`glslang` 16.6.0を `PATH` に通して `go generate ./vulkan` で `.spv`（Vulkan 1.1対象）を再生成する。CIの `vulkan-shaders` ジョブで、同じバージョンで再生成した `.spv` がコミット済みのファイルと一致することを確認する。通常ビルドには `glslang` は不要。

DirectML の `directml/shaders/clamp.hlsl` を変更した場合は、Windows SDK の `fxc` を `PATH` に追加し、Windows amd64 で `go generate ./directml` を実行して `clamp.cso` と `clamp.rootsig` を更新する。

## テスト

- GPU非搭載環境では実機テストを自動スキップする。スキップ禁止には `GOGRAD_REQUIRE_VULKAN=1` または `GOGRAD_REQUIRE_DIRECTML=1` を指定する。
- `GOGRAD_DIRECTML_SOFTWARE=1` でDirectMLのソフトウェアラスタライザ（WARP）を使用できる。
