# Linux版Vulkan（amd64）

## 対象と依存

- 対象は `linux/amd64` のFP32推論専用実装（Vulkan 1.1以上の計算GPUが必要、`llvmpipe` / `lavapipe` 等のCPU実装は自動除外）。`linux/arm64` やAndroidはスタブ（`Available() == false`）。
- 演算・メモリ管理・同期・埋め込みSPIR-VシェーダはWindows版と共通。
- FFI呼び出しには標準ライブラリと自前の最小 `linux/amd64` SysV ABI / 動的ローダー層を使用するため `CGO_ENABLED=0`・外部依存ゼロでビルド可能。ただし実行時に `libc` / `libdl` / `libpthread`、`libvulkan.so.1`、対応Vulkan ICDが必要であり完全静的リンクにはならない（glibc検証済み、musl未検証）。
- 初回利用時に `dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL)` と `vkGetInstanceProcAddr` で解決する（`./libvulkan.so` やバージョンなし `.so` への独自フォールバックは行わない）。

## 利用不可時の挙動と検証範囲

- ローダー・ICD・要件を満たすGPUがない場合、`Available() == false`、`Err()` は `gpu.ErrUnavailable` を返す。
- GPU非搭載環境では実機テストを自動スキップする。スキップを禁止する場合は `GOGRAD_REQUIRE_VULKAN=1` を指定する。
- WSL2 Debian（glibc 2.41）でローダー・SysV ABI・エラー処理・`checkptr`・`-race` 検査、および `llvmpipe` 自動除外を確認済み。Linux実GPU上での計算・精度・性能は未検証。
