# Linux版Vulkan（amd64）

## 実装と現在の位置づけ

Linux/amd64向けのVulkanは実験的な実装である。ローダーとFFI呼び出し層を実装し、演算・メモリ管理・同期・SPIR-VシェーダはWindows版と共有している（Linux実GPUでの計算は未検証）。

実行にはVulkan 1.1以上と、workgroup・共有メモリ・push constant要件を満たす計算GPUが必要である（`llvmpipe` / `lavapipe` 等のCPU実装は自動除外する）。Linux/arm64やAndroid等の他OSはスタブのままとなっている。推論専用であり、学習にはCPU / CUDAを使用する。

## 依存とローダー

外部関数呼び出しには `github.com/ebitengine/purego v0.11.2` を採用しており、`CGO_ENABLED=0` でCコンパイラやVulkan SDKなしにビルドできる（Windows版は標準の `syscall` を使用）。

実行時には `libc` / `libdl`、`libvulkan.so.1`、およびGPUに対応するVulkan ICDが必要である（glibc環境で検証済み、muslは未検証）。cgo不使用でも完全静的リンクにはならないため、`libc` / `libdl` が欠落している環境では `Available()` による判定前にプロセス起動が失敗する場合がある。

Vulkanライブラリは初回利用時に `dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL)` と `vkGetInstanceProcAddr` で解決するため、`LD_LIBRARY_PATH` 等の動的リンカ設定に従う（`./libvulkan.so` やversionなしの `.so` への独自フォールバックは行わない）。

## 使い方と利用不可時の扱い

```sh
CGO_ENABLED=0 go run ./cmd/vulkan-infer -n 100003
CGO_ENABLED=0 go run ./cmd/vulkan-infer -linear
CGO_ENABLED=0 go run ./cmd/gpu-infer-bench -devices cpu,vulkan -skip-unavailable=false -warmup 2 -iters 7
```

Vulkanローダー・対応ICD・要件を満たすGPUが存在しない場合、`Available()` は `false`、`Err()` は `gpu.ErrUnavailable` を返す。CPUへのフォールバックはモデルやテンソルの構築前に明示的に指定する。

```sh
CGO_ENABLED=0 go run ./cmd/vulkan-infer -cpu-fallback
```

通常のテストではGPU非搭載環境の実機テストを自動スキップする。実GPU環境でスキップを禁止する場合は `GOGRAD_REQUIRE_VULKAN=1` を指定する。

```sh
CGO_ENABLED=0 go test ./... -count=1
CGO_ENABLED=0 go vet ./...
CGO_ENABLED=0 GOGRAD_REQUIRE_VULKAN=1 go test ./vulkan ./autograd -count=1
```

## 検証した範囲

WSL2 Debian（kernel 5.15.167.4、glibc 2.41、Go 1.25.0）にて、全テスト、`go vet`、`checkptr` 検査に合格している。主な検証項目は以下の通り。

- ローダーの読み込み・解放、関数シンボル解決、デバイス列挙、ライブラリ・シンボル・ICD欠落時のエラー処理、遅延ロード、失敗後の割当拒否
- Goアセンブラ製のテスト用SysV ABI関数を用いた、10引数、64-bit値、戻り値、出力ポインタ、スタックアライメント、並行呼び出し、およびGC耐性の検証
- Go 1.27.0でクロスコンパイルしたLinuxバイナリでのローダー・ABIテスト、およびcgo有効下での `-race` データ競合検査

同環境ではCPU実装である `llvmpipe (LLVM 19.1.7, 256 bits)`（type=4、Vulkan 1.4.305）のみが列挙され、`Available() == false` となることを確認した。共有コードの回帰テストはWindows/amd64（RTX 3060 Ti）で合格しているが、Linux実GPUでの演算・精度・性能検証は別途必要である。