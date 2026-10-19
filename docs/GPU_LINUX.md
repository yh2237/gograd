# Linux版Vulkan（amd64）

## 実装と現在の位置づけ

Linux/amd64のVulkanは実験的な対応。ローダーと外部関数呼出しを実装し、演算・メモリ管理・同期・SPIR-VはWindowsと共有している。Linux上の実GPU計算は未検証。

Vulkan 1.1以上と、workgroup・共有メモリ・push constantの要件を満たす計算GPUが必要。llvmpipe / lavapipeなどのCPU型デバイスは選ばない。Linux/arm64、Android、その他のOSはスタブのまま。推論専用で、学習はCPU/CUDAを使う。

## 依存とローダー

Linuxの外部関数呼出しには`github.com/ebitengine/purego v0.11.2`を使う。`CGO_ENABLED=0`でビルドでき、CコンパイラやVulkan SDKは不要。Windowsは標準の`syscall`を使う。

実行時にはlibc / libdl、`libvulkan.so.1`、対応するGPUのVulkan ICDが必要。glibc環境で検証しており、muslは未検証。cgoなしでも完全静的リンクにはならず、libc / libdlが欠けると`Available()`による判定より前にプロセスが起動できない場合がある。

Vulkanの読込みは初回利用時に行う。`dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL)`と`vkGetInstanceProcAddr`で関数を解決するため、`LD_LIBRARY_PATH`などの動的リンカの設定に従う。`./libvulkan.so`や開発用の非version付きライブラリへの独自の切替はしない。

## 使い方と利用不可時の扱い

```sh
CGO_ENABLED=0 go run ./cmd/vulkan-infer -n 100003
CGO_ENABLED=0 go run ./cmd/vulkan-infer -linear
CGO_ENABLED=0 go run ./cmd/gpu-infer-bench -devices cpu,vulkan -skip-unavailable=false -warmup 2 -iters 7
```

Vulkanローダー・対応ICD・条件を満たすGPUがなければ`Available()`はfalse、`Err()`は`gpu.ErrUnavailable`を返す。CPUへの切替はモデルやテンソルを作る前に明示する。

```sh
CGO_ENABLED=0 go run ./cmd/vulkan-infer -cpu-fallback
```

通常のテストは利用できないGPUの実機テストをスキップする。実GPUのある環境では`GOGRAD_REQUIRE_VULKAN=1`でスキップを禁止できる。

```sh
CGO_ENABLED=0 go test ./... -count=1
CGO_ENABLED=0 go vet ./...
CGO_ENABLED=0 GOGRAD_REQUIRE_VULKAN=1 go test ./vulkan ./autograd -count=1
```

## 検証した範囲

WSL2 Debian、kernel 5.15.167.4、glibc 2.41、Go 1.25.0で、全テスト・`go vet`・checkptr検査が成功した。ローダーの読込みと解放、関数解決、デバイス列挙、ライブラリ・symbol・ICD欠落、遅延読込み、失敗後の割当拒否を検証している。

テスト用のSysV関数で、10引数、64 bit値、戻り値、出力ポインタ、スタックalignment、並行呼出しとGCを検証した。Go assembler製のためCコンパイラは不要。Go 1.27.0でクロスコンパイルしたLinuxバイナリでもローダー・ABIテストが成功した。選択したローダー・ABI・共通層・autogradテストは、race検査用にcgoとCコンパイラを使った`-race`でも成功している。

列挙されたのはCPU型の`llvmpipe (LLVM 19.1.7, 256 bits)`（type=4、Vulkan 1.4.305）だけで、`Available() == false`を確認した。LinuxのGPU演算・CPU一致・性能は未検証。共有コードのWindows/amd64での回帰テストはRTX 3060 Tiで成功しているが、Linux実機での確認は別途必要。
