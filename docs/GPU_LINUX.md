# Linux版Vulkan（amd64）

## 実装と現在の位置づけ

Linux/amd64のVulkanバックエンドを実装した。Windowsと同じ演算、メモリ管理、バリア、フェンス待ち、
埋め込みSPIR-Vを使い、ローダーと外部関数呼出しだけをOS別に分けている。
**Linux上の実GPU計算はまだ未検証のため、現時点では実験的な対応。**

- 対象はLinux/amd64（x86_64）。Linux/arm64、Android、その他のOSはスタブのまま。
- 推論のみ。既存のCPU/CUDA学習経路は変更していない。
- Vulkan 1.1以上と、既存のworkgroup・共有メモリ・push constant要件を満たす計算GPUが必要。
- llvmpipe / lavapipeなどのCPU型デバイスは選ばない。CPU実装のVulkanをGPUとして表示しない。

## 依存とローダー

Linuxの呼出し基盤として、純Goの`github.com/ebitengine/purego v0.11.2`を1依存追加した。
`CGO_ENABLED=0`でビルドでき、利用者にCコンパイラやVulkan SDKを要求しない。
Windowsのローダーは従来の標準`syscall`を使い、`purego`のコードをimportしない。

Linuxでは次の共有ライブラリが実行時に必要:

- `purego`のUnix呼出し基盤が使う標準のlibc / libdl。glibc環境で検証した。musl環境は未検証。
- Vulkanローダー`libvulkan.so.1`と、GPUベンダーの対応するVulkan ICD。

`dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL)`で読み、`vkGetInstanceProcAddr`から関数を解決する。
SONAMEは通常の動的リンカの規則で解決されるため、`LD_LIBRARY_PATH`などのシステム設定の影響を受ける。
`./libvulkan.so`や開発用の非version付きライブラリを独自の代替候補にはしない。
バックエンドのimport / 登録だけではVulkanローダーやICDを読み込まない。

**cgoなしは完全静的リンクを意味しない。** libc / libdlはプロセス起動の依存となり得る。
これらが欠けて起動できない環境と、起動後に`Available()`で判定できるVulkanの未導入は別の条件になる。

## 使い方と利用不可時の扱い

```sh
CGO_ENABLED=0 go run ./cmd/vulkan-infer -n 100003
CGO_ENABLED=0 go run ./cmd/vulkan-infer -linear
CGO_ENABLED=0 go run ./cmd/gpu-infer-bench -devices cpu,vulkan -skip-unavailable=false -warmup 2 -iters 7
```

Vulkanローダー、対応ICD、条件を満たすGPUがなければ、`Available()`はfalse、`Err()`は`gpu.ErrUnavailable`を返す。
GPU上の演算を勝手にCPUで実行しない。CPUを使う場合はテンソルやモデルの構築前に明示的に選ぶ。
既存の例では、次のフラグがその選択に相当する:

```sh
CGO_ENABLED=0 go run ./cmd/vulkan-infer -cpu-fallback
```

GPUなしでも動くテストは通常実行できる。利用可能なGPUがなければ実機テストだけがスキップされる。

```sh
CGO_ENABLED=0 go test ./... -count=1
CGO_ENABLED=0 go vet ./...
```

実GPUのあるLinux環境で、スキップを許可せずに検証する場合:

```sh
CGO_ENABLED=0 GOGRAD_REQUIRE_VULKAN=1 go test ./vulkan ./autograd -count=1
```

## 検証した範囲

WSL2 Debian（kernel 5.15.167.4、glibc 2.41、Go 1.25.0）で確認した:

- `CGO_ENABLED=0`の全リポジトリテストと`go vet ./...`。
- Vulkanローダー読込み・symbol解決・physical device列挙・ライブラリの繰り返し解放。
- ライブラリ欠落、symbol欠落、ICD欠落、import時の遅延読込み、失敗後の割当拒否。
- 専用のテスト用SysV関数に対する10引数、64 bit値、戻り値、出力ポインタ、スタックalignment。
  並行呼出しとGCを重ねたテストも含む。テスト関数はGo assembler製で、Cコンパイラは不要。
- `-gcflags=all=-d=checkptr=2`でのVulkan / autograd検査。
- Go 1.27.0でLinux用にクロスコンパイルしたテストバイナリでも、同じWSLでローダー・ABIテストを実行して成功した。
- `CGO_ENABLED=1`とWSLのCコンパイラを使った、選択したローダー・ABI・GPU共通層・autogradテストの`-race`検査。
  race検査のためにcgoを有効にしただけで、通常ビルドの要件ではない。

このWSLで列挙されたのは`llvmpipe (LLVM 19.1.7, 256 bits)`、CPU型（type=4）、Vulkan 1.4.305だった。
正しく拒否されることを確認したが、**LinuxのGPU演算・CPU一致・性能は検証できていない。**
Windows/amd64のRTX 3060 Tiでは、共有コードへの移行後もstrict GPU全テストとcheckptr検査が成功した。
これは共有演算経路の回帰確認であり、Linux実機検証の代わりではない。

次に必要なのはLinux実GPUでのstrictテストとベンチマーク、およびAMD / Intel GPUでの確認。
