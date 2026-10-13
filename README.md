# gograd

Goで書いた逆伝播の自動微分ライブラリです。参照エンジンはCPUのfloat64で計算し、汎用の`tensor`と`nn`はCPUまたはCUDAでfloat32を扱います。cgoやCコンパイラ、Pythonは必要ありません。WindowsではCUDAランタイム、cuBLAS、NVRTCのDLLをシステムローダー経由で呼び出します。他のプラットフォームではCUDAの操作は`ErrUnavailable`を返し、`MemoryStats`は0を返します。

## 概要

参照エンジンはCPUのfloat64で実装し、TCNとその損失に必要な演算を持ちます。

- `Linear`、`tanh`、要素ごとの加算・減算・スケール、`clamp`
- 対称ゼロパディングのdilated `Conv1d`（カーネル3、層ごとのdilation）
- transposeとreshape、隣接フレームの差分
- マスク付きgather、行ごとのメディアン中心化、Smooth L1損失
- decoupled weight decayとグローバル勾配ノルムクリップ付きのAdamW
- Goランタイムが読む`frame_pitch` JSONの出力

`cuda`パッケージは、デバイスの照会、明示的な`Free`を持つデバイスバッファ、stream、event、row-majorの`SgemmRowMajor`、im2colとSGEMMによる`Conv1dForward`、CUDAドライバ経由で読み込むNVRTCカーネルを提供します。連続した処理を`Graph`に記録し、1回のlaunchで再生できます。解放したバッファはサイズクラスごとのプールから再利用します。既定の上限は2 GiBで、系列長が変わっても割当サイズごとに保持しません。`cuda.MemoryStats`は使用中・キャッシュ・予約・ピークのバイト数を返し、`SetPoolCacheLimit`は上限を変更し、`ReleasePool`はキャッシュをドライバへ返します。device、暗黙のstream、capture状態は呼び出したOSスレッドに属します。プールの項目と`Buffer`は自分のdeviceを記録し、コピー・memset・freeはそのdeviceを選ぶため、`autograd.NewCUDAContextOn`で別々のコンテキストを持つgoroutineは独立したstream、capture、deviceを保ちます。

`autograd`パッケージはdefine-by-runのfloat32グラフで、rank 6までのview、N次元のブロードキャストとreduction、バッチmatmul、パディング付きembedding、Conv1d、groups/depthwiseのNCHW Conv2d、正規化、softmax、max/average/adaptive-average pooling、log-softmax、cross-entropy、決定的dropout、scaled attention、マスク付き損失、クリップ、AdamW、OneCycle LRを持ちます。BatchNormはchannels-first/last、学習時と評価時の統計、affineパラメータ、名前付きrunning stateをCPU/CUDAで扱います。AdamWとSGDはOptimizerインターフェースとOneCycleのスケジューリングを共有します。SGDはmomentum/dampening、Nesterov、coupled decay、ascentに対応し、状態の保存と再開をCPU/CUDAで行います。両者は名前付きのパラメータグループを持て、グループごとの学習率スケールとweight decayをoptimizer stateと学習checkpointに記録します。opレジストリは公開する微分可能なテンソル演算ごとにCPUとCUDAの実装を並べ、deviceのバックエンドを選びます。dispatchを迂回した演算はソースレベルのテストで検出します。pre-normの`TransformerEncoderLayer`とbatch-firstの`MultiheadAttention`はこれらの演算を組み合わせます。`Module`は名前付きstateと、PyTorchの出力と互換なF32 safetensorsファイルを扱います。再利用できる`EmbeddingLayer`、`Conv1dLayer`、`LayerNormLayer`、`LinearLayer`、`DropoutLayer`、`GELULayer`を組み合わせてモデルを構成できます。`Module.Train`は登録した子へtrain/evalモードを再帰的に伝え、`WindowSampler`と`IndexLoader`は再開できるサンプリング状態を公開します。学習APIの詳細は[`docs/API.md`](docs/API.md)にあります。

`ExecutionContext`は入力とモジュールのstateを明示的な勾配記録スコープに関連付けます。NoGradは入れ子になり、panic後も復帰します。独立したCPUモデルは、互いのグラフを止めずに学習と推論を並行して実行できます。`cmd/cnn-train`はパラメータと入力のバインド、checkpointの再読込を示します。各コンテキストはBF16 autocastとアテンション選択の`ExecutionOptions`を持ち、既定はFP32と`auto`です。`execution.Autocast`は精度を局所的に切り替え、backwardはforward時に選んだ方針をscopeの終了後やオプション変更後も保ちます。

`data`パッケージは、汎用のDataset、indexのサンプリング、ホスト側のcollateを分離します。Loaderはseed付きshuffle、subset/split、パディング付き系列、有効値マスク、epoch途中のcheckpoint状態に対応します。任意のworkerと上限付きprefetchはバッチ順を保ち、checkpointには受け渡し済みのサンプルだけを記録します。worker 0では同期読込になります。数値のCSVテーブルと直列化できる特徴量標準化で回帰のデータを扱います。非圧縮WAVの読み書きは音声を扱い、例ではフレーム化とaugmentationをアプリ側に置き、サンプルごとに決まるノイズを使います。`cmd/mnist-train`はraw/gzipのMNIST IDXを読み、MLPをCPU/CUDAで学習・評価し、推論用の重みを保存し、model/AdamW/OneCycleとデータ順をPythonなしで再開します。`cmd/tabular-train`は数値CSVを読み（または生成し）、学習データの統計で特徴量を標準化し、同じ状態で再開します。`cmd/audio-train`はWAVを読み（または生成し）、indexから決まるノイズでwindowをaugmentし、denoiseモデルを同じ重みで再開します。

テンソルとviewのストレージはallocation leaseで共有します。ベースをcloseしたり、あるグラフの枝をreleaseしても、別の依存がまだ必要とする値は保持します。`RetainData`は値を固定し、`BackwardWithOptions`は複数の損失のために履歴を明示的に保持します。履歴が終わると保存したstateはプールへ戻ります。inputとパラメータの管理された変更はbackwardの前にversionを検証します。

CUDAのバッチmatmulは、通常とブロードキャストのバッチを1回のstrided cuBLAS呼び出しで処理します。アテンションはバッチGEMMと融合したマスク付きsoftmaxカーネルを組み合わせます。任意のtiled CUDA経路はonline softmaxを使い、backwardでスコアを再計算し、完全なスコア行列を保持しません。加算的なブロードキャストマスクに対応します。GEMM経路はscratchの見積もりが1 GiB未満のとき自動的に選ばれます。計測した系列長ではこちらが速いためです。CUDAのlayer normとencoderのfeed-forwardではbias/GELU/residualを融合カーネルで処理します。warp softmaxは任意の軸を扱います。任意の`-bf16`はBF16のGEMMオペランドをデバイスにキャッシュし、viewを通してBF16シャドウを保ち、pointwiseカーネルへのBF16出力書き込みを融合します。GEMMExはFP32で累積し、勾配とmaster weightはFP32のままです。既定はFP32です。

`dsp`パッケージは、2の冪の複素FFT、周期Hann窓、magnitude/phaseのSTFTとその逆変換、polyphaseリサンプラを提供します。FFTはfloat64で累積し、出力時に丸めるため、PyTorchのfloat32 pocketfftと約1 ulpで一致します。`torchrng`は`torch.Generator(device="cpu")`をbit単位で再現します。float32のlog、sin、cosにPyTorchのAVX2カーネルが使うcephes多項式近似も含むため、Pythonでseedしたノイズfixtureが不要になります。

`autograd.Acoustic`は音響モデルを構成し、PyTorchの学習ステップfixtureとCPU/CUDAで照合しています。CUDAのテンソルと勾配はデバイスバッファ上にあり、グラフの演算はcuBLASとNVRTCカーネルを使い、クリップとAdamWはデバイス上で動きます。CUDAを使う呼び出し側は`autograd.NewCUDAContext`でOSスレッドを固定します。CPUのConv1dはAVX2/FMAの8x8マイクロカーネルを持つブロックSGEMMと純Goのフォールバックを使い、バッチmatmulも同じ実装です。Acousticのステップは、固定したデバイスindexとデバイス上のAdamWステップカウンタでCUDAグラフに記録できます。設計の説明は[`docs/DESIGN.md`](docs/DESIGN.md)にあります。

`tensor`と`nn`は先にあった汎用のfloat32経路です。`NewOn`、`FromHostOn`、`NewLinearOn`、`NewConv1dOn`で構築時に`tensor.CPU`または`tensor.CUDA`を選びます。元のコンストラクタはCUDAを選びます。CPUの行列積は、AVX2/FMAが使えるときはマルチスレッドのブロックSGEMM、それ以外は純Goのフォールバックを使います。モジュールは`[batch,time,channels]`を扱い、Linear、same-paddingのdilated Conv1d、ReLU、LeakyReLU、tanh近似のGELU、Tanh、Sequential、Residualを含みます。2つのマスク付き損失は`[batch,time]`のフレーム重み（パディングは0）と任意のチャネル重みを受け取り、要素重みの合計で割って損失と出力勾配を返します。AdamWはグローバル勾配ノルムをクリップし、どちらのデバイスでもdecoupled weight decayを使います。

汎用のCUDA Conv1dはim2col/col2imカーネルとcuBLASでforwardとbackwardを行います。活性化、残差加算、マスク付き損失、AdamWはデバイス上で動きます。学習ループは損失の重みをデバイスに置いたまま、各ステップで必要なスカラー損失だけをダウンロードします。

`gputcn`パッケージはTCNのforwardとbackwardをGPUのfloat32で実行します。forwardの出力はPyTorchのfloat64参照と約5e-08、パラメータ勾配は約1e-05で一致します。損失とAdamW（グローバルノルムクリップを含む）はGPU上で動きます。`SequenceLossGrad`はCPUの`SequenceLoss`（行ごとのメディアン中心化、clampしたtarget、Smooth L1、delta項）を行ごとに1ブロックで再現し、`LossGrad`と照合します。GPUの学習ループで合成データの損失が下がります。

## 検証

参照エンジンはPyTorchと照合します。PyTorchは比較の基準です。

- forward、損失、backward、勾配クリップ、AdamW 1ステップがPyTorchのfixture（`testdata/tcn_step.json`）と一致します。float64のforwardと勾配で約1e-16です。
- メディアンは`torch.median`に従います。要素数が偶数のときは下側の中央値をとり、メディアンと等しい要素へ勾配を等分します。
- 勾配はPyTorchとは別に、中心差分でも確認します。
- 汎用のfloat32モジュールとマスク付き損失はCPUの差分チェックを持ちます。`tools/gen_nn_fixture.py`が生成する`testdata/nn_conv_step.json`で、forward、マスク付き損失、backward、AdamW 1更新をCPUと、CUDAが使える環境ではCUDAで照合します。
- `tools/gen_acoustic_fixture.py`が生成する`testdata/acoustic_step.json`で、Acousticの出力、損失、全パラメータ勾配、クリップノルム、AdamW 1ステップを両デバイスで照合します。
- `tools/gen_transformer_fixture.py`はテスト中に一時ディレクトリへPyTorchの値を生成します。softmax、log-softmax、cross-entropy、dropout、layer norm、ブロードキャストのバッチmatmul、アテンション、パディング付きembedding、2層のtransformerスタックを出力と勾配の両方で両デバイスで照合し、スタックではAdamW 1更新も確認します。CUDAのテストはカーネルの起動を確認します。PyTorchが作ったsafetensorsファイルをGoで読み込み、PyTorchへ戻して往復します。CUDAグラフのテストは、optimizerの重み更新を含めて2回のAcoustic再生とeagerステップを比較します。
- レジストリの棚卸しテストは、公開する微分可能な演算ごとにレジストリ項目とdispatch呼び出しを確認します。dispatchを1つ削除するとテストが失敗することも確認済みです。tiledアテンションは、系列長256、1024、4096を含めてマスク付きの出力と勾配を、実体化したCUDA経路と比較します。CUDAのresidencyテストはデバイスカーネルの起動を確認し、BF16のグラフテストは記録した学習ステップを確認します。
- `testdata/conv2d.json`（`tools/gen_conv2d_fixture.py`が生成）は、groupedとdepthwise multiplier、矩形・偶数のカーネル、stride、dilation、biasなし、最後の部分タイルを含めて、Conv2dの出力と全input/weight/bias勾配をCPU/CUDAで確認します。別のテストで数値勾配、非連続のview、パラメータ共有による加算、凍結パラメータを扱います。純GoのCNN例は空間パターンを学習し、両バックエンドでsafetensorsの再読込を確認します。
- `testdata/pool2d.json`（`tools/gen_pool2d_fixture.py`が生成）はMaxPool2d、AvgPool2d、AdaptiveAvgPool2dの出力と勾配をCPU/CUDAで確認します。ceil modeの境界、パディングの除数、dilation、重複する窓、view、履歴を保持したグラフ、CUDAの再生を扱います。`cmd/cnn-train`は畳み込みにmax poolingとadaptive global average poolingを組み合わせます。
- `testdata/batchnorm.json`（`tools/gen_batchnorm_fixture.py`が生成）はBatchNormの出力とVJPを17ケース、running stateを3つの複数ステップ列で確認します。固定momentumと累積平均、trackingを無効にした評価、NoGrad学習、stateの再読込、viewと保持した履歴、CUDAの再生を扱います。CNN例はBatchNormを含み、state再読込後の評価を確認します。

## コマンド

```
go test ./...
go vet ./...
go run ./cmd/tcn-train
go run ./cmd/cnn-train -device cpu -steps 60 -out out/cnn.safetensors
go run ./cmd/cnn-train -device cuda -steps 60 -out out/cnn-cuda.safetensors
go run ./cmd/mnist-train -download -device cpu -steps 100 -train-limit 2000 -valid-limit 1000
go run ./cmd/mnist-train -device cuda -steps 100 -train-limit 2000 -valid-limit 1000 -out out/mnist-cuda.safetensors
go run ./cmd/mnist-train -device cpu -workers 4 -prefetch 3 -steps 100 -train-limit 2000 -valid-limit 1000 -out out/mnist-prefetch.safetensors
go run ./cmd/mnist-train -device cpu -batch-size 16 -accum 4 -steps 100 -train-limit 2000 -valid-limit 1000 -out out/mnist-accum.safetensors
go run ./cmd/mnist-train -device cpu -optimizer sgd -momentum .9 -nesterov -steps 100 -train-limit 2000 -valid-limit 1000 -out out/mnist-sgd.safetensors
go run ./cmd/mnist-train -device cpu -optimizer sgd -momentum .9 -nesterov -no-decay-suffix bias -steps 100 -train-limit 2000 -valid-limit 1000 -out out/mnist-sgd-groups.safetensors
go run ./cmd/tabular-train -generate -device cpu -steps 400 -out out/tabular.safetensors
go run ./cmd/tabular-train -data out/tabular.csv -device cuda -steps 400 -out out/tabular-cuda.safetensors
go run ./cmd/audio-train -generate -device cpu -steps 300 -out out/denoise.safetensors
go run ./cmd/audio-train -wav out/tone.wav -device cuda -steps 300 -out out/denoise-cuda.safetensors
go run ./cmd/gputcn-train
go run ./cmd/nn-conv-train -device cpu -hidden 128 -kernel 5 -steps 3
go run ./cmd/nn-conv-train -device cpu -hidden 128 -kernel 5 -steps 5 -cpuprofile "$env:TEMP/gograd-nn-cpu.pprof"
go test ./tensor -run '^$' -bench 'Benchmark(SGEMM|NaiveSGEMM)$' -benchtime=1x -cpu=12
go test ./gputcn/ -run XXX -bench .
go run ./cmd/acoustic-bench -device cpu
go run ./cmd/acoustic-bench -device cuda -warmup 1 -gpu-profile
go run ./cmd/acoustic-bench -device cuda -warmup 2 -graph
go run ./cmd/acoustic-bench -device cuda -warmup 2 -memory-stats
python tools/bench_acoustic.py --device cuda --threads 12
python tools/bench_acoustic.py --device cuda --threads 12 --graph
go run ./cmd/transformer-bench -device cuda -warmup 1
go run ./cmd/transformer-bench -device cuda -warmup 2 -graph -bf16
python tools/bench_transformer.py --device cuda --threads 12 --warmup 1
python tools/bench_transformer.py --device cuda --warmup 2 --graph --bf16
go run ./cmd/attention-bench -seq 4096 -algorithm materialized
go run ./cmd/attention-bench -seq 4096 -algorithm flash
```

`tcn-train`は小さな合成データをCPUで学習し、`out/synthetic-tcn.json`を書き出します。`gputcn-train`は同じ内容をGPUのforwardとbackwardで行い、段階ごとの時間を表示して`out/gputcn-synthetic.json`を書き出します。`-graph`を付けると`cuda.Capture`でステップ全体を記録して再生し、`-profile N`はCUDA eventで段階ごとのGPU時間を表示します。どちらもエンジンの動作確認用で、実際の学習手順ではありません。

`nn-conv-train`はパディング付きの合成系列（入力74、出力72、dilation 1,2,4,1,2,4）で6ブロックのresidual Conv1dネットワークを学習し、学習ステップ全体の時間を表示します。CUDAがある環境では`-device cuda`を受け付けます。既定の損失はマスク付きL1で、`-loss mse`でマスク付きMSEを選びます。12スレッドのベンチマークでは`GOMAXPROCS=12`を設定し、`-cpuprofile`でGoのCPUプロファイルを書き出します。

RTX 3060 Tiと12 CPUスレッド（Goは`GOMAXPROCS=12`、PyTorchは`torch.set_num_threads(12)`）で、温めた学習ステップ全体を3回、対で測った中央値です。

| モデルと実行環境 | CPU FP32 | CUDA FP32 eager | CUDA FP32 graph | CUDA BF16 eager | CUDA BF16 graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acoustic, PyTorch | 2,506 ms | 58.5 ms | 53.7 ms | 47.1 ms | 42.6 ms |
| Acoustic, gograd | 3,876 ms | 82.3 ms | 82.0 ms | 63.8 ms | 64.1 ms |
| Transformer, PyTorch | 1,292 ms | 25.9 ms | 23.9 ms | 13.5 ms | 12.0 ms |
| Transformer, gograd | 2,276 ms | 27.2 ms | 27.8 ms | 22.4 ms | 23.2 ms |

Acousticはhidden 384、batch 24、400フレーム、10ブロックです。Transformerはpre-normの4層で、モデル幅256、4ヘッド、feed-forward幅1024、batch 16、系列長256です。ステップにはforward、backward、グローバルノルムのクリップ、AdamWを含みます。Acousticのgraph再生はスカラー損失の読み戻しを含み、デバイス上のステップカウンタがlaunchごとにAdamWのbias correctionを更新します。CPUの計測値は測定中に大きく変動したため、これらの中央値は1ステップより有用ですが、安定したスループットとして扱うべきではありません。BF16の値はPyTorchのautocastと、実際の値をFP32で保持するgogradのBF16シャドウ経路を使っていますが、対象範囲はまだ異なります。実装は異なるseedの乱数を使うため、比較できるのはアーキテクチャと次元であり、損失値そのものではありません。

TransformerのCPUはPyTorchの1.76倍で、2.5倍の目標を満たします。AcousticのCPUは1.55倍で、1.5倍の目標にはわずかに届きません。TransformerのCUDA FP32 eagerはPyTorchの1.05倍です。BF16 eagerはtransformerでPyTorchの1.66倍、Acousticで1.35倍です。graph再生はgogradではeagerと同程度で、デバイス側の処理がlaunchのコストを上回っています。プロファイリングは同期を加えるため、表と比較しないでください。

別のアテンションのforward/backwardベンチマーク（batch 1、4ヘッド、head次元64、3回）のCUDA event中央値です。

| 系列長 | 実体化ステップ | tiledステップ | 実体化のピーク増分 | tiledのピーク増分 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.366 ms | 0.766 ms | 3.50 MiB | 1.01 MiB |
| 1024 | 1.601 ms | 9.606 ms | 50.00 MiB | 4.03 MiB |
| 4096 | 18.916 ms | 127.889 ms | 776.00 MiB | 16.12 MiB |

ピーク増分は、損失と勾配のバッファを含めて、入力の基準値から増えた使用中CUDAバッファの最大値です。tiledアテンションは速度を犠牲にして、長い系列でscratchの使用量を大きく下げます。出力と勾配は3つの系列長すべてで実体化経路と一致し、加算的なブロードキャストマスクとその勾配は別のテストで扱います。

### 可変長系列のCUDAメモリ

64ステップ、batch 8でフレーム長が変わる学習を再現したところ、使用中の割当バイト数はほぼ一定でしたが、旧来の正確なサイズごとのプールは6.15 GBを保持しました。サイズクラスを持つ上限付きのプールは、同じ処理のキャッシュを0.76 GBに減らしました（以下は10進のバイト数）。200ステップのテストは既定の上限を確認し、256 MiBの上限でも学習を繰り返します。

| 64ステップ後のアロケータ | 使用中バイト | キャッシュバイト | 予約バイト |
| --- | ---: | ---: | ---: |
| 旧 | 10,927,812 | 6,154,440,208 | 6,165,368,020 |
| 現行 | 10,949,888 | 762,475,264 | 773,425,152 |

固定形状のAcousticのCUDAステップは、最後の3対で前が77.3 ms、後が77.8 msでした。それより前の対は、デバイスの負荷が変わってどちらのバイナリも遅くなりました。記録したグラフの割当は`Graph.Close`まで予約されたままで、グラフが生きている間はキャッシュの上限から除きます。

`gputcn-fit`は合成データの代わりに、用意したデータセット（フレームの特徴量、target、mask）で学習します。

```
go run ./cmd/gputcn-fit -write-synthetic out/prepared.json
go run ./cmd/gputcn-fit -dataset out/prepared.json -epochs 30
```

データセットは`feature_names`と、疎な特徴量を`rows`、`cols`、`vals`で持つレコード、`targets`（cent）、`mask`を持つJSONです。レコードはidのハッシュで学習と検証に分かれます。

PyTorchのfixtureを作り直すには`torch`が必要です。

```
python tools/gen_fixture.py
python tools/gen_nn_fixture.py
python tools/gen_losses_fixture.py
```

`cuda`のテストは、CUDAランタイムとcuBLASのDLLを読み込めるときだけ実行します。WindowsではCUDAの`bin\x64`ディレクトリを`PATH`に通します。例を示します。

```
$env:PATH = "$env:CUDA_PATH\bin\x64;$env:PATH"
go test ./cuda/
```

## ディレクトリ構成

- `autograd/layers.go`：再利用できるモジュール
- `docs/API.md`：公開APIと現状の制約
- `autograd/cpu_binary.go`、`autograd/cuda_flash_attention.go`、`autograd/cuda_autocast.go`：最適化したCPU勾配、tiledアテンション、キャッシュしたBF16デバイスバッファ
- `cmd/attention-bench`：CUDAアテンションの時間と使用中バッファの比較
- `autograd/views.go`、`autograd/registry.go`、`autograd/transformer*.go`、`autograd/safetensors.go`：view、要素ごとのdispatch、transformerの演算とモジュール、checkpoint
- `cmd/transformer-bench`、`tools/bench_transformer.py`：4層encoderの時間
- `tools/gen_transformer_fixture.py`、`tools/safetensors_fixture.py`：PyTorchとの照合に使う参照
- `tensor.go`、`ops.go`、`conv.go`、`loss.go`：自動微分エンジン
- `model.go`：`FrameIntonationTCN`
- `optim.go`：AdamWと勾配クリップ
- `export.go`：ランタイム用JSONの形式
- `cuda/`：CUDAランタイム、cuBLAS、NVRTCのバインディング、stream、グラフ記録
- `kernels/`：`cuda`の上に載る独自GPUカーネル（活性化、畳み込み、損失、AdamW）
- `dsp/`：FFT、Hann窓、STFT/iSTFT、リサンプリング。torchとtorchaudioに対応します
- `torchrng/`：float32とfloat64の`torch.Generator(device="cpu")`をbit単位で再現
- `tensor/`：形状を扱うfloat32のCPU/CUDAテンソルと、ブロック化したCPU SGEMM
- `autograd/`：define-by-runグラフ、演算、Acousticモデル
- `data/`、`cmd/mnist-train`：任意の上限付きホストprefetchを持つ汎用バッチ処理、再開できるサンプリング、raw/gzip IDXの読み込み、Pythonなしの実データ学習と再開の例
- `autograd/conv2d*.go`、`autograd/cuda_conv2d.go`、`cmd/cnn-train`：tiledなCPU/CUDA Conv2dとbackward、名前付き層、PythonなしのCNN学習例
- `nn/`：CPU/CUDAのモジュール、マスク付き損失、AdamW、JSONのcheckpoint
- `gputcn/`：GPU float32のTCN forward、backward、AdamW
- `cmd/tcn-train`：CPUの合成データ学習例
- `cmd/gputcn-train`：GPUの合成データ学習例。段階ごとの時間を表示します
- `cmd/gputcn-fit`：用意したデータセットでの学習
- `cmd/nn-conv-train`：residual Conv1dのfloat32合成データ学習例
- `tools/gen_fixture.py`、`tools/gen_nn_fixture.py`：PyTorchのfixture生成
- `tools/gen_acoustic_fixture.py`、`tools/bench_acoustic.py`：Acousticの照合と時間計測
- `docs/DESIGN.md`：設計の説明
- `testdata/`：リポジトリに含めたfixture

## モデルのJSON形式

`nn.Save(writer, model)`はバージョン付きのJSONオブジェクトを書き出し、`nn.Load(reader, model)`は読み込む前に層の構成、パラメータ名、形状を確認します。`layers`はforwardの順に並んだリストです。各項目は`name`、`type`（`linear`、`conv1d`、`activation`、`residual`のいずれか）、`in_channels`、`out_channels`を持ち、必要に応じて`kernel`、`dilation`、`activation`、`slope`を持ちます。residualの`layers`リストは内側の系列を表します。`parameters`は`{name,shape,values}`の順序付き配列で、row-majorのfloat32値を持ちます。Linearの重みは`[out,in]`、biasは`[out]`です。Conv1dの重みはcross-correlationの順で`[out,in,kernel]`、biasは`[out]`です。フレームは`[batch,time,channels]`で、ゼロsame-paddingは両側へ`dilation*(kernel-1)/2`だけ広げます。名前は`model.1.inner.0.weight`のような安定したパスです。optimizerの状態は含みません。

## 現状の制約

- グラフは多くのfloat32の連続結果を実体化します。BF16は任意で、実際の値はFP32のまま保持します。FP16、loss scaling、dtypeとlayoutをキーにしたカーネル登録はありません。
- 特徴量キャッシュの読み込みとwindowのサンプリングはホストで行います。`data`パッケージは同期または並列のホスト読込と上限付きprefetchに対応し、CUDAテンソルの生成は読み込む側のスレッドに留まります。
- レジストリはCPUとCUDAの実装を選びますが、実行時にカーネルを追加する仕組みはありません。
- 互換用のNoGradスコープとBF16・アテンションの設定はプロセス共有です。明示的な実行コンテキストは記録・精度・アテンション選択を分離します。CUDAのdevice、暗黙のstream、capture状態はOSスレッド単位で、プールはdeviceごとに分けています。stream単位のプール分離と、1回の記録の中での複数streamの重ね合わせには対応していません。
- tiledアテンションはhead次元128までで、計測した系列長では実体化したGEMM経路より遅くなります。GPUで推論する場合は、不要になったグラフを`ReleaseGraph`で解放してください。
- 旧`nn`経路はモジュールごとの明示的なbackwardを使います。

## ライセンス

MITです。`LICENSE`を参照してください。
