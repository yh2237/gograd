package autograd

// IndexLoaderは決定的なシャッフル順でミニバッチの添字を列挙する。
// エポックごとに順列が変わり、同じseedなら再現できる。
type IndexLoader struct {
	Size      int
	BatchSize int
	seed      uint64
	epoch     uint64
}

func NewIndexLoader(size, batchSize int, seed uint64) *IndexLoader {
	if batchSize < 1 {
		batchSize = 1
	}
	return &IndexLoader{Size: size, BatchSize: batchSize, seed: seed}
}

// Epochは1エポック分のバッチを返し、次のエポックへ進む。最後のバッチは小さいことがある。
func (l *IndexLoader) Epoch() [][]int {
	order := make([]int, l.Size)
	for i := range order {
		order[i] = i
	}
	state := l.seed + l.epoch*0x9E3779B97F4A7C15
	for i := l.Size - 1; i > 0; i-- {
		j := int(nextRandom(&state) % uint64(i+1))
		order[i], order[j] = order[j], order[i]
	}
	l.epoch++
	batches := make([][]int, 0, (len(order)+l.BatchSize-1)/l.BatchSize)
	for start := 0; start < len(order); start += l.BatchSize {
		end := start + l.BatchSize
		if end > len(order) {
			end = len(order)
		}
		batches = append(batches, order[start:end])
	}
	return batches
}

func nextRandom(state *uint64) uint64 {
	*state += 0x9E3779B97F4A7C15
	z := *state
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}
