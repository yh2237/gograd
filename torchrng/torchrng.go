// Package torchrng reproduces the random streams of PyTorch's CPU generator.
//
// Porting a trained model to Go usually means its inference path has to
// consume the same random numbers as the reference, or parity fixtures cannot
// be generated without Python. This package implements the generator that
// backs torch.Generator(device="cpu"): Mersenne Twister MT19937 with the
// state initialization and tempering of at::mt19937, the 24-bit uniform
// transform, and the vectorized Box-Muller transform that
// torch.randn(float32) uses for contiguous tensors of sixteen elements or
// more. The float32 transcendental functions are the cephes polynomial
// approximations of PyTorch's AVX2 kernels, not libm, because the two differ
// by one ulp and that difference is visible downstream.
//
// The streams are bit-exact for the same seed, element count and dtype. A
// state captured from Python is not byte-compatible: use State and SetState
// to move a generator inside Go.
//
// Bit-exactness also depends on how the target compiles floating point. Go
// permits an implementation to contract a multiply and an add into one fused
// operation, and on arm64 the compiler does that by default while amd64's
// baseline does not. The reference's amd64 kernels use separate multiply and
// add, so every product-then-sum in the cephes chains and the uniform
// transforms goes through mulAdd32 or mulAdd64 instead of a bare expression.
// That is why the code reads the way it does, and it is worth keeping if this
// file is ever simplified.
package torchrng

import "math"

const (
	stateN = 624
	stateM = 397

	matrixA   = 0x9908b0df
	upperMask = 0x80000000
	lowerMask = 0x7fffffff
)

// Generator is one reproducible PyTorch CPU random stream. It is not safe for
// concurrent use, matching the reference's generator-per-thread model.
type Generator struct {
	state [stateN]uint32
	left  int
	next  int
	seed  uint64

	// cachedFloat32 and cachedFloat64 hold the pending second Box-Muller
	// sample, used only by the serial path for tensors shorter than sixteen
	// elements.
	cachedFloat32 bool
	nextFloat32   float32
	cachedFloat64 bool
	nextFloat64   float64
}

// New returns a generator seeded like torch.Generator().manual_seed(seed).
// Only the low 32 bits of seed initialize the state array, as in the
// reference; the full seed is retained for Seed.
func New(seed uint64) *Generator {
	g := &Generator{seed: seed, left: 1}
	g.state[0] = uint32(seed)
	for j := 1; j < stateN; j++ {
		g.state[j] = 1812433253*(g.state[j-1]^(g.state[j-1]>>30)) + uint32(j)
	}
	return g
}

// Seed reports the seed this generator was constructed with.
func (g *Generator) Seed() uint64 { return g.seed }

func (g *Generator) nextState() {
	g.left = stateN
	g.next = 0
	i := 0
	for j := stateN - stateM + 1; j > 1; j-- {
		g.state[i] = g.state[i+stateM] ^ twist(g.state[i], g.state[i+1])
		i++
	}
	for j := stateM; j > 1; j-- {
		g.state[i] = g.state[i+stateM-stateN] ^ twist(g.state[i], g.state[i+1])
		i++
	}
	g.state[i] = g.state[i+stateM-stateN] ^ twist(g.state[i], g.state[0])
}

func twist(u, v uint32) uint32 {
	mixed := (u & upperMask) | (v & lowerMask)
	out := mixed >> 1
	if v&1 != 0 {
		out ^= matrixA
	}
	return out
}

// Uint32 returns the next tempered engine word, matching torch.Generator's
// random(). It is exposed so callers can build other distributions on the same
// stream.
func (g *Generator) Uint32() uint32 {
	g.left--
	if g.left == 0 {
		g.nextState()
	}
	y := g.state[g.next]
	g.next++
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18
	return y
}

// Uint64 returns the next 64-bit word, matching torch.Generator's random64().
func (g *Generator) Uint64() uint64 {
	return uint64(g.Uint32())<<32 | uint64(g.Uint32())
}

// State is a snapshot of a generator. It round-trips through SetState so a
// stream can be saved and resumed, but it is not the byte layout of
// torch.get_rng_state.
type State struct {
	State         [stateN]uint32
	Left, Next    int
	Seed          uint64
	CachedFloat32 bool
	NextFloat32   float32
	CachedFloat64 bool
	NextFloat64   float64
}

// State returns a snapshot of the generator.
func (g *Generator) State() State {
	return State{State: g.state, Left: g.left, Next: g.next, Seed: g.seed,
		CachedFloat32: g.cachedFloat32, NextFloat32: g.nextFloat32,
		CachedFloat64: g.cachedFloat64, NextFloat64: g.nextFloat64}
}

// SetState restores a snapshot produced by State. The receiver's generator
// position is replaced entirely.
func (g *Generator) SetState(s State) {
	g.state = s.State
	g.left, g.next, g.seed = s.Left, s.Next, s.Seed
	g.cachedFloat32, g.nextFloat32 = s.CachedFloat32, s.NextFloat32
	g.cachedFloat64, g.nextFloat64 = s.CachedFloat64, s.NextFloat64
}

// Uniform fills dst with samples of torch.rand(*dst.shape), drawing from the
// low 24 bits of each engine word. Values lie in [from, to).
func (g *Generator) Uniform(dst []float32, from, to float32) {
	for i := range dst {
		dst[i] = uniform32(g.Uint32(), from, to)
	}
}

// Uniform64 fills dst with samples of torch.rand(dtype=torch.float64), which
// consumes a full 64-bit engine word per element and uses 53 fraction bits.
func (g *Generator) Uniform64(dst []float64, from, to float64) {
	for i := range dst {
		dst[i] = uniform64(g.Uint64(), from, to)
	}
}

func uniform32(v uint32, from, to float32) float32 {
	const mask = 0x00ffffff
	const divisor = float32(1) / float32(uint32(1)<<24)
	x := float32(v&mask) * divisor
	// The reference writes this as one product followed by one sum.
	return mulAdd32(x, to-from, from)
}

// product64 exists to keep a float64 multiply and its following add as two
// roundings. Go may contract the pair into a fused operation on some
// architectures, as mulAdd32 documents, and float64 has no wider type to widen
// into, so the product goes through a call the compiler cannot fold.
//
//go:noinline
func product64(a, b float64) float64 { return a * b }

// mulAdd64 evaluates a*b+c with the separate roundings of a float64 multiply
// and add, matching the reference's serial Box-Muller.
func mulAdd64(a, b, c float64) float64 { return product64(a, b) + c }

func uniform64(v uint64, from, to float64) float64 {
	const mask = (uint64(1) << 53) - 1
	const divisor = 1.0 / float64(uint64(1)<<53)
	x := float64(v&mask) * divisor
	return mulAdd64(x, to-from, from)
}

// Normal fills dst with torch.randn(*dst.shape, mean=mean, std=std) for a
// contiguous float32 tensor, which is the layout every exported checkpoint
// uses. Tensors of sixteen elements or more consume one engine word per
// element and then apply Box-Muller to blocks of sixteen, recomputing a short
// final block from sixteen further words. Shorter tensors take the reference's
// serial path, which runs in float64 and casts each pair down.
func (g *Generator) Normal(dst []float32, mean, std float32) {
	if len(dst) >= 16 {
		g.uniform32Into(dst)
		for i := 0; i+16 <= len(dst); i += 16 {
			normalFill16Vector(dst[i:i+16], mean, std)
		}
		if len(dst)%16 != 0 {
			var tail [16]float32
			g.uniform32Into(tail[:])
			normalFill16Vector(tail[:], mean, std)
			copy(dst[len(dst)-16:], tail[:])
		}
		return
	}
	for i := range dst {
		dst[i] = float32(g.normalSerial64(float64(mean), float64(std)))
	}
}

// Normal64 fills dst with torch.randn(*dst.shape, dtype=torch.float64) for a
// contiguous tensor. Blocks of sixteen are transformed in place, with a short
// final block recomputed from sixteen further 64-bit words; shorter tensors
// use the serial pairing path.
func (g *Generator) Normal64(dst []float64, mean, std float64) {
	if len(dst) >= 16 {
		g.uniform64Into(dst)
		for i := 0; i+16 <= len(dst); i += 16 {
			normalFill16Scalar(dst[i:i+16], mean, std)
		}
		if len(dst)%16 != 0 {
			var tail [16]float64
			g.uniform64Into(tail[:])
			normalFill16Scalar(tail[:], mean, std)
			copy(dst[len(dst)-16:], tail[:])
		}
		return
	}
	for i := range dst {
		dst[i] = g.normalSerial64(mean, std)
	}
}

func (g *Generator) uniform32Into(dst []float32) {
	for i := range dst {
		dst[i] = uniform32(g.Uint32(), 0, 1)
	}
}

func (g *Generator) uniform64Into(dst []float64) {
	for i := range dst {
		dst[i] = uniform64(g.Uint64(), 0, 1)
	}
}

func (g *Generator) normalSerial64(mean, std float64) float64 {
	if g.cachedFloat64 {
		g.cachedFloat64 = false
		return mulAdd64(g.nextFloat64, std, mean)
	}
	u1 := uniform64(g.Uint64(), 0, 1)
	u2 := uniform64(g.Uint64(), 0, 1)
	r := math.Sqrt(-2 * math.Log1p(-u2))
	theta := 2 * math.Pi * u1
	sin, cos := math.Sincos(theta)
	g.nextFloat64 = r * sin
	g.cachedFloat64 = true
	return mulAdd64(r*cos, std, mean)
}
