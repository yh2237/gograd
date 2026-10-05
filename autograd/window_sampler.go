package autograd

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
)

// Window identifies a sampled sequence and a frame offset. Short sequences are
// returned from offset zero; the caller pads the remaining frames.
type Window struct {
	Index, Start int
	DropContext  bool
}

// WindowSamplerState includes both the data geometry and the PCG stream.
type WindowSamplerState struct {
	Frames          []int
	WindowSize      int
	BatchSize       int
	DropProbability float64
	Random          []byte
}

type WindowSampler struct {
	frames          []int
	windowSize      int
	batchSize       int
	dropProbability float64
	source          *rand.PCG
	random          *rand.Rand
}

func NewWindowSampler(frames []int, windowSize, batchSize int, dropProbability float64, seed uint64) (*WindowSampler, error) {
	if len(frames) == 0 || windowSize < 1 || batchSize < 1 || math.IsNaN(dropProbability) || dropProbability < 0 || dropProbability > 1 {
		return nil, fmt.Errorf("autograd: invalid window sampler configuration")
	}
	for _, n := range frames {
		if n < 1 {
			return nil, fmt.Errorf("autograd: window sampler sequences must contain frames")
		}
	}
	source := rand.NewPCG(seed, seed^0x9e3779b97f4a7c15)
	return &WindowSampler{slices.Clone(frames), windowSize, batchSize, dropProbability, source, rand.New(source)}, nil
}

func (s *WindowSampler) Batch() []Window {
	windows := make([]Window, s.batchSize)
	for i := range windows {
		index := s.random.IntN(len(s.frames))
		// Include the final complete window, unlike the old trainer's exclusive
		// upper bound. Padding handles sequences shorter than windowSize.
		start := s.random.IntN(max(1, s.frames[index]-s.windowSize+1))
		drop := s.dropProbability > 0 && s.random.Float64() < s.dropProbability
		windows[i] = Window{index, start, drop}
	}
	return windows
}

func (s *WindowSampler) State() WindowSamplerState {
	data, _ := s.source.MarshalBinary()
	return WindowSamplerState{slices.Clone(s.frames), s.windowSize, s.batchSize, s.dropProbability, data}
}

// LoadState rejects a different dataset geometry before replacing the stream.
func (s *WindowSampler) LoadState(state WindowSamplerState) error {
	if !slices.Equal(state.Frames, s.frames) || state.WindowSize != s.windowSize || state.BatchSize != s.batchSize || state.DropProbability != s.dropProbability {
		return fmt.Errorf("autograd: window sampler state does not match configuration")
	}
	source := rand.NewPCG(0, 0)
	if err := source.UnmarshalBinary(state.Random); err != nil {
		return fmt.Errorf("autograd: invalid window sampler random state: %w", err)
	}
	s.source, s.random = source, rand.New(source)
	return nil
}
