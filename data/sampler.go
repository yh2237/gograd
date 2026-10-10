package data

import (
	"encoding/json"
	"fmt"
	"io"
	"math/rand/v2"
	"slices"
)

// Sampler separates index order from loading/collation. Peek must not advance
// state. Loader advances only after a batch succeeds (or drops the final tail).
// Peek accepts arbitrary positive lookahead counts (clamped at epoch end), and
// overlapping peeks must describe the same order until Advance/NextEpoch.
// Methods are serial: do not share a mutable sampler between independent loaders.
type Sampler interface {
	Len() int
	Peek(int) ([]int, error)
	Advance(int) error
	NextEpoch() error
}

// StatefulSampler supports Loader checkpoints. Restore must validate fully
// before mutating state. The payload is opaque to Loader.
type StatefulSampler interface {
	Sampler
	MarshalBinary() ([]byte, error)
	UnmarshalBinary([]byte) error
}

type SamplerOptions struct {
	Shuffle bool
	Seed    uint64
}
type IndexSampler struct {
	size, position int
	options        SamplerOptions
	epoch          uint64
	order          []int
	source         *rand.PCG
}
type IndexSamplerState struct {
	Version, Size, Position int
	Shuffle                 bool
	Seed, Epoch             uint64
	Order                   []int
	Random                  []byte
}

func NewIndexSampler(size int, options SamplerOptions) (*IndexSampler, error) {
	if size < 0 {
		return nil, fmt.Errorf("data: negative sampler size")
	}
	s := &IndexSampler{size: size, options: options, source: rand.NewPCG(options.Seed, options.Seed^0x9e3779b97f4a7c15)}
	s.resetOrder()
	return s, nil
}
func (s *IndexSampler) resetOrder() {
	s.order = make([]int, s.size)
	for i := range s.order {
		s.order[i] = i
	}
	if s.options.Shuffle {
		rand.New(s.source).Shuffle(s.size, func(i, j int) { s.order[i], s.order[j] = s.order[j], s.order[i] })
	}
	s.position = 0
}
func (s *IndexSampler) Len() int { return s.size }
func (s *IndexSampler) Peek(count int) ([]int, error) {
	if count < 1 {
		return nil, fmt.Errorf("data: positive sample count required")
	}
	if s.position == s.size {
		return nil, io.EOF
	}
	end := s.position + min(count, s.size-s.position)
	return slices.Clone(s.order[s.position:end]), nil
}
func (s *IndexSampler) Advance(count int) error {
	if count < 0 || count > s.size-s.position {
		return fmt.Errorf("data: invalid sampler advance")
	}
	s.position += count
	return nil
}

// NextEpoch starts a new permutation and discards any unconsumed current tail.
func (s *IndexSampler) NextEpoch() error {
	if s.epoch == ^uint64(0) {
		return fmt.Errorf("data: sampler epoch overflow")
	}
	s.epoch++
	s.resetOrder()
	return nil
}
func (s *IndexSampler) State() IndexSamplerState {
	random, _ := s.source.MarshalBinary()
	return IndexSamplerState{1, s.size, s.position, s.options.Shuffle, s.options.Seed, s.epoch, slices.Clone(s.order), random}
}
func (s *IndexSampler) LoadState(state IndexSamplerState) error {
	if state.Version != 1 || state.Size != s.size || state.Shuffle != s.options.Shuffle || state.Seed != s.options.Seed || state.Position < 0 || state.Position > s.size || len(state.Order) != s.size {
		return fmt.Errorf("data: sampler state does not match configuration")
	}
	seen := make([]bool, s.size)
	for i, v := range state.Order {
		if v < 0 || v >= s.size || seen[v] || !s.options.Shuffle && v != i {
			return fmt.Errorf("data: invalid sampler permutation")
		}
		seen[v] = true
	}
	source := rand.NewPCG(0, 0)
	if err := source.UnmarshalBinary(state.Random); err != nil {
		return fmt.Errorf("data: invalid sampler random state: %w", err)
	}
	s.source, s.order, s.position, s.epoch = source, slices.Clone(state.Order), state.Position, state.Epoch
	return nil
}
func (s *IndexSampler) MarshalBinary() ([]byte, error) { return json.Marshal(s.State()) }
func (s *IndexSampler) UnmarshalBinary(payload []byte) error {
	var state IndexSamplerState
	if err := json.Unmarshal(payload, &state); err != nil {
		return fmt.Errorf("data: invalid sampler payload: %w", err)
	}
	return s.LoadState(state)
}
