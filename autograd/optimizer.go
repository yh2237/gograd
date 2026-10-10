package autograd

import (
	"encoding/json"
	"fmt"
	"math"
)

// Optimizer is the common eager training surface. Step mutates parameters;
// ZeroGrad resets gradients, and Close releases optimizer-owned state only.
type Optimizer interface {
	Step()
	ZeroGrad()
	LearningRate() float32
	SetLearningRate(float32)
	Close()
}

// CheckpointOptimizer adds typed snapshots. ValidateSnapshot must reject bad
// input without mutation; LoadSnapshot validates before copying state.
type CheckpointOptimizer interface {
	Optimizer
	Snapshot() (OptimizerSnapshot, error)
	ValidateSnapshot(OptimizerSnapshot) error
	LoadSnapshot(OptimizerSnapshot) error
}
type OptimizerSnapshot struct {
	Version int
	Kind    string
	AdamW   *OptimizerState
	SGD     *SGDState
}

func validateLearningRate(lr float32) error {
	if lr < 0 || math.IsNaN(float64(lr)) || math.IsInf(float64(lr), 0) {
		return fmt.Errorf("autograd: invalid learning rate")
	}
	return nil
}
func (o *AdamW) LearningRate() float32 { return o.LR }
func (o *AdamW) SetLearningRate(lr float32) {
	if err := validateLearningRate(lr); err != nil {
		panic(err)
	}
	o.LR = lr
}
func (o *AdamW) Snapshot() (OptimizerSnapshot, error) {
	if o == nil {
		return OptimizerSnapshot{}, fmt.Errorf("autograd: nil AdamW")
	}
	state, err := o.State()
	if err != nil {
		return OptimizerSnapshot{}, err
	}
	return OptimizerSnapshot{Version: 1, Kind: "adamw", AdamW: &state}, nil
}
func (o *AdamW) ValidateSnapshot(s OptimizerSnapshot) error {
	if o == nil {
		return fmt.Errorf("autograd: nil AdamW")
	}
	if s.Version != 1 || s.Kind != "adamw" || s.AdamW == nil || s.SGD != nil {
		return fmt.Errorf("autograd: incompatible AdamW snapshot")
	}
	return o.validateState(*s.AdamW)
}
func (o *AdamW) LoadSnapshot(s OptimizerSnapshot) error {
	if err := o.ValidateSnapshot(s); err != nil {
		return err
	}
	return o.LoadState(*s.AdamW)
}

var _ CheckpointOptimizer = (*AdamW)(nil)
var _ CheckpointOptimizer = (*SGD)(nil)

func snapshotStep(s OptimizerSnapshot) int {
	if s.Kind == "adamw" && s.AdamW != nil {
		return s.AdamW.StepCount
	}
	if s.Kind == "sgd" && s.SGD != nil {
		return s.SGD.StepCount
	}
	return -1
}
func snapshotEntries(s OptimizerSnapshot) (map[string]safeTensorEntry, map[string]string, error) {
	if s.Kind == "adamw" && s.AdamW != nil {
		return optimizerEntries(*s.AdamW)
	}
	if s.Kind == "sgd" && s.SGD != nil {
		return sgdEntries(*s.SGD)
	}
	return nil, nil, fmt.Errorf("autograd: unknown optimizer snapshot")
}
func snapshotFromEntries(entries map[string]safeTensorEntry, metadata map[string]string) (OptimizerSnapshot, error) {
	kind := metadata["optimizer_kind"]
	if kind == "" {
		kind = "adamw"
	} // legacy training-1
	if kind == "sgd" {
		metadata["format"] = sgdStateFormat
		s, err := sgdStateFromEntries(entries, metadata)
		if err != nil {
			return OptimizerSnapshot{}, err
		}
		return OptimizerSnapshot{Version: 1, Kind: kind, SGD: &s}, nil
	}
	if kind != "adamw" {
		return OptimizerSnapshot{}, fmt.Errorf("autograd: unknown optimizer kind %q", kind)
	}
	var config OptimizerState
	if err := json.Unmarshal([]byte(metadata["optimizer"]), &config); err != nil {
		return OptimizerSnapshot{}, err
	}
	metadata["format"] = optimizerStateFormat
	s, err := optimizerStateFromEntries(entries, metadata, len(config.Names))
	if err != nil {
		return OptimizerSnapshot{}, err
	}
	return OptimizerSnapshot{Version: 1, Kind: kind, AdamW: &s}, nil
}
