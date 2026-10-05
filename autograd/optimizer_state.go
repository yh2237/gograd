package autograd

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strconv"
	"unsafe"
)

// AdamWStateは1つのパラメータのモーメント。
type AdamWState struct {
	M, V []float32
}

// OptimizerStateはAdamWの再開に必要な状態。
type OptimizerState struct {
	StepCount                          int
	Moments                            []AdamWState
	Names                              []string
	Shapes                             [][]int
	LR, WeightDecay, Beta1, Beta2, Eps float32
}

const (
	optimizerStateFormat  = "gograd-adamw"
	optimizerStateVersion = 1
)

// Stateはホスト上のスナップショットを返す。
func (o *AdamW) State() (OptimizerState, error) {
	if err := o.validateStorage(); err != nil {
		return OptimizerState{}, err
	}
	if len(o.M)+len(o.mGPU) != len(o.Params) {
		return OptimizerState{}, fmt.Errorf("autograd: optimizer has %d state entries for %d params", len(o.M)+len(o.mGPU), len(o.Params))
	}
	step := o.StepCount
	if o.graphStep != nil {
		var err error
		step, err = o.GraphStepCount()
		if err != nil {
			return OptimizerState{}, err
		}
	}
	state := OptimizerState{StepCount: step, Moments: make([]AdamWState, len(o.Params)),
		LR: o.LR, WeightDecay: o.WeightDecay, Beta1: o.Beta1, Beta2: o.Beta2, Eps: o.Eps}
	for _, p := range o.Params {
		state.Names = append(state.Names, p.Name)
		state.Shapes = append(state.Shapes, slices.Clone(p.Value.Shape))
	}
	if len(o.mGPU) > 0 {
		for i, p := range o.Params {
			n := p.Value.Numel()
			m, v := make([]float32, n), make([]float32, n)
			if n > 0 {
				if err := o.mGPU[i].CopyToHost(unsafe.Slice((*byte)(unsafe.Pointer(&m[0])), n*4)); err != nil {
					return OptimizerState{}, err
				}
				if err := o.vGPU[i].CopyToHost(unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), n*4)); err != nil {
					return OptimizerState{}, err
				}
			}
			state.Moments[i] = AdamWState{M: m, V: v}
		}
		return state, nil
	}
	for i := range o.Params {
		state.Moments[i] = AdamWState{M: append([]float32(nil), o.M[i]...), V: append([]float32(nil), o.V[i]...)}
	}
	return state, nil
}

// LoadStateは保存済みの状態を検証して復元する。
func (o *AdamW) LoadState(state OptimizerState) error {
	if err := o.validateState(state); err != nil {
		return err
	}
	if len(o.mGPU) > 0 {
		for i, p := range o.Params {
			n := p.Value.Numel()
			if len(state.Moments[i].M) != n || len(state.Moments[i].V) != n {
				return fmt.Errorf("autograd: optimizer state %d has wrong size", i)
			}
			if n == 0 {
				continue
			}
			if err := o.mGPU[i].CopyFromHost(unsafe.Slice((*byte)(unsafe.Pointer(&state.Moments[i].M[0])), n*4)); err != nil {
				return err
			}
			if err := o.vGPU[i].CopyFromHost(unsafe.Slice((*byte)(unsafe.Pointer(&state.Moments[i].V[0])), n*4)); err != nil {
				return err
			}
		}
	} else {
		if len(o.M) != len(o.Params) {
			return fmt.Errorf("autograd: optimizer has %d host state entries for %d params", len(o.M), len(o.Params))
		}
		for i, p := range o.Params {
			n := p.Value.Numel()
			if len(state.Moments[i].M) != n || len(state.Moments[i].V) != n {
				return fmt.Errorf("autograd: optimizer state %d has wrong size", i)
			}
			copy(o.M[i], state.Moments[i].M)
			copy(o.V[i], state.Moments[i].V)
		}
	}
	o.StepCount = state.StepCount
	o.LR, o.WeightDecay, o.Beta1, o.Beta2, o.Eps = state.LR, state.WeightDecay, state.Beta1, state.Beta2, state.Eps
	if o.graphStep != nil {
		step := int32(state.StepCount)
		if err := o.graphStep.CopyFromHost(unsafe.Slice((*byte)(unsafe.Pointer(&step)), 4)); err != nil {
			return err
		}
	}
	return nil
}

// SaveSafeTensorsはoptimizerの状態をsafetensorsへ保存する。
func (o *AdamW) SaveSafeTensors(path string) error {
	state, err := o.State()
	if err != nil {
		return err
	}
	entries, metadata, err := optimizerEntries(state)
	if err != nil {
		return err
	}
	return writeSafetensors(path, entries, metadata)
}

func optimizerEntries(state OptimizerState) (map[string]safeTensorEntry, map[string]string, error) {
	entries := make(map[string]safeTensorEntry, len(state.Moments)*2)
	for i, moment := range state.Moments {
		entries[fmt.Sprintf("m.%d", i)] = safeTensorEntry{Values: moment.M, Shape: []int{len(moment.M)}}
		entries[fmt.Sprintf("v.%d", i)] = safeTensorEntry{Values: moment.V, Shape: []int{len(moment.V)}}
	}
	metadata := map[string]string{
		"format":     optimizerStateFormat,
		"version":    strconv.Itoa(optimizerStateVersion),
		"step_count": strconv.Itoa(state.StepCount),
	}
	state.Moments = nil
	config, err := json.Marshal(state)
	if err != nil {
		return nil, nil, err
	}
	metadata["optimizer"] = string(config)
	return entries, metadata, nil
}

// LoadSafeTensorsはSaveSafeTensorsの出力を検証して復元する。
func (o *AdamW) LoadSafeTensors(path string) error {
	entries, metadata, err := readSafetensors(path)
	if err != nil {
		return err
	}
	state, err := optimizerStateFromEntries(entries, metadata, len(o.Params))
	if err != nil {
		return err
	}
	return o.LoadState(state)
}

func optimizerStateFromEntries(entries map[string]safeTensorEntry, metadata map[string]string, count int) (OptimizerState, error) {
	if metadata["format"] != optimizerStateFormat {
		return OptimizerState{}, fmt.Errorf("autograd: not an optimizer state")
	}
	if version, err := strconv.Atoi(metadata["version"]); err != nil || version != optimizerStateVersion {
		return OptimizerState{}, fmt.Errorf("autograd: unsupported optimizer state version %q", metadata["version"])
	}
	stepCount, err := strconv.Atoi(metadata["step_count"])
	if err != nil || stepCount < 0 {
		return OptimizerState{}, fmt.Errorf("autograd: invalid optimizer step count %q", metadata["step_count"])
	}
	var state OptimizerState
	if err := json.Unmarshal([]byte(metadata["optimizer"]), &state); err != nil {
		return OptimizerState{}, fmt.Errorf("autograd: invalid optimizer configuration: %w", err)
	}
	if state.StepCount != stepCount || len(entries) != count*2 {
		return OptimizerState{}, fmt.Errorf("autograd: optimizer state count or step mismatch")
	}
	state.Moments = make([]AdamWState, count)
	for i := 0; i < count; i++ {
		m, mOK := entries[fmt.Sprintf("m.%d", i)]
		v, vOK := entries[fmt.Sprintf("v.%d", i)]
		if !mOK || !vOK {
			return OptimizerState{}, fmt.Errorf("autograd: optimizer state is missing parameter %d", i)
		}
		if !slices.Equal(m.Shape, []int{len(m.Values)}) || !slices.Equal(v.Shape, []int{len(v.Values)}) {
			return OptimizerState{}, fmt.Errorf("autograd: optimizer moment shape mismatch %d", i)
		}
		state.Moments[i] = AdamWState{M: m.Values, V: v.Values}
	}
	return state, nil
}

func (o *AdamW) validateStorage() error {
	if len(o.Params) == 0 {
		return nil
	}
	device := o.Params[0].Value.Device
	for _, p := range o.Params {
		if p.Value.Device != device {
			return fmt.Errorf("autograd: optimizer parameters must use one device")
		}
	}
	if len(o.mGPU) > 0 {
		if len(o.mGPU) != len(o.Params) || len(o.vGPU) != len(o.Params) {
			return fmt.Errorf("autograd: incomplete CUDA optimizer storage")
		}
	} else if len(o.M) != len(o.Params) || len(o.V) != len(o.Params) {
		return fmt.Errorf("autograd: incomplete host optimizer storage")
	}
	return nil
}

func (o *AdamW) validateState(state OptimizerState) error {
	if err := o.validateStorage(); err != nil {
		return err
	}
	if state.StepCount < 0 || state.StepCount > math.MaxInt32 || len(state.Moments) != len(o.Params) || len(state.Names) != len(o.Params) || len(state.Shapes) != len(o.Params) {
		return fmt.Errorf("autograd: invalid optimizer step or parameter count")
	}
	for _, value := range []float32{state.LR, state.WeightDecay, state.Beta1, state.Beta2, state.Eps} {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("autograd: non-finite optimizer configuration")
		}
	}
	if state.LR < 0 || state.WeightDecay < 0 || state.Beta1 < 0 || state.Beta1 >= 1 || state.Beta2 < 0 || state.Beta2 >= 1 || state.Eps <= 0 {
		return fmt.Errorf("autograd: invalid optimizer configuration")
	}
	for i, p := range o.Params {
		n := p.Value.Numel()
		if state.Names[i] != p.Name || !slices.Equal(state.Shapes[i], p.Value.Shape) || len(state.Moments[i].M) != n || len(state.Moments[i].V) != n {
			return fmt.Errorf("autograd: optimizer parameter %d name or shape mismatch", i)
		}
		for j, m := range state.Moments[i].M {
			v := state.Moments[i].V[j]
			if math.IsNaN(float64(m)) || math.IsInf(float64(m), 0) || math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v < 0 {
				return fmt.Errorf("autograd: invalid optimizer moment %d/%d", i, j)
			}
		}
	}
	return nil
}

// Close releases optimizer-owned CUDA buffers, leaving parameters untouched.
func (o *AdamW) Close() {
	for _, b := range append(o.mGPU, o.vGPU...) {
		_ = b.Free()
	}
	if o.unityGPU != nil {
		_ = o.unityGPU.Free()
	}
	if o.graphStep != nil {
		_ = o.graphStep.Free()
	}
	o.mGPU, o.vGPU, o.M, o.V = nil, nil, nil, nil
	o.unityGPU, o.graphStep = nil, nil
}
