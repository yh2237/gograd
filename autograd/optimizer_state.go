package autograd

import (
	"fmt"
	"strconv"
	"unsafe"
)

// AdamWStateは1つのパラメータのモーメント。
type AdamWState struct {
	M, V []float32
}

// OptimizerStateはAdamWの再開に必要な状態。
type OptimizerState struct {
	StepCount int
	Moments   []AdamWState
}

const (
	optimizerStateFormat  = "gograd-adamw"
	optimizerStateVersion = 1
)

// Stateはホスト上のスナップショットを返す。
func (o *AdamW) State() (OptimizerState, error) {
	if len(o.M)+len(o.mGPU) != len(o.Params) {
		return OptimizerState{}, fmt.Errorf("autograd: optimizer has %d state entries for %d params", len(o.M)+len(o.mGPU), len(o.Params))
	}
	state := OptimizerState{StepCount: o.StepCount, Moments: make([]AdamWState, len(o.Params))}
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
	if len(state.Moments) != len(o.Params) {
		return fmt.Errorf("autograd: optimizer state has %d entries for %d params", len(state.Moments), len(o.Params))
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
	return writeSafetensors(path, entries, metadata)
}

// LoadSafeTensorsはSaveSafeTensorsの出力を検証して復元する。
func (o *AdamW) LoadSafeTensors(path string) error {
	entries, metadata, err := readSafetensors(path)
	if err != nil {
		return err
	}
	if metadata["format"] != optimizerStateFormat {
		return fmt.Errorf("autograd: %s is not an optimizer state", path)
	}
	if version, err := strconv.Atoi(metadata["version"]); err != nil || version != optimizerStateVersion {
		return fmt.Errorf("autograd: unsupported optimizer state version %q", metadata["version"])
	}
	stepCount, err := strconv.Atoi(metadata["step_count"])
	if err != nil || stepCount < 0 {
		return fmt.Errorf("autograd: invalid optimizer step count %q", metadata["step_count"])
	}
	state := OptimizerState{StepCount: stepCount, Moments: make([]AdamWState, len(o.Params))}
	for i := range o.Params {
		m, mOK := entries[fmt.Sprintf("m.%d", i)]
		v, vOK := entries[fmt.Sprintf("v.%d", i)]
		if !mOK || !vOK {
			return fmt.Errorf("autograd: optimizer state is missing parameter %d", i)
		}
		state.Moments[i] = AdamWState{M: m.Values, V: v.Values}
	}
	return o.LoadState(state)
}
