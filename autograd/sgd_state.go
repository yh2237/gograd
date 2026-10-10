package autograd

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strconv"

	"github.com/yh2237/gograd/tensor"
)

type SGDState struct {
	StepCount   int
	Names       []string
	Shapes      [][]int
	Options     SGDOptions
	Initialized []bool
	Buffers     [][]float32
}

const sgdStateFormat = "gograd-sgd"

func (o *SGD) State() (SGDState, error) {
	if err := o.validateOpen(); err != nil {
		return SGDState{}, err
	}
	step := o.StepCount
	if o.graphStep != nil {
		var err error
		step, err = o.GraphStepCount()
		if err != nil {
			return SGDState{}, err
		}
	}
	s := SGDState{StepCount: step, Options: o.config(), Initialized: make([]bool, len(o.Params)), Buffers: make([][]float32, len(o.Params))}
	for i, p := range o.Params {
		s.Names = append(s.Names, p.Name)
		s.Shapes = append(s.Shapes, slices.Clone(p.Value.Shape))
		if len(o.initializedGPU) > 0 {
			var flag [1]float32
			if err := o.initializedGPU[i].CopyToHost(floatBytes(flag[:])); err != nil {
				return SGDState{}, err
			}
			if flag[0] != 0 && flag[0] != 1 {
				return SGDState{}, fmt.Errorf("autograd: invalid SGD initialization flag")
			}
			s.Initialized[i] = flag[0] == 1
			if s.Initialized[i] {
				s.Buffers[i] = make([]float32, p.Value.Numel())
				if err := o.bufferGPU[i].CopyToHost(floatBytes(s.Buffers[i])); err != nil {
					return SGDState{}, err
				}
			}
		} else {
			s.Initialized[i] = o.initialized[i]
			if s.Initialized[i] {
				s.Buffers[i] = slices.Clone(o.buffers[i])
			}
		}
	}
	return s, nil
}
func (o *SGD) validateState(s SGDState) error {
	if err := o.validateOpen(); err != nil {
		return err
	}
	if err := s.Options.validate(); err != nil {
		return err
	}
	n := len(o.Params)
	if s.StepCount < 0 || s.StepCount > math.MaxInt32 || len(s.Names) != n || len(s.Shapes) != n || len(s.Buffers) != n || len(s.Initialized) != n {
		return fmt.Errorf("autograd: invalid SGD state count")
	}
	for i, p := range o.Params {
		if s.Names[i] != p.Name || !slices.Equal(s.Shapes[i], p.Value.Shape) {
			return fmt.Errorf("autograd: SGD parameter %d name/shape mismatch", i)
		}
		if s.Initialized[i] {
			if s.Options.Momentum <= 0 || s.StepCount == 0 || len(s.Buffers[i]) != p.Value.Numel() {
				return fmt.Errorf("autograd: invalid SGD momentum buffer %d", i)
			}
		} else if len(s.Buffers[i]) != 0 {
			return fmt.Errorf("autograd: uninitialized SGD buffer has values")
		}
		for _, v := range s.Buffers[i] {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("autograd: non-finite SGD momentum")
			}
		}
	}
	return nil
}
func (o *SGD) LoadState(s SGDState) error {
	if err := o.validateState(s); err != nil {
		return err
	}
	var deviceBuffers, deviceFlags = o.bufferGPU, o.initializedGPU
	if len(o.Params) > 0 && o.Params[0].Value.Device == tensor.CUDA {
		deviceBuffers, deviceFlags = nil, nil
		if s.Options.Momentum > 0 {
			var err error
			deviceBuffers, deviceFlags, err = sgdDeviceStorage(o.Params)
			if err != nil {
				return err
			}
			for i, initialized := range s.Initialized {
				if !initialized {
					continue
				}
				if err := deviceBuffers[i].CopyFromHost(floatBytes(s.Buffers[i])); err != nil {
					sgdFreeStorage(deviceBuffers, deviceFlags)
					return err
				}
				if err := deviceFlags[i].CopyFromHost(floatBytes([]float32{1})); err != nil {
					sgdFreeStorage(deviceBuffers, deviceFlags)
					return err
				}
			}
		}
		sgdFreeStorage(o.bufferGPU, o.initializedGPU)
	}
	o.bufferGPU, o.initializedGPU = deviceBuffers, deviceFlags
	o.buffers = make([][]float32, len(s.Buffers))
	for i, b := range s.Buffers {
		o.buffers[i] = slices.Clone(b)
	}
	o.initialized = slices.Clone(s.Initialized)
	o.options = s.Options
	o.LR = s.Options.LR
	o.StepCount = s.StepCount
	if o.graphStep != nil {
		return o.PrepareGraph()
	}
	return nil
}
func sgdEntries(s SGDState) (map[string]safeTensorEntry, map[string]string, error) {
	entries := make(map[string]safeTensorEntry)
	for i, initialized := range s.Initialized {
		if initialized {
			b := s.Buffers[i]
			entries[fmt.Sprintf("buffer.%d", i)] = safeTensorEntry{Values: b, Shape: []int{len(b)}}
		}
	}
	s.Buffers = nil
	config, err := json.Marshal(s)
	if err != nil {
		return nil, nil, err
	}
	return entries, map[string]string{"format": sgdStateFormat, "version": "1", "step_count": strconv.Itoa(s.StepCount), "optimizer": string(config)}, nil
}
func sgdStateFromEntries(entries map[string]safeTensorEntry, metadata map[string]string) (SGDState, error) {
	var s SGDState
	if metadata["format"] != sgdStateFormat || metadata["version"] != "1" {
		return s, fmt.Errorf("autograd: unsupported SGD state format")
	}
	if err := json.Unmarshal([]byte(metadata["optimizer"]), &s); err != nil {
		return s, err
	}
	step, err := strconv.Atoi(metadata["step_count"])
	if err != nil || step != s.StepCount || step < 0 || len(s.Initialized) != len(s.Names) {
		return s, fmt.Errorf("autograd: invalid SGD state metadata")
	}
	s.Buffers = make([][]float32, len(s.Names))
	expected := 0
	for i, initialized := range s.Initialized {
		if !initialized {
			continue
		}
		expected++
		entry, ok := entries[fmt.Sprintf("buffer.%d", i)]
		if !ok || !slices.Equal(entry.Shape, []int{len(entry.Values)}) {
			return s, fmt.Errorf("autograd: missing or invalid SGD buffer %d", i)
		}
		s.Buffers[i] = entry.Values
	}
	if len(entries) != expected {
		return s, fmt.Errorf("autograd: unexpected SGD state entries")
	}
	return s, nil
}
func (o *SGD) SaveSafeTensors(path string) error {
	s, err := o.State()
	if err != nil {
		return err
	}
	if err := o.validateState(s); err != nil {
		return err
	}
	entries, metadata, err := sgdEntries(s)
	if err != nil {
		return err
	}
	return writeSafetensors(path, entries, metadata)
}
func (o *SGD) LoadSafeTensors(path string) error {
	entries, metadata, err := readSafetensors(path)
	if err != nil {
		return err
	}
	s, err := sgdStateFromEntries(entries, metadata)
	if err != nil {
		return err
	}
	return o.LoadState(s)
}
func (o *SGD) Snapshot() (OptimizerSnapshot, error) {
	s, err := o.State()
	if err != nil {
		return OptimizerSnapshot{}, err
	}
	return OptimizerSnapshot{Version: 1, Kind: "sgd", SGD: &s}, nil
}
func (o *SGD) ValidateSnapshot(s OptimizerSnapshot) error {
	if s.Version != 1 || s.Kind != "sgd" || s.SGD == nil || s.AdamW != nil {
		return fmt.Errorf("autograd: incompatible SGD snapshot")
	}
	return o.validateState(*s.SGD)
}
func (o *SGD) LoadSnapshot(s OptimizerSnapshot) error {
	if err := o.ValidateSnapshot(s); err != nil {
		return err
	}
	return o.LoadState(*s.SGD)
}
