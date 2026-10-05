package autograd

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

// SaveTrainingCheckpoint stores current model/buffers, AdamW hyperparameters and
// moments, OneCycle position, and caller metadata in one atomically replaced
// safetensors file. Sampler/trainer state can be JSON in caller metadata.
func SaveTrainingCheckpoint(path string, module *Module, optimizer *AdamW, scheduler *OneCycle, metadata map[string]string) error {
	if module == nil || optimizer == nil || scheduler == nil {
		return fmt.Errorf("autograd: model, optimizer and scheduler are required")
	}
	state, err := optimizer.State()
	if err != nil {
		return err
	}
	if err := optimizer.validateState(state); err != nil {
		return err
	}
	if err := validateOneCycle(*scheduler); err != nil {
		return err
	}
	if scheduler.StepCount != state.StepCount {
		return fmt.Errorf("autograd: optimizer/scheduler step mismatch")
	}
	model, err := module.stateEntries()
	if err != nil {
		return err
	}
	opt, optMeta, err := optimizerEntries(state)
	if err != nil {
		return err
	}
	entries := make(map[string]safeTensorEntry, len(model)+len(opt))
	for name, entry := range model {
		entries["model."+name] = entry
	}
	for name, entry := range opt {
		entries["optimizer."+name] = entry
	}
	schedule, err := json.Marshal(scheduler)
	if err != nil {
		return err
	}
	user, err := json.Marshal(metadata)
	if err != nil {
		return err
	}
	optMeta["format"] = "gograd-training-1"
	optMeta["scheduler"] = string(schedule)
	optMeta["user"] = string(user)
	return writeSafetensors(path, entries, optMeta)
}

// TrainingCheckpointMetadata reads caller metadata without mutating live state.
func TrainingCheckpointMetadata(path string) (map[string]string, error) {
	_, metadata, err := readSafetensors(path)
	if err != nil {
		return nil, err
	}
	if metadata["format"] != "gograd-training-1" {
		return nil, fmt.Errorf("autograd: not a training checkpoint")
	}
	var user map[string]string
	if err := json.Unmarshal([]byte(metadata["user"]), &user); err != nil {
		return nil, err
	}
	return user, nil
}

// LoadTrainingCheckpoint validates every name/shape and all state before copying
// model or optimizer data. Device transfer failures may still interrupt loading.
func LoadTrainingCheckpoint(path string, module *Module, optimizer *AdamW, scheduler *OneCycle) (map[string]string, error) {
	if module == nil || optimizer == nil || scheduler == nil {
		return nil, fmt.Errorf("autograd: model, optimizer and scheduler are required")
	}
	entries, metadata, err := readSafetensors(path)
	if err != nil {
		return nil, err
	}
	if metadata["format"] != "gograd-training-1" {
		return nil, fmt.Errorf("autograd: not a training checkpoint")
	}
	model, opt := map[string]safeTensorEntry{}, map[string]safeTensorEntry{}
	for name, entry := range entries {
		if suffix, ok := strings.CutPrefix(name, "model."); ok {
			model[suffix] = entry
		} else if suffix, ok := strings.CutPrefix(name, "optimizer."); ok {
			opt[suffix] = entry
		} else {
			return nil, fmt.Errorf("autograd: unknown checkpoint entry %q", name)
		}
	}
	if err := module.validateEntries(model); err != nil {
		return nil, err
	}
	metadata["format"] = optimizerStateFormat
	state, err := optimizerStateFromEntries(opt, metadata, len(optimizer.Params))
	if err != nil {
		return nil, err
	}
	if err := optimizer.validateState(state); err != nil {
		return nil, err
	}
	var schedule OneCycle
	if err := json.Unmarshal([]byte(metadata["scheduler"]), &schedule); err != nil {
		return nil, err
	}
	if err := validateOneCycle(schedule); err != nil {
		return nil, err
	}
	if schedule.StepCount != state.StepCount {
		return nil, fmt.Errorf("autograd: optimizer/scheduler step mismatch")
	}
	var user map[string]string
	if err := json.Unmarshal([]byte(metadata["user"]), &user); err != nil {
		return nil, err
	}
	if err := module.loadEntries(model); err != nil {
		return nil, err
	}
	if err := optimizer.LoadState(state); err != nil {
		return nil, err
	}
	*scheduler = schedule
	return user, nil
}

func validateOneCycle(c OneCycle) error {
	for _, value := range []float64{c.MaxLR, c.PctStart, c.DivFactor, c.FinalDivFactor} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("autograd: non-finite OneCycle configuration")
		}
	}
	if c.Total < 2 || c.StepCount < 0 || c.StepCount > c.Total || c.MaxLR <= 0 || c.PctStart <= 0 || c.PctStart >= 1 || c.DivFactor <= 0 || c.FinalDivFactor <= 0 {
		return fmt.Errorf("autograd: invalid OneCycle state")
	}
	return nil
}
