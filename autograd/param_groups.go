package autograd

import (
	"fmt"
	"math"
	"slices"
)

// DefaultParameterGroupName labels the implicit group built by NewAdamW and
// NewSGD when a caller passes a flat parameter slice without group specs.
const DefaultParameterGroupName = "default"

// ParameterGroupSpec names one set of parameters sharing a learning-rate scale
// and a weight decay. Groups must partition the optimizer's parameters exactly:
// every parameter belongs to one group and no parameter appears twice.
//
// LRScale multiplies the optimizer base learning rate, so schedulers keep
// working through OneCycle/SetLearningRate without knowing about groups. It must
// be finite and nonnegative; zero selects the unscaled base rate. WeightDecay is
// the group's own absolute decay, which overrides the base constructor value.
type ParameterGroupSpec struct {
	Name        string
	Params      []Parameter
	LRScale     float32
	WeightDecay float32
}

// ParameterGroup is a resolved spec: the parameters it owns by index into the
// optimizer parameter slice, plus the configuration actually used by Step.
type ParameterGroup struct {
	Name        string
	Indexes     []int
	LRScale     float32
	WeightDecay float32
}

// OptimizerGroupState is the serialized form of one group. Names lists the
// parameter names the group owned, in the order the optimizer stores them.
type OptimizerGroupState struct {
	Name        string
	Names       []string
	LRScale     float32
	WeightDecay float32
}

func normalizeLRScale(scale float32) (float32, error) {
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) || scale < 0 {
		return 0, fmt.Errorf("autograd: invalid parameter group learning-rate scale")
	}
	if scale == 0 {
		return 1, nil
	}
	return scale, nil
}

func validateGroupDecay(decay float32) error {
	if decay < 0 || math.IsNaN(float64(decay)) || math.IsInf(float64(decay), 0) {
		return fmt.Errorf("autograd: invalid parameter group weight decay")
	}
	return nil
}

// PartitionParameters builds one spec per label returned by assign. The order of
// the returned specs follows the first appearance of each label, so callers can
// set LRScale/WeightDecay positionally or by name afterwards. Every parameter
// must be assigned to exactly one non-empty group.
func PartitionParameters(params []Parameter, assign func(name string) string) ([]ParameterGroupSpec, error) {
	if assign == nil {
		return nil, fmt.Errorf("autograd: nil parameter group assignment")
	}
	order := make([]string, 0, 4)
	grouped := make(map[string][]Parameter, 4)
	seen := make(map[string]bool, len(params))
	for _, p := range params {
		if p.Value == nil {
			return nil, fmt.Errorf("autograd: partition needs parameter values")
		}
		if seen[p.Name] {
			return nil, fmt.Errorf("autograd: duplicate parameter name %q", p.Name)
		}
		seen[p.Name] = true
		name := assign(p.Name)
		if name == "" {
			return nil, fmt.Errorf("autograd: parameter %q is not assigned to a group", p.Name)
		}
		if _, ok := grouped[name]; !ok {
			order = append(order, name)
		}
		grouped[name] = append(grouped[name], p)
	}
	if len(order) == 0 {
		return nil, fmt.Errorf("autograd: no parameters to partition")
	}
	specs := make([]ParameterGroupSpec, 0, len(order))
	for _, name := range order {
		specs = append(specs, ParameterGroupSpec{Name: name, Params: grouped[name]})
	}
	return specs, nil
}

// resolveParameterGroups turns specs into resolved groups plus the group of each
// parameter. Empty specs produce one default group holding every parameter,
// matching constructors that take a flat parameter slice.
func resolveParameterGroups(params []Parameter, specs []ParameterGroupSpec) ([]ParameterGroup, []int, error) {
	if len(specs) == 0 {
		indexes := make([]int, len(params))
		for i := range indexes {
			indexes[i] = i
		}
		return []ParameterGroup{{Name: DefaultParameterGroupName, Indexes: indexes, LRScale: 1}}, make([]int, len(params)), nil
	}
	owner := make([]int, len(params))
	for i := range owner {
		owner[i] = -1
	}
	// Look the parameters up by their value instead of scanning the list for
	// every grouped parameter.
	lookup := make(map[*Tensor]int, len(params))
	for i, live := range params {
		if live.Value != nil {
			lookup[live.Value] = i
		}
	}
	groups := make([]ParameterGroup, 0, len(specs))
	specNames := make([]string, len(specs))
	names := make(map[string]bool, len(specs))
	for index, spec := range specs {
		if spec.Name == "" {
			return nil, nil, fmt.Errorf("autograd: parameter group %d has no name", index)
		}
		if names[spec.Name] {
			return nil, nil, fmt.Errorf("autograd: duplicate parameter group %q", spec.Name)
		}
		names[spec.Name], specNames[index] = true, spec.Name
		scale, err := normalizeLRScale(spec.LRScale)
		if err != nil {
			return nil, nil, err
		}
		if err := validateGroupDecay(spec.WeightDecay); err != nil {
			return nil, nil, err
		}
		if len(spec.Params) == 0 && len(params) > 0 {
			return nil, nil, fmt.Errorf("autograd: parameter group %q has no parameters", spec.Name)
		}
		group := ParameterGroup{Name: spec.Name, LRScale: scale, WeightDecay: spec.WeightDecay}
		for _, p := range spec.Params {
			if p.Value == nil {
				return nil, nil, fmt.Errorf("autograd: group %q has a parameter without a value", spec.Name)
			}
			found, ok := lookup[p.Value]
			if !ok {
				return nil, nil, fmt.Errorf("autograd: parameter %q is not part of this optimizer", p.Name)
			}
			if params[found].Name != p.Name {
				return nil, nil, fmt.Errorf("autograd: parameter %q is renamed inside group %q", p.Name, spec.Name)
			}
			if owner[found] >= 0 {
				return nil, nil, fmt.Errorf("autograd: parameter %q appears in groups %q and %q", p.Name, specNames[owner[found]], spec.Name)
			}
			owner[found] = index
			group.Indexes = append(group.Indexes, found)
		}
		groups = append(groups, group)
	}
	for i, owner := range owner {
		if owner < 0 {
			return nil, nil, fmt.Errorf("autograd: parameter %q is not in any group", params[i].Name)
		}
	}
	return groups, owner, nil
}

func parameterNames(params []Parameter) []string {
	names := make([]string, len(params))
	for i, p := range params {
		names[i] = p.Name
	}
	return names
}

func groupIndexOf(groups []ParameterGroup, count int) []int {
	owner := make([]int, count)
	for index, group := range groups {
		for _, paramIndex := range group.Indexes {
			owner[paramIndex] = index
		}
	}
	return owner
}

func groupStates(groups []ParameterGroup, names []string) []OptimizerGroupState {
	states := make([]OptimizerGroupState, 0, len(groups))
	for _, group := range groups {
		owned := make([]string, 0, len(group.Indexes))
		for _, index := range group.Indexes {
			owned = append(owned, names[index])
		}
		states = append(states, OptimizerGroupState{Name: group.Name, Names: owned, LRScale: group.LRScale, WeightDecay: group.WeightDecay})
	}
	return states
}

// validateGroupStates resolves saved groups against the live configuration and
// returns fresh groups to install. Validating never mutates live state, so
// callers can validate before loading anything. State without any group is the
// legacy single-group layout: it fits one live group without a learning-rate
// scale and takes its decay from the saved base hyperparameters.
func validateGroupStates(live []ParameterGroup, names []string, states []OptimizerGroupState, baseDecay float32) ([]ParameterGroup, error) {
	if len(states) == 0 {
		if len(live) != 1 {
			return nil, fmt.Errorf("autograd: saved state has no parameter groups for %d groups", len(live))
		}
		group := live[0]
		if group.LRScale != 1 {
			return nil, fmt.Errorf("autograd: saved state cannot express the learning-rate scale of group %q", group.Name)
		}
		if err := validateGroupDecay(baseDecay); err != nil {
			return nil, err
		}
		group.WeightDecay = baseDecay
		group.Indexes = slices.Clone(group.Indexes)
		return []ParameterGroup{group}, nil
	}
	if len(states) != len(live) {
		return nil, fmt.Errorf("autograd: saved state has %d parameter groups, live optimizer has %d", len(states), len(live))
	}
	resolved := make([]ParameterGroup, 0, len(states))
	for index, state := range states {
		group := live[index]
		if state.Name != group.Name {
			return nil, fmt.Errorf("autograd: parameter group %d is %q, saved state is %q", index, group.Name, state.Name)
		}
		if len(state.Names) != len(group.Indexes) {
			return nil, fmt.Errorf("autograd: parameter group %q holds %d parameters, saved state has %d", group.Name, len(group.Indexes), len(state.Names))
		}
		for position, paramIndex := range group.Indexes {
			if state.Names[position] != names[paramIndex] {
				return nil, fmt.Errorf("autograd: parameter group %q expects %q, saved state has %q", group.Name, names[paramIndex], state.Names[position])
			}
		}
		scale, err := normalizeLRScale(state.LRScale)
		if err != nil {
			return nil, err
		}
		if err := validateGroupDecay(state.WeightDecay); err != nil {
			return nil, err
		}
		resolved = append(resolved, ParameterGroup{Name: group.Name, Indexes: slices.Clone(group.Indexes), LRScale: scale, WeightDecay: state.WeightDecay})
	}
	return resolved, nil
}
