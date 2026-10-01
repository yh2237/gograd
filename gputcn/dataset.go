package gputcn

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"os"
	"sort"
)

// Dataset is a prepared frame-intonation corpus: the frame features, targets
// and mask that the training step consumes. Feature values are stored sparsely
// per record as (row, column, value) triples, matching how the Python pipeline
// materializes them.
type Dataset struct {
	Version      int      `json:"version"`
	FrameMS      float64  `json:"frame_ms"`
	FeatureNames []string `json:"feature_names"`
	Records      []Record `json:"records"`
}

// Record is one utterance.
type Record struct {
	ID      string    `json:"id"`
	Length  int       `json:"length"`
	Rows    []int32   `json:"rows"`
	Cols    []int32   `json:"cols"`
	Vals    []float32 `json:"vals"`
	Targets []float64 `json:"targets"`
	Mask    []bool    `json:"mask"`
}

// LoadDataset reads a prepared dataset JSON file.
func LoadDataset(path string) (*Dataset, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var dataset Dataset
	if err := json.Unmarshal(raw, &dataset); err != nil {
		return nil, fmt.Errorf("gputcn: parse dataset: %w", err)
	}
	if dataset.Version != 1 {
		return nil, fmt.Errorf("gputcn: unsupported dataset version %d", dataset.Version)
	}
	if len(dataset.FeatureNames) == 0 {
		return nil, fmt.Errorf("gputcn: dataset has no feature names")
	}
	for i := range dataset.Records {
		if err := dataset.Records[i].validate(len(dataset.FeatureNames)); err != nil {
			return nil, err
		}
	}
	return &dataset, nil
}

func (r *Record) validate(features int) error {
	if r.Length <= 0 || len(r.Targets) != r.Length || len(r.Mask) != r.Length {
		return fmt.Errorf("gputcn: record %q has inconsistent length", r.ID)
	}
	if len(r.Rows) != len(r.Cols) || len(r.Rows) != len(r.Vals) {
		return fmt.Errorf("gputcn: record %q has inconsistent sparse arrays", r.ID)
	}
	for i := range r.Rows {
		if int(r.Rows[i]) < 0 || int(r.Rows[i]) >= r.Length {
			return fmt.Errorf("gputcn: record %q row out of range", r.ID)
		}
		if int(r.Cols[i]) < 0 || int(r.Cols[i]) >= features {
			return fmt.Errorf("gputcn: record %q column out of range", r.ID)
		}
	}
	return nil
}

// Features returns the feature count.
func (d *Dataset) Features() int { return len(d.FeatureNames) }

// Split divides the records into train and validation indices. Records whose
// id hash is divisible by ten go to validation, which is stable across runs.
func (d *Dataset) Split() (train, validation []int) {
	for i := range d.Records {
		if idHash(d.Records[i].ID)%10 == 0 {
			validation = append(validation, i)
		} else {
			train = append(train, i)
		}
	}
	if len(train) == 0 && len(validation) > 1 {
		validation = validation[:1]
		for i := range d.Records {
			if i != validation[0] {
				train = append(train, i)
			}
		}
	}
	if len(validation) == 0 && len(train) > 1 {
		validation = train[:1]
		train = train[1:]
	}
	return train, validation
}

func idHash(id string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(id))
	return h.Sum32()
}

// Batch is a dense, padded group of records ready for the GPU.
type Batch struct {
	Batch    int
	Time     int
	Features int
	Values   []float32 // [batch,time,features] row-major
	Targets  []float64 // [batch,time]
	Mask     [][]bool  // [batch][time]
}

// BuildBatch expands the selected records into a dense padded batch.
func (d *Dataset) BuildBatch(indices []int) *Batch {
	length := 0
	for _, index := range indices {
		if d.Records[index].Length > length {
			length = d.Records[index].Length
		}
	}
	features := d.Features()
	batch := &Batch{
		Batch:    len(indices),
		Time:     length,
		Features: features,
		Values:   make([]float32, len(indices)*length*features),
		Targets:  make([]float64, len(indices)*length),
		Mask:     make([][]bool, len(indices)),
	}
	for row, index := range indices {
		record := &d.Records[index]
		batch.Mask[row] = make([]bool, length)
		for t := 0; t < record.Length; t++ {
			batch.Targets[row*length+t] = record.Targets[t]
			batch.Mask[row][t] = record.Mask[t]
		}
		for i := range record.Rows {
			values := (row*length+int(record.Rows[i]))*features + int(record.Cols[i])
			batch.Values[values] = record.Vals[i]
		}
	}
	return batch
}

// SortIndicesByLength orders indices by record length, which keeps padding
// small when batching.
func (d *Dataset) SortIndicesByLength(indices []int) {
	sort.SliceStable(indices, func(a, b int) bool {
		return d.Records[indices[a]].Length < d.Records[indices[b]].Length
	})
}
