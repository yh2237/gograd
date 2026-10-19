package data

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"slices"
	"strconv"
	"strings"
)

// ParseOptions configures ParseNumericCSV.
type ParseOptions struct {
	// Delimiter separates fields; zero means comma.
	Delimiter rune
	// Comment, when nonzero, skips lines whose first rune matches it. It follows
	// encoding/csv's ReaderOptions.Comment rules.
	Comment rune
	// NoHeader treats the first line as data instead of column names.
	NoHeader bool
}

// NumericTable is a rectangular table of finite float32 values. Names is empty
// when the source had no header.
type NumericTable struct {
	Names []string
	Rows  [][]float32
}

func (t *NumericTable) Len() int     { return len(t.Rows) }
func (t *NumericTable) Columns() int { return len(t.Names) }

// ParseNumericCSV reads a rectangular numeric CSV/TSV table. Fields and header
// names are space-trimmed. Values must be finite: NaN or infinity usually means a
// malformed or uncleaned file, so cleaning stays the caller's explicit step. A
// malformed row reports its 1-based source line.
func ParseNumericCSV(r io.Reader, options ParseOptions) (*NumericTable, error) {
	if r == nil {
		return nil, fmt.Errorf("data: nil CSV reader")
	}
	reader := csv.NewReader(r)
	if options.Delimiter != 0 {
		reader.Comma = options.Delimiter
	}
	reader.Comment = options.Comment
	reader.FieldsPerRecord = -1
	reader.ReuseRecord = true
	table := &NumericTable{}
	width := 0
	line := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("data: CSV line %d: %w", line+1, err)
		}
		line++
		if len(record) == 1 && strings.TrimSpace(record[0]) == "" {
			continue
		}
		if width == 0 {
			width = len(record)
		}
		if len(record) != width {
			return nil, fmt.Errorf("data: CSV line %d has %d columns, want %d", line, len(record), width)
		}
		if len(table.Rows) == 0 && len(table.Names) == 0 && !options.NoHeader {
			table.Names = make([]string, width)
			for i, field := range record {
				table.Names[i] = strings.TrimSpace(field)
			}
			continue
		}
		row := make([]float32, width)
		for i, field := range record {
			value, err := strconv.ParseFloat(strings.TrimSpace(field), 32)
			if err != nil {
				return nil, fmt.Errorf("data: CSV line %d column %d: %w", line, i, err)
			}
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("data: CSV line %d column %d is not finite", line, i)
			}
			row[i] = float32(value)
		}
		table.Rows = append(table.Rows, row)
	}
	if len(table.Rows) == 0 && len(table.Names) == 0 {
		return nil, fmt.Errorf("data: CSV has no data rows")
	}
	if len(table.Names) == 0 {
		table.Names = make([]string, width)
	}
	return table, nil
}

// RegressionSample is one row split into a fixed-length feature vector and a
// scalar target. Features may borrow table memory.
type RegressionSample struct {
	Features []float32
	Target   float32
}

// RegressionBatch is a stacked regression batch: Values holds the flattened
// [batch, features] rows and Shape records that layout.
type RegressionBatch struct {
	Values  []float32
	Shape   []int
	Targets []float32
}

// StackRegression copies equal-length samples into one batch. All Stack* helpers
// must not mutate borrowed samples.
func StackRegression(samples []RegressionSample) (RegressionBatch, error) {
	if len(samples) == 0 {
		return RegressionBatch{}, fmt.Errorf("data: cannot stack an empty regression batch")
	}
	features := len(samples[0].Features)
	if features == 0 {
		return RegressionBatch{}, fmt.Errorf("data: regression samples need at least one feature")
	}
	batch := RegressionBatch{Values: make([]float32, len(samples)*features), Shape: []int{len(samples), features}, Targets: make([]float32, len(samples))}
	for i, sample := range samples {
		if len(sample.Features) != features {
			return RegressionBatch{}, fmt.Errorf("data: stack requires equal feature counts")
		}
		copy(batch.Values[i*features:], sample.Features)
		batch.Targets[i] = sample.Target
	}
	return batch, nil
}

// RegressionSamples splits table rows into features and a scalar target. Empty
// featureColumns selects every column except the target in ascending order;
// targetColumn of -1 selects the last column. Feature samples borrow table rows,
// so keep the table alive while iterating.
func (t *NumericTable) RegressionSamples(featureColumns []int, targetColumn int) ([]RegressionSample, error) {
	if t == nil || len(t.Rows) == 0 {
		return nil, fmt.Errorf("data: empty numeric table")
	}
	width := len(t.Rows[0])
	for _, row := range t.Rows {
		if len(row) != width {
			return nil, fmt.Errorf("data: ragged numeric table")
		}
	}
	if targetColumn < 0 {
		targetColumn = width - 1
	}
	if targetColumn < 0 || targetColumn >= width {
		return nil, fmt.Errorf("data: target column %d out of range", targetColumn)
	}
	columns := featureColumns
	if len(columns) == 0 {
		columns = make([]int, 0, width-1)
		for i := 0; i < width; i++ {
			if i != targetColumn {
				columns = append(columns, i)
			}
		}
	}
	seen := make(map[int]bool, len(columns))
	for _, column := range columns {
		if column < 0 || column >= width {
			return nil, fmt.Errorf("data: feature column %d out of range", column)
		}
		if column == targetColumn || seen[column] {
			return nil, fmt.Errorf("data: feature column %d is duplicated or is the target", column)
		}
		seen[column] = true
	}
	samples := make([]RegressionSample, len(t.Rows))
	for i, row := range t.Rows {
		features := make([]float32, len(columns))
		for j, column := range columns {
			features[j] = row[column]
		}
		samples[i] = RegressionSample{Features: features, Target: row[targetColumn]}
	}
	return samples, nil
}

// StandardizerState is the JSON-serializable form of a fitted Standardizer.
type StandardizerState struct {
	Version   int
	Mean, Std []float32
}

// Standardizer centers and scales each feature with statistics fitted on the
// training split. Reusing the training statistics for validation and inference
// is what keeps a run reproducible, so store State in the training checkpoint
// and restore it before evaluating or predicting.
type Standardizer struct {
	Mean []float32
	Std  []float32
}

// FitStandardizer computes per-feature mean and population standard deviation
// over the samples, accumulating in float64 so the result does not depend on the
// platform's float32 summation order. A constant feature gets a unit scale, so it
// becomes exactly zero after Apply instead of dividing by zero.
func FitStandardizer(samples []RegressionSample) (Standardizer, error) {
	if len(samples) == 0 {
		return Standardizer{}, fmt.Errorf("data: cannot fit on an empty sample set")
	}
	features := len(samples[0].Features)
	if features == 0 {
		return Standardizer{}, fmt.Errorf("data: samples need at least one feature")
	}
	mean := make([]float64, features)
	for _, sample := range samples {
		if len(sample.Features) != features {
			return Standardizer{}, fmt.Errorf("data: inconsistent feature counts")
		}
		for i, value := range sample.Features {
			mean[i] += float64(value)
		}
	}
	count := float64(len(samples))
	for i := range mean {
		mean[i] /= count
	}
	// The variance is accumulated about the mean. The one-pass formula
	// E[x^2]-E[x]^2 loses every significant digit when the mean is large
	// compared with the spread.
	square := make([]float64, features)
	for _, sample := range samples {
		for i, value := range sample.Features {
			difference := float64(value) - mean[i]
			square[i] += difference * difference
		}
	}
	standardizer := Standardizer{Mean: make([]float32, features), Std: make([]float32, features)}
	for i := range mean {
		m := mean[i]
		variance := square[i] / count
		if variance < 0 {
			variance = 0
		}
		standardizer.Mean[i] = float32(m)
		std := math.Sqrt(variance)
		if std == 0 {
			std = 1
		}
		standardizer.Std[i] = float32(std)
	}
	return standardizer, nil
}

// Apply returns a centered/scaled copy of the sample.
func (s Standardizer) Apply(sample RegressionSample) RegressionSample {
	out := RegressionSample{Features: make([]float32, len(s.Mean)), Target: sample.Target}
	for i := range out.Features {
		if i < len(sample.Features) {
			out.Features[i] = (sample.Features[i] - s.Mean[i]) / s.Std[i]
		}
	}
	return out
}

// ApplyBatch standardizes a collated batch in place, which is what training and
// evaluation loops want before transferring the batch to a device.
func (s Standardizer) ApplyBatch(batch *RegressionBatch) error {
	if batch == nil || len(batch.Shape) != 2 || batch.Shape[1] != len(s.Mean) ||
		batch.Shape[0] < 0 || len(batch.Values) != batch.Shape[0]*batch.Shape[1] {
		return fmt.Errorf("data: batch features do not match the standardizer")
	}
	for row := 0; row < batch.Shape[0]; row++ {
		base := row * batch.Shape[1]
		for i := range s.Mean {
			batch.Values[base+i] = (batch.Values[base+i] - s.Mean[i]) / s.Std[i]
		}
	}
	return nil
}

func (s Standardizer) State() StandardizerState {
	return StandardizerState{Version: 1, Mean: slices.Clone(s.Mean), Std: slices.Clone(s.Std)}
}

// LoadState replaces the statistics after validating them, so a checkpoint with
// non-finite or mismatched statistics cannot silently distort a run.
func (s *Standardizer) LoadState(state StandardizerState) error {
	if state.Version != 1 {
		return fmt.Errorf("data: unsupported standardizer state version %d", state.Version)
	}
	if len(state.Mean) == 0 || len(state.Mean) != len(state.Std) {
		return fmt.Errorf("data: invalid standardizer state size")
	}
	for i := range state.Mean {
		if math.IsNaN(float64(state.Mean[i])) || math.IsInf(float64(state.Mean[i]), 0) ||
			math.IsNaN(float64(state.Std[i])) || math.IsInf(float64(state.Std[i]), 0) || state.Std[i] <= 0 {
			return fmt.Errorf("data: non-finite standardizer state at feature %d", i)
		}
	}
	s.Mean, s.Std = slices.Clone(state.Mean), slices.Clone(state.Std)
	return nil
}
