// Package metrics accumulates host-side evaluation statistics without tensor or
// backend dependencies. Values are zero for empty accumulators; Count identifies
// that case. Accumulators are not safe for concurrent mutation.
package metrics

import (
	"fmt"
	"math"
)

func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

func addCount(count, n int) (int, error) {
	if n < 0 || count > int(^uint(0)>>1)-n {
		return 0, fmt.Errorf("metrics: count overflow")
	}
	return count + n, nil
}

// Mean accumulates weighted means, such as a mean loss weighted by the number of
// valid samples/elements in each batch, rather than by the number of batches.
type Mean struct {
	sum   float64
	count int
}

// Add adds a mean value representing weight observations. Zero weight is a
// no-op, but value must still be finite. Errors leave the accumulator unchanged.
func (m *Mean) Add(value float64, weight int) error {
	if !finite(value) || weight < 0 {
		return fmt.Errorf("metrics: invalid mean value/weight")
	}
	count, err := addCount(m.count, weight)
	if err != nil {
		return err
	}
	sum := m.sum + value*float64(weight)
	if !finite(sum) {
		return fmt.Errorf("metrics: mean sum overflow")
	}
	m.sum, m.count = sum, count
	return nil
}

func (m *Mean) Count() int { return m.count }
func (m *Mean) Value() float64 {
	if m.count == 0 {
		return 0
	}
	return m.sum / float64(m.count)
}
func (m *Mean) Reset() { *m = Mean{} }

// Accuracy accumulates top-1 classification accuracy from row-major scores.
type Accuracy struct {
	correct, count int
}

// UpdateLogits accepts [len(labels), classes] scores. Scores need not be
// normalized; a tie selects the lowest class index. ignoreIndex labels are
// excluded from both counts. All scores, including ignored rows, must be finite.
// Errors leave the accumulator unchanged.
func (a *Accuracy) UpdateLogits(scores []float32, labels []int, classes, ignoreIndex int) error {
	if classes < 1 || len(scores)%classes != 0 || len(scores)/classes != len(labels) {
		return fmt.Errorf("metrics: scores/labels shape mismatch")
	}
	for _, score := range scores {
		if !finite(float64(score)) {
			return fmt.Errorf("metrics: non-finite classification score")
		}
	}
	correct, count := a.correct, a.count
	for i, label := range labels {
		if label == ignoreIndex {
			continue
		}
		if label < 0 || label >= classes {
			return fmt.Errorf("metrics: label %d out of range", label)
		}
		var err error
		count, err = addCount(count, 1)
		if err != nil {
			return err
		}
		row := scores[i*classes : (i+1)*classes]
		best := 0
		for class := 1; class < classes; class++ {
			if row[class] > row[best] {
				best = class
			}
		}
		if best == label {
			correct++
		}
	}
	a.correct, a.count = correct, count
	return nil
}

func (a *Accuracy) Count() int   { return a.count }
func (a *Accuracy) Correct() int { return a.correct }
func (a *Accuracy) Value() float64 {
	if a.count == 0 {
		return 0
	}
	return float64(a.correct) / float64(a.count)
}
func (a *Accuracy) Reset() { *a = Accuracy{} }

// Regression accumulates element-weighted MSE, RMSE and MAE. Each pair is
// converted to float64 before subtraction and summation. For multi-output
// models, Count is the number of elements, not the number of samples.
type Regression struct {
	squared, absolute float64
	count             int
}

// Update adds equally sized, finite prediction/target slices. Empty slices are a
// no-op. Errors leave the accumulator unchanged.
func (r *Regression) Update(prediction, target []float32) error {
	if len(prediction) != len(target) {
		return fmt.Errorf("metrics: prediction/target length mismatch")
	}
	count, err := addCount(r.count, len(prediction))
	if err != nil {
		return err
	}
	squared, absolute := r.squared, r.absolute
	for i, value := range prediction {
		if !finite(float64(value)) || !finite(float64(target[i])) {
			return fmt.Errorf("metrics: non-finite regression value")
		}
		difference := float64(value) - float64(target[i])
		squared += difference * difference
		absolute += math.Abs(difference)
	}
	if !finite(squared) || !finite(absolute) {
		return fmt.Errorf("metrics: regression sum overflow")
	}
	r.squared, r.absolute, r.count = squared, absolute, count
	return nil
}

func (r *Regression) Count() int { return r.count }
func (r *Regression) MSE() float64 {
	if r.count == 0 {
		return 0
	}
	return r.squared / float64(r.count)
}
func (r *Regression) RMSE() float64 { return math.Sqrt(r.MSE()) }
func (r *Regression) MAE() float64 {
	if r.count == 0 {
		return 0
	}
	return r.absolute / float64(r.count)
}
func (r *Regression) Reset() { *r = Regression{} }
