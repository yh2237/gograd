package metrics

import (
	"fmt"
	"math"
)

// ConfusionMatrix accumulates a classes x classes count table. Rows are actual
// labels and columns are predicted labels.
type ConfusionMatrix struct {
	classes int
	counts  []int
	count   int
}

// Update adds paired predicted/actual labels. ignoreIndex pairs are excluded from
// both the table and Count. The class count is fixed by the first successful
// update. Errors leave the accumulator unchanged.
func (c *ConfusionMatrix) Update(predicted, actual []int, classes, ignoreIndex int) error {
	if classes < 1 {
		return fmt.Errorf("metrics: classes must be positive")
	}
	if classes > math.MaxInt/classes {
		return fmt.Errorf("metrics: class count overflows the table size")
	}
	if len(predicted) != len(actual) {
		return fmt.Errorf("metrics: predicted/actual length mismatch")
	}
	if c.classes != 0 && c.classes != classes {
		return fmt.Errorf("metrics: classes changed from %d to %d", c.classes, classes)
	}
	count := c.count
	for i, label := range actual {
		if label == ignoreIndex {
			continue
		}
		if label < 0 || label >= classes {
			return fmt.Errorf("metrics: label %d out of range", label)
		}
		if p := predicted[i]; p < 0 || p >= classes {
			return fmt.Errorf("metrics: prediction %d out of range", p)
		}
		var err error
		if count, err = addCount(count, 1); err != nil {
			return err
		}
	}
	if c.classes == 0 {
		c.classes = classes
		c.counts = make([]int, classes*classes)
	}
	for i, label := range actual {
		if label == ignoreIndex {
			continue
		}
		c.counts[label*classes+predicted[i]]++
	}
	c.count = count
	return nil
}

func (c *ConfusionMatrix) Classes() int { return c.classes }
func (c *ConfusionMatrix) Count() int   { return c.count }

// Matrix returns a copy of the table. Rows are actual labels.
func (c *ConfusionMatrix) Matrix() [][]int {
	if c.classes == 0 {
		return nil
	}
	out := make([][]int, c.classes)
	for i := range out {
		out[i] = append([]int(nil), c.counts[i*c.classes:(i+1)*c.classes]...)
	}
	return out
}

// Support is the number of actual labels for class.
func (c *ConfusionMatrix) Support(class int) int {
	if class < 0 || class >= c.classes {
		return 0
	}
	var total int
	for predicted := 0; predicted < c.classes; predicted++ {
		total += c.counts[class*c.classes+predicted]
	}
	return total
}

// Precision is the fraction of predicted class labels that are correct.
func (c *ConfusionMatrix) Precision(class int) float64 {
	if class < 0 || class >= c.classes {
		return 0
	}
	var predicted int
	for actual := 0; actual < c.classes; actual++ {
		predicted += c.counts[actual*c.classes+class]
	}
	if predicted == 0 {
		return 0
	}
	return float64(c.counts[class*c.classes+class]) / float64(predicted)
}

// Recall is the fraction of actual class labels that are predicted correctly.
func (c *ConfusionMatrix) Recall(class int) float64 {
	support := c.Support(class)
	if support == 0 {
		return 0
	}
	return float64(c.counts[class*c.classes+class]) / float64(support)
}

// F1 is the harmonic mean of Precision and Recall.
func (c *ConfusionMatrix) F1(class int) float64 {
	precision, recall := c.Precision(class), c.Recall(class)
	if precision+recall == 0 {
		return 0
	}
	return 2 * precision * recall / (precision + recall)
}

// MacroF1 averages F1 over classes with at least one actual label.
func (c *ConfusionMatrix) MacroF1() float64 {
	var sum float64
	var n int
	for class := 0; class < c.classes; class++ {
		if c.Support(class) == 0 {
			continue
		}
		sum += c.F1(class)
		n++
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

func (c *ConfusionMatrix) Reset() { *c = ConfusionMatrix{} }
