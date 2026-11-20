package autograd

import (
	"context"
	"encoding"
	"errors"
	"fmt"
	"io"
	"reflect"
)

// BatchReader reads one collated batch per call in sampler order. It returns
// io.EOF at the end of an epoch, and NextEpoch advances to the next one.
// Each newly started epoch must contain at least one batch.
// *data.Loader[T, B] satisfies this interface.
type BatchReader[B any] interface {
	Next(ctx context.Context) (B, error)
	NextEpoch() error
}

// PositionSampler saves and restores sampled positions as an opaque payload.
// To defer encoding, SnapshotPosition() encoding.BinaryMarshaler can return
// an independent snapshot of the sampler's concrete type.
type PositionSampler interface {
	MarshalBinary() ([]byte, error)
	UnmarshalBinary([]byte) error
}

func capturePosition(sampler PositionSampler) (func() ([]byte, error), error) {
	if source, ok := sampler.(interface {
		SnapshotPosition() encoding.BinaryMarshaler
	}); ok {
		snapshot := source.SnapshotPosition()
		// An embedding wrapper may have its own checkpoint format.
		if reflect.TypeOf(snapshot) == reflect.TypeOf(sampler) {
			return snapshot.MarshalBinary, nil
		}
	}
	payload, err := sampler.MarshalBinary()
	if err != nil {
		return nil, err
	}
	return func() ([]byte, error) { return payload, nil }, nil
}

// UpdateSchedule sets the learning rate after every optimizer update and
// reports how many updates it has applied. *OneCycle satisfies this interface.
type UpdateSchedule interface {
	Step(Optimizer)
	Updates() int
}

// Updates reports how many updates the schedule has applied. A schedule
// restored from a checkpoint counts those updates.
func (c *OneCycle) Updates() int { return c.StepCount }

// TrainingLoop groups micro-batches into optimizer updates. Cancellation
// discards unfinished gradients. When an unfinished group must be replayed,
// Save receives its starting sampler position. The caller handles forward-pass
// state such as running statistics when resuming.
type TrainingLoop[B any] struct {
	// Reader supplies collated batches in sampler order. Required.
	Reader BatchReader[B]
	// Sampler is required for accumulation and to checkpoint a canceled Step.
	Sampler PositionSampler
	// Optimizer owns the updated parameters. Required.
	Optimizer Optimizer
	// Schedule, when set, sets the learning rate after every update.
	Schedule UpdateSchedule
	// Updates is the total number of updates the run reaches, including the
	// Applied count a resumed run restores.
	Updates int
	// Applied counts the updates already applied, which a resumed run takes
	// from the restored schedule. Run increments it.
	Applied int
	// Accumulate is the number of micro-batches per update. Values below two
	// update after every batch.
	Accumulate int
	// LogEvery is the reporting cadence in updates. Zero means 25. The first
	// and the last update are always reported.
	LogEvery int
	// Step reads one micro-batch, builds its graph and back-propagates the
	// loss through accumulator. It owns the lifetime of the tensors it creates
	// and returns the loss value with the update completion flag, which
	// GradAccumulator.Backward reports.
	Step func(ctx context.Context, batch B, accumulator *GradAccumulator) (loss float64, complete bool, err error)
	// Report is called for the first, every LogEvery'th and the last update.
	Report func(update int, loss float64)
	// Save is called after the planned updates, after a canceled read, or when
	// Step returns a context error and a sampler position is available. A
	// non-nil savepoint replaces the reader's current position in the checkpoint.
	Save func(savepoint []byte) error
}

// Run applies updates until the planned count is reached or a read or step fails.
func (l *TrainingLoop[B]) Run(ctx context.Context) error {
	if l.Reader == nil || l.Optimizer == nil || l.Step == nil {
		return fmt.Errorf("autograd: training loop needs a reader, an optimizer and a step")
	}
	if l.Updates < 0 || l.Applied < 0 || l.Applied > l.Updates {
		return fmt.Errorf("autograd: invalid training loop update count")
	}
	groups := l.Accumulate
	if groups < 2 {
		groups = 1
	}
	if groups > 1 && l.Sampler == nil {
		return fmt.Errorf("autograd: accumulation rewinds need a position sampler")
	}
	accumulator, err := NewGradAccumulator(groups)
	if err != nil {
		return err
	}
	every := l.LogEvery
	if every < 1 {
		every = 25
	}
	// A resumed run reports its own first update, not the absolute first.
	start := l.Applied
	captureStart := groups > 1 || (l.Sampler != nil && l.Save != nil)
	var savepoint func() ([]byte, error)
	var stopped error
	awaitingFirstBatch := false
	for l.Applied < l.Updates {
		if accumulator.Pending() == 0 {
			if captureStart {
				savepoint, err = capturePosition(l.Sampler)
				if err != nil {
					return err
				}
			}
			l.Optimizer.ZeroGrad()
		}
		batch, err := l.Reader.Next(ctx)
		if errors.Is(err, io.EOF) {
			if awaitingFirstBatch {
				err = fmt.Errorf("autograd: training epoch contains no batches")
			} else {
				if err := l.Reader.NextEpoch(); err != nil {
					return err
				}
				awaitingFirstBatch = true
				continue
			}
		}
		if err != nil {
			if accumulator.Pending() != 0 {
				// The partial gradients are dropped, and the rewound position
				// replays those micro-batches after a resume.
				l.Optimizer.ZeroGrad()
				accumulator.Reset()
			} else {
				savepoint = nil
			}
			stopped = err
			break
		}
		awaitingFirstBatch = false
		loss, complete, err := l.Step(ctx, batch, accumulator)
		if err != nil {
			if !isTrainingCancellation(err) {
				return err
			}
			l.Optimizer.ZeroGrad()
			accumulator.Reset()
			if savepoint == nil {
				return err
			}
			stopped = err
			break
		}
		if !complete {
			continue
		}
		// A completed group leaves nothing to rewind.
		savepoint = nil
		l.Applied++
		l.Optimizer.Step()
		if l.Schedule != nil {
			l.Schedule.Step(l.Optimizer)
		}
		if l.Applied == start+1 || l.Applied%every == 0 || l.Applied == l.Updates {
			if l.Report != nil {
				l.Report(l.Applied, loss)
			}
		}
	}
	if stopped == nil {
		if l.Save == nil {
			return nil
		}
		return l.Save(nil)
	}
	if !isTrainingCancellation(stopped) {
		return stopped
	}
	if l.Save != nil {
		var rewind []byte
		if savepoint != nil {
			rewind, err = savepoint()
			if err != nil {
				return errors.Join(stopped, err)
			}
		}
		if err := l.Save(rewind); err != nil {
			return errors.Join(stopped, err)
		}
	}
	return stopped
}

func isTrainingCancellation(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}
