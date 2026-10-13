package autograd

import (
	"context"
	"errors"
	"fmt"
	"io"
)

// BatchReader reads one collated batch per call in sampler order. It returns
// io.EOF at the end of an epoch, and NextEpoch advances to the next one.
// *data.Loader[T, B] satisfies this interface.
type BatchReader[B any] interface {
	Next(ctx context.Context) (B, error)
	NextEpoch() error
}

// PositionSampler saves and restores sampled positions as an opaque payload.
// *data.IndexSampler satisfies this interface.
type PositionSampler interface {
	MarshalBinary() ([]byte, error)
	UnmarshalBinary([]byte) error
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

// TrainingLoop applies optimizer updates to the batches of a BatchReader.
//
// The caller builds and back-propagates every micro-batch in Step. The loop
// owns the update boundary instead: zeroing gradients at the start of an update
// or of an accumulation group, stepping the optimizer and the schedule once per
// update, reporting the loss at a fixed cadence, and saving state when the run
// stops.
//
// If ctx ends in the middle of an accumulation group, the loop drops the
// pending gradients and rewinds the sampler to the group's first micro-batch,
// so the model, optimizer and schedule keep describing the last completed
// update. Save receives that rewound payload. Only the sampler is rewound;
// forward passes that change other state, such as running statistics or
// counters, must be restored by the caller.
type TrainingLoop[B any] struct {
	// Reader supplies collated batches in sampler order. Required.
	Reader BatchReader[B]
	// Sampler is required for accumulation, which rewinds it when a group is
	// interrupted.
	Sampler PositionSampler
	// Optimizer owns the updated parameters. Required.
	Optimizer Optimizer
	// Schedule, when set, sets the learning rate after every update.
	Schedule UpdateSchedule
	// Updates is the number of updates this run applies.
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
	// Save is called once when the run stops after the planned updates or when
	// ctx ends. savepoint rewinds a partially accumulated group and is nil
	// otherwise. A read or step error stops the run without calling Save.
	Save func(savepoint []byte) error
}

// Run applies updates until the planned count is reached, the context ends or
// a read/step fails. When the context ends, it saves the state of the last
// completed update through Save and returns the context error.
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
	var savepoint []byte
	var rewind []byte
	var stopped error
	for l.Applied < l.Updates {
		if accumulator.Pending() == 0 {
			if groups > 1 {
				savepoint, err = l.Sampler.MarshalBinary()
				if err != nil {
					return err
				}
			}
			l.Optimizer.ZeroGrad()
		}
		batch, err := l.Reader.Next(ctx)
		if errors.Is(err, io.EOF) {
			if err := l.Reader.NextEpoch(); err != nil {
				return err
			}
			continue
		}
		if err != nil {
			if accumulator.Pending() != 0 {
				// The partial gradients are dropped, and the rewound position
				// replays those micro-batches after a resume.
				l.Optimizer.ZeroGrad()
				accumulator.Reset()
				rewind = savepoint
			}
			stopped = err
			break
		}
		loss, complete, err := l.Step(ctx, batch, accumulator)
		if err != nil {
			return err
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
		if l.Applied == 1 || l.Applied%every == 0 || l.Applied == l.Updates {
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
	if !errors.Is(stopped, context.Canceled) && !errors.Is(stopped, context.DeadlineExceeded) {
		return stopped
	}
	if l.Save != nil {
		if err := l.Save(rewind); err != nil {
			return errors.Join(stopped, err)
		}
	}
	return stopped
}
