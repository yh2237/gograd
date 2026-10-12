package autograd

import (
	"fmt"

	"github.com/yh2237/gograd/cuda"
)

// StreamBranch is one independent part of a step that runs on its own stream.
// Run enqueues the branch from the calling OS thread; use the same tensors and
// model state the surrounding step uses, and keep independent branches free of
// data dependencies on each other.
type StreamBranch struct {
	// Stream is the stream the branch is enqueued on. It must be non-nil; a
	// caller that wants the default stream simply does not split the work.
	Stream *cuda.Stream
	// Run enqueues the branch. Returning an error stops the split and is
	// reported by RunBranches.
	Run func() error
}

// RunBranches enqueues each branch on its own stream from the calling OS thread,
// then makes that thread's current stream wait for every branch. Afterwards the
// caller can consume the branches' results on its own stream without naming
// events: the waits are recorded for it. The branches' buffers stay valid
// because the pool caches them per stream, so another stream cannot reuse them
// while a branch is still reading them.
//
// RunBranches does not decide how to split work; the caller names the branches.
// It does not synchronize the host, so enqueued work may still be running when
// it returns. Branches run on the calling thread, so one branch must not wait
// for another inside Run.
//
// Branch descriptions are validated before any work is enqueued. If a branch
// fails, the branches enqueued before it are still joined into the caller's
// stream before the error is returned, so buffers freed later on that stream are
// ordered after their work.
func RunBranches(branches []StreamBranch) error {
	if len(branches) == 0 {
		return fmt.Errorf("autograd: RunBranches needs at least one branch")
	}
	for i, branch := range branches {
		if branch.Stream == nil {
			return fmt.Errorf("autograd: branch %d has no stream", i)
		}
		if branch.Run == nil {
			return fmt.Errorf("autograd: branch %d has no function", i)
		}
	}
	target := cuda.CurrentStream()
	var firstErr error
	for _, branch := range branches {
		if firstErr == nil {
			if err := cuda.WithStream(branch.Stream, branch.Run); err != nil {
				firstErr = err
			}
		}
		event, err := cuda.NewEvent(false)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			break
		}
		if err := event.Record(branch.Stream); err != nil {
			event.Destroy()
			if firstErr == nil {
				firstErr = err
			}
			break
		}
		if err := event.WaitOn(target); err != nil {
			event.Destroy()
			if firstErr == nil {
				firstErr = err
			}
			break
		}
		event.Destroy()
	}
	return firstErr
}
