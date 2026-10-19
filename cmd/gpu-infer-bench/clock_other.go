//go:build !windows

package main

import "time"

var clockEpoch = time.Now()

func clockNow() int64               { return time.Since(clockEpoch).Nanoseconds() }
func elapsedMS(start int64) float64 { return float64(clockNow()-start) / 1e6 }
