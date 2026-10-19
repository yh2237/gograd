//go:build (windows || (linux && !android)) && amd64

package vulkan

import "testing"

// TestAuditVulkanDeviceScoring checks that a discrete GPU outranks an
// integrated one. The Vulkan device types are 1 for integrated and 2 for
// discrete, so the ranking has to follow that order.
func TestAuditVulkanDeviceScoring(t *testing.T) {
	const (
		integrated uint32 = 1
		discrete   uint32 = 2
		cpu        uint32 = 4
	)
	const computeOnly uint32 = 2
	const combined uint32 = 3
	if deviceScore(discrete, computeOnly) <= deviceScore(integrated, computeOnly) {
		t.Fatalf("discrete %d does not outrank integrated %d", deviceScore(discrete, computeOnly), deviceScore(integrated, computeOnly))
	}
	if deviceScore(discrete, computeOnly) <= deviceScore(discrete, combined) {
		t.Fatal("a compute-only family does not outrank a combined one")
	}
	if deviceScore(cpu, computeOnly) >= deviceScore(integrated, computeOnly) {
		t.Fatal("a CPU device is not ranked below an integrated GPU")
	}
}
