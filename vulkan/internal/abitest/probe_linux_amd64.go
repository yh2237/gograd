//go:build linux && !android && amd64

// Package abitest supplies a test-only native SysV target without a C compiler.
// It is imported only by the Linux Vulkan tests, never by the backend itself.
package abitest

// Address returns a native function with one output-pointer and nine uint64
// arguments. It hashes the integer arguments, writes the hash and entry SP%16
// to the output pointer, and also returns the hash in RAX.
func Address() uintptr
