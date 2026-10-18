package vulkan

import _ "embed"

// Rebuilding shaders is a developer-only action. glslang is not needed for
// go build/go test or by applications using this package.
//go:generate glslang -V --target-env vulkan1.1 -o shaders/pointwise.spv shaders/pointwise.comp
//go:generate glslang -V --target-env vulkan1.1 -o shaders/view.spv shaders/view.comp
//go:generate glslang -V --target-env vulkan1.1 -o shaders/gemm.spv shaders/gemm.comp
//go:generate glslang -V --target-env vulkan1.1 -o shaders/reduce.spv shaders/reduce.comp
//go:generate glslang -V --target-env vulkan1.1 -o shaders/conv.spv shaders/conv.comp

//go:embed shaders/pointwise.spv
var pointwiseSPV []byte

//go:embed shaders/view.spv
var viewSPV []byte

//go:embed shaders/gemm.spv
var gemmSPV []byte

//go:embed shaders/reduce.spv
var reduceSPV []byte

//go:embed shaders/conv.spv
var convSPV []byte
