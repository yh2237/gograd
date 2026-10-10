package autograd

import "github.com/yh2237/gograd/tensor"

// KernelPair names the CPU implementation and the CUDA kernel or implementation
// used by one differentiable operation. Device selection is centralized here;
// shape checks and arguments remain with the public operation.
type KernelPair struct{ CPU, CUDA string }

var operationRegistry = map[string]KernelPair{
	"add": {"binary", "binary_f/b"}, "sub": {"binary", "binary_f/b"},
	"mul": {"binary", "binary_f/b"}, "div": {"binary", "binary_f/b"},
	"scalar":     {"New", "New"},
	"add_scalar": {"cpuScalar", "scalar_f/b"}, "sub_scalar": {"cpuScalar", "scalar_f/b"},
	"mul_scalar": {"cpuScalar", "scalar_f/b"}, "div_scalar": {"cpuScalar", "scalar_f/b"},
	"exp": {"unary", "unary_f/b"}, "log": {"unary", "unary_f/b"},
	"abs": {"unary", "unary_f/b"}, "tanh": {"unary", "unary_f/b"},
	"relu": {"unary", "unary_f/b"}, "gelu": {"unary", "unary_f/b"},
	"reshape":    {"makeView", "view_gather/scatter"},
	"permute":    {"makeView", "view_gather/scatter"},
	"transpose":  {"makeView", "view_gather/scatter"},
	"slice":      {"makeView", "view_gather/scatter"},
	"expand":     {"makeView", "view_gather/scatter"},
	"contiguous": {"storageIndex", "view_gather/scatter"},
	"sum":        {"reduce", "reduce_nd_f/b"}, "mean": {"reduce", "reduce_nd_f/b"},
	"concat":                 {"concat", "concat_f/b"},
	"matmul":                 {"SGEMMOp", "cublasSgemmStridedBatched"},
	"embedding":              {"scatter-add", "embedding_f/b"},
	"embedding_padding":      {"scatter-add", "embedding_f/b"},
	"embedding_index_buffer": {"scatter-add", "embedding_f/b"},
	"masked_loss":            {"masked-loss", "masked_f/b"},
	"conv1d":                 {"Conv1d reference", "im2col/col2im+cuBLAS"},
	"conv1d_gemm":            {"SGEMMOp", "im2col/col2im+cuBLAS"},
	"conv2d":                 {"tiled im2col+SGEMMOp", "tiled im2col/col2im+cuBLAS"},
	"max_pool2d":             {"NCHW max windows", "pool2d_f/b"},
	"avg_pool2d":             {"NCHW average windows", "pool2d_f/b"},
	"adaptive_avg_pool2d":    {"adaptive NCHW bins", "pool2d_f/b"},
	"group_norm":             {"GroupNorm", "group_f/b"},
	"batch_norm":             {"channel batch/running statistics", "bn_stats/f/grad_stats/dx"},
	"layer_norm":             {"LayerNorm", "layernorm_f/b"},
	"rms_norm_last":          {"Mean+Log+Exp+Mul", "reduce_nd+unary+binary"},
	"rotary_half":            {"Slice+Mul+Concat", "view+binary+concat"},
	"geglu":                  {"Slice+GELU+Mul", "view+unary+binary"},
	"swiglu_projection":      {"MatMul+Exp+Mul", "cuBLAS+unary+binary"},
	"key_padding_attention":  {"ScaledDotProductAttention", "attention"},
	"softmax":                {"softmax", "softmax_f/b"},
	"log_softmax":            {"softmax", "softmax_f/b"},
	"dropout":                {"Dropout", "dropout_f/b"},
	"attention":              {"MatMul+Softmax", "attn_qk/pv+softmax or flash_attention_tiled_f/b"},
	"cross_entropy":          {"CrossEntropy", "ce_f/b"},
	"bias_gelu":              {"Add+GELU", "bias_gelu_f/b"},
	"bias_residual":          {"Add", "bias_residual_f/b"},
}

// dispatchBackend validates registration and selects the implementation.
// The bool is true for CUDA, false for CPU.
func dispatchBackend(name string, device tensor.Device) bool {
	k, ok := operationRegistry[name]
	if !ok || k.CPU == "" || k.CUDA == "" {
		panic("autograd: unregistered operation " + name)
	}
	switch device {
	case tensor.CPU:
		return false
	case tensor.CUDA:
		return true
	default:
		panic("autograd: unsupported device")
	}
}

// ElementwiseKernel pairs the CPU scalar rule and CUDA kernel opcode. The
// registry is the dispatch point for elementwise forward and backward rules.
type ElementwiseKernel struct {
	CPUForward, CPUGradA, CPUGradB func(float32, float32) float32
	CUDAOpcode                     int
}

var elementwiseRegistry = map[string]ElementwiseKernel{
	"add": {func(x, y float32) float32 { return x + y }, func(x, y float32) float32 { return 1 }, func(x, y float32) float32 { return 1 }, 0},
	"sub": {func(x, y float32) float32 { return x - y }, func(x, y float32) float32 { return 1 }, func(x, y float32) float32 { return -1 }, 1},
	"mul": {func(x, y float32) float32 { return x * y }, func(x, y float32) float32 { return y }, func(x, y float32) float32 { return x }, 2},
	"div": {func(x, y float32) float32 { return x / y }, func(x, y float32) float32 { return 1 / y }, func(x, y float32) float32 { return -x / (y * y) }, 3},
}

func dispatchElementwise(name string, a, b *Tensor) *Tensor {
	spec, ok := elementwiseRegistry[name]
	if !ok {
		panic("autograd: unregistered operation " + name)
	}
	same(a, b)
	a, b = a.Contiguous(), b.Contiguous()
	if dispatchBackend(name, a.Device) {
		return gpuBinary(a, b, spec.CUDAOpcode)
	}
	return cpuBinaryOp(a, b, spec.CUDAOpcode)
}

// RegisteredElementwiseOps lists operations handled by this dispatch table.
func RegisteredElementwiseOps() []string { return []string{"add", "sub", "mul", "div"} }
