package autograd

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// Keep this list explicit: adding a differentiable public Tensor operation
// must update the registry and pass the source-level dispatch check below.
var registeredPublicOps = map[string]string{
	"Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div", "Scalar": "scalar",
	"AddScalar": "add_scalar", "SubScalar": "sub_scalar", "MulScalar": "mul_scalar", "DivScalar": "div_scalar",
	"Exp": "exp", "Log": "log", "Abs": "abs", "Tanh": "tanh", "ReLU": "relu", "GELU": "gelu",
	"Reshape": "reshape", "Permute": "permute", "Transpose": "transpose", "Slice": "slice", "Expand": "expand", "Contiguous": "contiguous",
	"Sum": "sum", "Mean": "mean", "Concat": "concat", "MatMul": "matmul",
	"Embedding": "embedding", "EmbeddingWithPadding": "embedding_padding", "EmbeddingFromIndexBuffer": "embedding_index_buffer",
	"MaskedLoss": "masked_loss", "Conv1d": "conv1d", "Conv1dGEMM": "conv1d_gemm",
	"GroupNorm": "group_norm", "LayerNorm": "layer_norm", "Softmax": "softmax", "LogSoftmax": "log_softmax",
	"RMSNormLast": "rms_norm_last",
	"Dropout": "dropout", "ScaledDotProductAttention": "attention", "CrossEntropy": "cross_entropy",
	"BiasGELU": "bias_gelu", "BiasResidual": "bias_residual",
}

func TestDifferentiableOpsUseRegistry(t *testing.T) {
	for name, key := range registeredPublicOps {
		pair, ok := operationRegistry[key]
		if !ok || pair.CPU == "" || pair.CUDA == "" {
			t.Errorf("%s lacks CPU/CUDA registration", name)
		}
	}
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, entry := range entries {
		if !strings.HasSuffix(entry.Name(), ".go") || strings.HasSuffix(entry.Name(), "_test.go") {
			continue
		}
		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, filepath.Join(".", entry.Name()), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || !fn.Name.IsExported() || fn.Body == nil || fn.Type.Results == nil || len(fn.Type.Results.List) != 1 {
				continue
			}
			star, ok := fn.Type.Results.List[0].Type.(*ast.StarExpr)
			if !ok {
				continue
			}
			ident, ok := star.X.(*ast.Ident)
			if !ok || ident.Name != "Tensor" {
				continue
			}
			name := fn.Name.Name
			if name == "Forward" || name == "ForwardSeed" || name == "Param" || name == "Must" || name == "Detach" {
				continue
			}
			key, ok := registeredPublicOps[name]
			if !ok {
				t.Errorf("public Tensor operation %s in %s lacks registry entry", name, entry.Name())
				continue
			}
			seen[name] = true
			found := false
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok || len(call.Args) == 0 {
					return true
				}
				id, ok := call.Fun.(*ast.Ident)
				if !ok || (id.Name != "dispatchBackend" && id.Name != "dispatchElementwise") {
					return true
				}
				lit, ok := call.Args[0].(*ast.BasicLit)
				if !ok || lit.Kind != token.STRING {
					return true
				}
				value, err := strconv.Unquote(lit.Value)
				if err == nil && value == key {
					found = true
				}
				return true
			})
			// Softmax/LogSoftmax and Sum/Mean enter a shared helper, but each
			// public entry must still declare its registry key.
			if !found {
				t.Errorf("%s bypasses registry dispatch for %s", name, key)
			}
		}
	}
	for name := range registeredPublicOps {
		if !seen[name] {
			t.Errorf("registered public op %s was not found", name)
		}
	}
}
