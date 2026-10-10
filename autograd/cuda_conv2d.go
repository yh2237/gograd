package autograd

import (
	"sync"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

const conv2dSource = `
struct C2Spec { int ci,h,w,co,kh,kw,oh,ow,sh,sw,ph,pw,dh,dw,groups; };
extern "C" __global__ void c2_columns(const float* x,float* col,C2Spec s,int batch,int group,int start,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x,ci=s.ci/s.groups,depth=ci*s.kh*s.kw;
 if(i>=depth*count)return;
 int d=i/count,j=i%count,p=start+j,tap=d%(s.kh*s.kw),ch=group*ci+d/(s.kh*s.kw);
 long long y=(long long)(p/s.ow)*s.sh-s.ph+(long long)(tap/s.kw)*s.dh;
 long long z=(long long)(p%s.ow)*s.sw-s.pw+(long long)(tap%s.kw)*s.dw;
 col[i]=(y>=0&&y<s.h&&z>=0&&z<s.w)?x[((batch*s.ci+ch)*s.h+(int)y)*s.w+(int)z]:0;
}
extern "C" __global__ void c2_col2im(const float* col,float* dx,C2Spec s,int batch,int group,int start,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x,ci=s.ci/s.groups,depth=ci*s.kh*s.kw;
 if(i>=depth*count)return;
 int d=i/count,j=i%count,p=start+j,tap=d%(s.kh*s.kw),ch=group*ci+d/(s.kh*s.kw);
 long long y=(long long)(p/s.ow)*s.sh-s.ph+(long long)(tap/s.kw)*s.dh;
 long long z=(long long)(p%s.ow)*s.sw-s.pw+(long long)(tap%s.kw)*s.dw;
 if(y>=0&&y<s.h&&z>=0&&z<s.w)atomicAdd(dx+((batch*s.ci+ch)*s.h+(int)y)*s.w+(int)z,col[i]);
}
extern "C" __global__ void c2_output(const float* tile,const float* b,float* out,C2Spec s,int batch,int group,int start,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x,co=s.co/s.groups;
 if(i>=co*count)return;
 int ch=group*co+i/count,p=start+i%count;
 out[(batch*s.co+ch)*s.oh*s.ow+p]=tile[i]+(b?b[ch]:0);
}
extern "C" __global__ void c2_upstream(const float* g,float* tile,C2Spec s,int batch,int group,int start,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x,co=s.co/s.groups;
 if(i>=co*count)return;
 tile[i]=g[(batch*s.co+group*co+i/count)*s.oh*s.ow+start+i%count];
}
extern "C" __global__ void c2_bias_grad(const float* g,float* db,C2Spec s,int batch){
 int ch=blockIdx.x*blockDim.x+threadIdx.x;if(ch>=s.co)return;
 float sum=0;for(int n=0;n<batch;n++)for(int p=0;p<s.oh*s.ow;p++)sum+=g[(n*s.co+ch)*s.oh*s.ow+p];
 db[ch]+=sum;
}
`

var conv2dOnce sync.Once
var conv2dProgram *cuda.Program
var conv2dKernels map[string]*cuda.Kernel
var conv2dCompileErr error

func launchConv2d(name string, n int, args ...unsafe.Pointer) {
	conv2dOnce.Do(func() {
		conv2dProgram, conv2dCompileErr = cuda.Compile(conv2dSource)
		if conv2dCompileErr != nil {
			return
		}
		conv2dKernels = make(map[string]*cuda.Kernel)
		for _, key := range []string{"c2_columns", "c2_col2im", "c2_output", "c2_upstream", "c2_bias_grad"} {
			conv2dKernels[key], conv2dCompileErr = conv2dProgram.Function(key)
			if conv2dCompileErr != nil {
				return
			}
		}
	})
	if conv2dCompileErr != nil {
		panic(conv2dCompileErr)
	}
	timedGPU(name, func() error {
		return conv2dKernels[name].Launch([3]int{(n + 255) / 256, 1, 1}, [3]int{256, 1, 1}, 0, nil, args)
	})
}

func (s conv2dSpec) gpuMeta() [15]int32 {
	return [15]int32{int32(s.ci), int32(s.h), int32(s.width), int32(s.co), int32(s.kh), int32(s.kw), int32(s.oh), int32(s.ow),
		int32(s.options.Stride[0]), int32(s.options.Stride[1]), int32(s.options.Padding[0]), int32(s.options.Padding[1]),
		int32(s.options.Dilation[0]), int32(s.options.Dilation[1]), int32(s.options.Groups)}
}

func conv2dTileLaunch(name string, size int, source, dest uintptr, meta [15]int32, batch, group, start, count int) {
	n, gr, st, ct := int32(batch), int32(group), int32(start), int32(count)
	launchConv2d(name, size, unsafe.Pointer(&source), unsafe.Pointer(&dest), unsafe.Pointer(&meta),
		unsafe.Pointer(&n), unsafe.Pointer(&gr), unsafe.Pointer(&st), unsafe.Pointer(&ct))
}

func gpuConv2d(x, w, b *Tensor, s conv2dSpec, parents []*Tensor) *Tensor {
	depth, co, spatial, tile := s.depth(), s.outChannels(), s.oh*s.ow, s.tileColumns()
	meta := s.gpuMeta()
	cols, values := mustAlloc(depth*tile), mustAlloc(co*tile)
	defer cols.Free()
	defer values.Free()
	out := mustAlloc(numel(s.outputShape()))
	var bp uintptr
	if b != nil {
		bp = ptr(b.buf)
	}
	for n := 0; n < s.batch; n++ {
		for group := 0; group < s.options.Groups; group++ {
			wp := ptr(w.buf) + uintptr(group*co*depth*4)
			for start := 0; start < spatial; start += tile {
				count := min(tile, spatial-start)
				conv2dTileLaunch("c2_columns", depth*count, ptr(x.buf), ptr(cols), meta, n, group, start, count)
				// Conv2d is FP32 even when BF16 shadows are enabled elsewhere.
				timedGPU("conv2d_forward_sgemm", func() error {
					return blas().SgemmRowMajor(co, count, depth, 1, wp, depth, ptr(cols), count, 0, ptr(values), count)
				})
				vp, yp := ptr(values), ptr(out)
				nn, gr, st, ct := int32(n), int32(group), int32(start), int32(count)
				launchConv2d("c2_output", co*count, unsafe.Pointer(&vp), unsafe.Pointer(&bp), unsafe.Pointer(&yp), unsafe.Pointer(&meta),
					unsafe.Pointer(&nn), unsafe.Pointer(&gr), unsafe.Pointer(&st), unsafe.Pointer(&ct))
			}
		}
	}
	return resultGPU(out, s.outputShape(), parents, func(g *cuda.Buffer) {
		dx, dw := x.ensureGradGPU(), w.ensureGradGPU()
		if b != nil {
			if db := b.ensureGradGPU(); db != nil {
				gp, dp, batch := ptr(g), ptr(db), int32(s.batch)
				launchConv2d("c2_bias_grad", s.co, unsafe.Pointer(&gp), unsafe.Pointer(&dp), unsafe.Pointer(&meta), unsafe.Pointer(&batch))
			}
		}
		if dx == nil && dw == nil {
			return
		}
		cols, upstream := mustAlloc(depth*tile), mustAlloc(co*tile)
		defer cols.Free()
		defer upstream.Free()
		for n := 0; n < s.batch; n++ {
			for group := 0; group < s.options.Groups; group++ {
				wp := ptr(w.buf) + uintptr(group*co*depth*4)
				for start := 0; start < spatial; start += tile {
					count := min(tile, spatial-start)
					conv2dTileLaunch("c2_upstream", co*count, ptr(g), ptr(upstream), meta, n, group, start, count)
					if dw != nil {
						conv2dTileLaunch("c2_columns", depth*count, ptr(x.buf), ptr(cols), meta, n, group, start, count)
						dwp := ptr(dw) + uintptr(group*co*depth*4)
						timedGPU("conv2d_dw_sgemm", func() error {
							return blas().SgemmRowMajorNT(co, depth, count, 1, ptr(upstream), count, ptr(cols), count, 1, dwp, depth)
						})
					}
					if dx != nil {
						timedGPU("conv2d_dx_sgemm", func() error {
							return blas().SgemmRowMajorTransposeA(depth, count, co, 1, wp, depth, ptr(upstream), count, 0, ptr(cols), count)
						})
						conv2dTileLaunch("c2_col2im", depth*count, ptr(cols), ptr(dx), meta, n, group, start, count)
					}
				}
			}
		}
	})
}
