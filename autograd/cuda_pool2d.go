package autograd

import (
	"sync"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
)

const pool2dSource = `
struct P2Spec { int h,w,oh,ow,kh,kw,sh,sw,ph,pw,dh,dw,exclude,divisor; };
__device__ void p2_axis(int pos,int input,int output,int kernel,int stride,int padding,int dilation,int kind,int exclude,
                       long long* start,long long* end,int* step,long long* area){
 *step=1;
 if(kind==2){
  *start=(long long)pos*input/output;
  *end=((long long)(pos+1)*input+output-1)/output;
 }else{
  *start=(long long)pos*stride-padding;
  *end=*start+(long long)(kernel-1)*dilation+1;
  if(kind==0){
   *step=dilation;
   if(*start<0)*start+=((-*start+dilation-1)/dilation)*dilation;
  }else{
   if(*end>(long long)input+padding)*end=(long long)input+padding;
   if(!exclude)*area*=*end-*start;
   if(*start<0)*start=0;
  }
  if(*end>input)*end=input;
 }
 if(kind==2||exclude)*area*=*end-*start;
}
__device__ float p2_window(P2Spec s,int pos,int kind,long long* hs,long long* he,long long* ws,long long* we,int* dh,int* dw){
 long long area=1;
 p2_axis(pos/s.ow,s.h,s.oh,s.kh,s.sh,s.ph,s.dh,kind,s.exclude,hs,he,dh,&area);
 p2_axis(pos%s.ow,s.w,s.ow,s.kw,s.sw,s.pw,s.dw,kind,s.exclude,ws,we,dw,&area);
 return (float)(s.divisor?s.divisor:area);
}
extern "C" __global__ void pool2d_f(const float* x,float* y,int* indices,P2Spec s,int n,int kind){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int plane=i/(s.oh*s.ow),pos=i%(s.oh*s.ow),dh,dw;
 long long hs,he,ws,we;
 float divisor=p2_window(s,pos,kind,&hs,&he,&ws,&we,&dh,&dw);
 float value=kind==0?-__int_as_float(0x7f800000):0;int index=-1;
 for(long long h=hs;h<he;h+=dh)for(long long w=ws;w<we;w+=dw){
  int p=(plane*s.h+(int)h)*s.w+(int)w;float v=x[p];
  if(kind==0){if(index<0||v>value||isnan(v)){value=v;index=p;}}else value+=v;
 }
 y[i]=kind==0?value:value/divisor;
 if(indices)indices[i]=index;
}
extern "C" __global__ void pool2d_b(const float* g,const int* indices,float* dx,P2Spec s,int n,int kind){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 if(kind==0){int p=indices[i];if(p>=0)atomicAdd(dx+p,g[i]);return;}
 int plane=i/(s.oh*s.ow),pos=i%(s.oh*s.ow),dh,dw;
 long long hs,he,ws,we;
 float value=g[i]/p2_window(s,pos,kind,&hs,&he,&ws,&we,&dh,&dw);
 for(long long h=hs;h<he;h+=dh)for(long long w=ws;w<we;w+=dw)
  atomicAdd(dx+(plane*s.h+(int)h)*s.w+(int)w,value);
}
`

var pool2dOnce sync.Once
var pool2dProgram *cuda.Program
var pool2dKernels map[string]*cuda.Kernel
var pool2dCompileErr error

func launchPool2d(name string, n int, args ...unsafe.Pointer) {
	pool2dOnce.Do(func() {
		pool2dProgram, pool2dCompileErr = cuda.Compile(pool2dSource)
		if pool2dCompileErr != nil {
			return
		}
		pool2dKernels = make(map[string]*cuda.Kernel)
		for _, key := range []string{"pool2d_f", "pool2d_b"} {
			pool2dKernels[key], pool2dCompileErr = pool2dProgram.Function(key)
			if pool2dCompileErr != nil {
				return
			}
		}
	})
	if pool2dCompileErr != nil {
		panic(pool2dCompileErr)
	}
	timedGPU(name, func() error {
		return pool2dKernels[name].Launch([3]int{(n + 255) / 256, 1, 1}, [3]int{256, 1, 1}, 0, nil, args)
	})
}

func gpuPool2d(x *Tensor, s pool2dSpec, kind int) *Tensor {
	n := numel(s.outputShape())
	out := mustAlloc(n)
	_, recording := graphRecording([]*Tensor{x})
	var indices *cuda.Buffer
	if kind == 0 && recording {
		indices = mustAlloc(n)
	}
	exclude := int32(0)
	if s.excludePad {
		exclude = 1
	}
	meta := [14]int32{int32(s.h), int32(s.w), int32(s.oh), int32(s.ow), int32(s.kernel[0]), int32(s.kernel[1]),
		int32(s.stride[0]), int32(s.stride[1]), int32(s.padding[0]), int32(s.padding[1]), int32(s.dilation[0]), int32(s.dilation[1]), exclude, int32(s.divisor)}
	xp, yp, ip := ptr(x.buf), ptr(out), ptr(indices)
	count, mode := int32(n), int32(kind)
	launchPool2d("pool2d_f", n, unsafe.Pointer(&xp), unsafe.Pointer(&yp), unsafe.Pointer(&ip), unsafe.Pointer(&meta), unsafe.Pointer(&count), unsafe.Pointer(&mode))
	r := resultGPU(out, s.outputShape(), []*Tensor{x}, func(g *cuda.Buffer) {
		if dx := x.ensureGradGPU(); dx != nil {
			gp, dp := ptr(g), ptr(dx)
			launchPool2d("pool2d_b", n, unsafe.Pointer(&gp), unsafe.Pointer(&ip), unsafe.Pointer(&dp), unsafe.Pointer(&meta), unsafe.Pointer(&count), unsafe.Pointer(&mode))
		}
	})
	if indices != nil {
		r.aux = []*cuda.Buffer{indices}
	}
	return r
}
