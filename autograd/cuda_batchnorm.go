package autograd

import (
	"sync"
	"unsafe"

	"github.com/yh2237/gograd/cuda"
	"github.com/yh2237/gograd/tensor"
)

const batchNormSource = `
__device__ int bn_index(int sample,int ch,int channels,int inner){return (sample/inner*channels+ch)*inner+sample%inner;}
extern "C" __global__ void bn_counter(float* count){if(threadIdx.x==0&&blockIdx.x==0)count[0]+=1;}
extern "C" __global__ void bn_stats(const float* x,float* mean,float* variance,const float* count,float* stats,
 int channels,int inner,int samples,int training,int cumulative,float eps,float momentum){
 int ch=blockIdx.x,t=threadIdx.x;
 __shared__ double scratch[256];__shared__ double center;
 if(!training){if(t==0){stats[ch]=mean[ch];stats[channels+ch]=rsqrtf(variance[ch]+eps);}return;}
 double sum=0;
 for(long long sample=t;sample<samples;sample+=256)sum+=(double)x[bn_index((int)sample,ch,channels,inner)];
 scratch[t]=sum;__syncthreads();
 for(int stride=128;stride;stride>>=1){if(t<stride)scratch[t]+=scratch[t+stride];__syncthreads();}
 if(t==0)center=scratch[0]/samples;__syncthreads();
 double squares=0;
 for(long long sample=t;sample<samples;sample+=256){double d=(double)x[bn_index((int)sample,ch,channels,inner)]-center;squares+=d*d;}
 scratch[t]=squares;__syncthreads();
 for(int stride=128;stride;stride>>=1){if(t<stride)scratch[t]+=scratch[t+stride];__syncthreads();}
 if(t==0){
  float mu=(float)center,v=(float)(scratch[0]/samples);stats[ch]=mu;stats[channels+ch]=rsqrtf(v+eps);
  if(mean){float m=cumulative?1/count[0]:momentum;mean[ch]=(1-m)*mean[ch]+m*mu;variance[ch]=(1-m)*variance[ch]+m*(v*(float)samples/(samples-1));}
 }
}
extern "C" __global__ void bn_f(const float* x,const float* weight,const float* bias,const float* stats,float* out,int n,int channels,int inner){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int ch=i/inner%channels;
 float norm=(x[i]-stats[ch])*stats[channels+ch];out[i]=norm*(weight?weight[ch]:1)+(bias?bias[ch]:0);
}
extern "C" __global__ void bn_grad_stats(const float* x,const float* g,const float* stats,float* sums,float* dw,float* db,int channels,int inner,int samples){
 int ch=blockIdx.x,t=threadIdx.x;__shared__ double a[256],b[256];double sg=0,sgn=0;
 for(long long sample=t;sample<samples;sample+=256){int i=bn_index((int)sample,ch,channels,inner);float norm=(x[i]-stats[ch])*stats[channels+ch];sg+=g[i];sgn+=(double)g[i]*norm;}
 a[t]=sg;b[t]=sgn;__syncthreads();
 for(int stride=128;stride;stride>>=1){if(t<stride){a[t]+=a[t+stride];b[t]+=b[t+stride];}__syncthreads();}
 if(t==0){sums[ch]=(float)(a[0]/samples);sums[channels+ch]=(float)(b[0]/samples);if(dw)dw[ch]+=(float)b[0];if(db)db[ch]+=(float)a[0];}
}
extern "C" __global__ void bn_dx(const float* x,const float* g,const float* weight,const float* stats,const float* sums,float* dx,int n,int channels,int inner,int training){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int ch=i/inner%channels;float value=g[i],inv=stats[channels+ch];
 if(training)value-=sums[ch]+(x[i]-stats[ch])*inv*sums[channels+ch];
 dx[i]+=value*(weight?weight[ch]:1)*inv;
}
`

var batchNormOnce sync.Once
var batchNormProgram *cuda.Program
var batchNormKernels map[string]*cuda.Kernel
var batchNormCompileErr error

func launchBatchNorm(name string, grid, block int, args ...unsafe.Pointer) {
	batchNormOnce.Do(func() {
		batchNormProgram, batchNormCompileErr = cuda.Compile(batchNormSource)
		if batchNormCompileErr != nil {
			return
		}
		batchNormKernels = make(map[string]*cuda.Kernel)
		for _, key := range []string{"bn_counter", "bn_stats", "bn_f", "bn_grad_stats", "bn_dx"} {
			batchNormKernels[key], batchNormCompileErr = batchNormProgram.Function(key)
			if batchNormCompileErr != nil {
				return
			}
		}
	})
	if batchNormCompileErr != nil {
		panic(batchNormCompileErr)
	}
	timedGPU(name, func() error {
		return batchNormKernels[name].Launch([3]int{grid, 1, 1}, [3]int{block, 1, 1}, 0, nil, args)
	})
}

func gpuBatchNorm(x, w, b, mean, variance, count *Tensor, o BatchNormOptions, s batchNormSpec, cumulative bool, context *ExecutionContext, parents []*Tensor) *Tensor {
	stats, out := mustAlloc(2*s.channels), mustAlloc(x.Numel())
	var wp, bp, mp, vp, cp uintptr
	if w != nil {
		wp = ptr(w.buf)
	}
	if b != nil {
		bp = ptr(b.buf)
	}
	if mean != nil {
		mp, vp = ptr(mean.buf), ptr(variance.buf)
	}
	if count != nil {
		cp = ptr(count.buf)
	}
	xp, sp, yp := ptr(x.buf), ptr(stats), ptr(out)
	channels, inner, samples, n := int32(s.channels), int32(s.inner), int32(s.samples), int32(x.Numel())
	training, cum := int32(0), int32(0)
	if o.Training {
		training = 1
	}
	if cumulative {
		cum = 1
	}
	if o.Training && count != nil {
		launchBatchNorm("bn_counter", 1, 1, unsafe.Pointer(&cp))
		count.storage.version.Add(1)
	}
	launchBatchNorm("bn_stats", s.channels, 256, unsafe.Pointer(&xp), unsafe.Pointer(&mp), unsafe.Pointer(&vp), unsafe.Pointer(&cp), unsafe.Pointer(&sp),
		unsafe.Pointer(&channels), unsafe.Pointer(&inner), unsafe.Pointer(&samples), unsafe.Pointer(&training), unsafe.Pointer(&cum), unsafe.Pointer(&o.Eps), unsafe.Pointer(&o.Momentum))
	if o.Training && mean != nil {
		mean.invalidateBF16()
		variance.invalidateBF16()
	}
	launchBatchNorm("bn_f", (x.Numel()+255)/256, 256, unsafe.Pointer(&xp), unsafe.Pointer(&wp), unsafe.Pointer(&bp), unsafe.Pointer(&sp), unsafe.Pointer(&yp), unsafe.Pointer(&n), unsafe.Pointer(&channels), unsafe.Pointer(&inner))
	saved := &Tensor{buf: stats, storage: deviceStorage(stats), Shape: []int{2, s.channels}, Strides: []int{s.channels, 1}, DType: Float32, Device: tensor.CUDA, execution: context, ephemeral: true}
	return resultGPUWithSaved(out, x.Shape, parents, []*Tensor{saved}, func(g *cuda.Buffer) {
		sums := mustAlloc(2 * s.channels)
		defer sums.Free()
		gp, sumPtr := ptr(g), ptr(sums)
		var dw, db uintptr
		if w != nil {
			dw = ptr(w.ensureGradGPU())
		}
		if b != nil {
			db = ptr(b.ensureGradGPU())
		}
		launchBatchNorm("bn_grad_stats", s.channels, 256, unsafe.Pointer(&xp), unsafe.Pointer(&gp), unsafe.Pointer(&sp), unsafe.Pointer(&sumPtr), unsafe.Pointer(&dw), unsafe.Pointer(&db), unsafe.Pointer(&channels), unsafe.Pointer(&inner), unsafe.Pointer(&samples))
		if dx := x.ensureGradGPU(); dx != nil {
			dp := ptr(dx)
			launchBatchNorm("bn_dx", (x.Numel()+255)/256, 256, unsafe.Pointer(&xp), unsafe.Pointer(&gp), unsafe.Pointer(&wp), unsafe.Pointer(&sp), unsafe.Pointer(&sumPtr), unsafe.Pointer(&dp), unsafe.Pointer(&n), unsafe.Pointer(&channels), unsafe.Pointer(&inner), unsafe.Pointer(&training))
		}
	})
}
