package autograd

import (
	"github.com/yh2237/gograd/cuda"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

type OpTiming struct {
	Count    int
	Duration time.Duration
}

var gpuProfiling atomic.Bool

// gpuTimings is a process-wide diagnostic profile, so its map is guarded: two OS
// threads can drive independent CUDA work while profiling is enabled.
var (
	gpuTimingsMu sync.Mutex
	gpuTimings   = map[string]OpTiming{}
)

func EnableGPUProfile(enabled bool) {
	gpuProfiling.Store(enabled)
	if enabled {
		gpuTimingsMu.Lock()
		gpuTimings = map[string]OpTiming{}
		gpuTimingsMu.Unlock()
	}
}
func GPUProfile() map[string]OpTiming {
	gpuTimingsMu.Lock()
	defer gpuTimingsMu.Unlock()
	out := map[string]OpTiming{}
	for k, v := range gpuTimings {
		out[k] = v
	}
	return out
}
func timedGPU(name string, fn func() error) {
	if !gpuProfiling.Load() {
		if e := fn(); e != nil {
			panic(e)
		}
		return
	}
	start, e := cuda.NewEvent(true)
	if e != nil {
		panic(e)
	}
	defer start.Destroy()
	end, e := cuda.NewEvent(true)
	if e != nil {
		panic(e)
	}
	defer end.Destroy()
	if e = start.Record(nil); e != nil {
		panic(e)
	}
	if e := fn(); e != nil {
		panic(e)
	}
	if e = end.Record(nil); e != nil {
		panic(e)
	}
	if e = end.Synchronize(); e != nil {
		panic(e)
	}
	ms, e := cuda.ElapsedTime(start, end)
	if e != nil {
		panic(e)
	}
	v := OpTiming{}
	gpuTimingsMu.Lock()
	v = gpuTimings[name]
	v.Count++
	v.Duration += time.Duration(float64(ms) * float64(time.Millisecond))
	gpuTimings[name] = v
	gpuTimingsMu.Unlock()
}

const graphSource = `
extern "C" __global__ void copy_values(const float* a,float* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i];}
extern "C" __global__ void f32_to_bf16(const float* x,unsigned short* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){unsigned int bits=__float_as_uint(x[i]);unsigned int round=0x7fffu+((bits>>16)&1u);y[i]=(unsigned short)((bits+round)>>16);}}
__device__ unsigned short round_bf16(float x){unsigned int bits=__float_as_uint(x);return (unsigned short)((bits+0x7fffu+((bits>>16)&1u))>>16);}
extern "C" __global__ void set_one(float* x){x[0]=1.0f;}
extern "C" __global__ void seed_grad(float* x, float seed, int accumulate) {
    if (accumulate) x[0] += seed;
    else x[0] = seed;
}
extern "C" __global__ void write_clip_entry(unsigned long long* ptrs,int* sizes,int index,unsigned long long address,int size){ptrs[index]=address;sizes[index]=size;}
struct Axes6 {int d[6];};
__device__ int view_index(int i,Axes6 shape,Axes6 stride,int offset){int j=offset;for(int axis=5;axis>=0;axis--){int dim=shape.d[axis];int c=i%dim;i/=dim;j+=c*stride.d[axis];}return j;}
extern "C" __global__ void view_gather(const float* x,float* y,int n,Axes6 shape,Axes6 stride,int offset){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=x[view_index(i,shape,stride,offset)];}
extern "C" __global__ void view_gather_bf16(const unsigned short* x,unsigned short* y,int n,Axes6 shape,Axes6 stride,int offset){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=x[view_index(i,shape,stride,offset)];}
extern "C" __global__ void view_scatter(const float* g,float* dx,int n,Axes6 shape,Axes6 stride,int offset){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)atomicAdd(&dx[view_index(i,shape,stride,offset)],g[i]);}
extern "C" __global__ void binary_nd_f(const float* a,const float* b,float* y,int n,Axes6 shape,Axes6 as,Axes6 bs,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float x=a[view_index(i,shape,as,0)],z=b[view_index(i,shape,bs,0)];y[i]=op==0?x+z:op==1?x-z:op==2?x*z:op==3?x/z:op==4?fmaxf(x,z):fminf(x,z);}
extern "C" __global__ void binary_nd_bf16_f(const float* a,const float* b,float* y,unsigned short* y16,int n,Axes6 shape,Axes6 as,Axes6 bs,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float x=a[view_index(i,shape,as,0)],z=b[view_index(i,shape,bs,0)];float result=op==0?x+z:op==1?x-z:op==2?x*z:op==3?x/z:op==4?fmaxf(x,z):fminf(x,z);y[i]=result;y16[i]=round_bf16(result);}
extern "C" __global__ void binary_nd_b(const float* a,const float* b,const float* g,float* da,float* db,int n,Axes6 shape,Axes6 as,Axes6 bs,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int ai=view_index(i,shape,as,0),bi=view_index(i,shape,bs,0);float x=a[ai],z=b[bi],up=g[i];if(da)atomicAdd(&da[ai],up*(op==2?z:op==3?1.0f/z:op==4?(x>z?1.0f:0.0f):op==5?(x<z?1.0f:0.0f):1.0f));if(db)atomicAdd(&db[bi],up*(op==0?1.0f:op==1?-1.0f:op==2?x:op==4?(x>z?0.0f:1.0f):op==5?(x<z?0.0f:1.0f):-x/(z*z)));}
extern "C" __global__ void reduce_nd_f(const float* x,float* y,int n,Axes6 shape,Axes6 map,float scale){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)atomicAdd(&y[view_index(i,shape,map,0)],x[i]*scale);}
extern "C" __global__ void reduce_nd_b(const float* g,float* dx,int n,Axes6 shape,Axes6 map,float scale){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dx[i]+=g[view_index(i,shape,map,0)]*scale;}
// Scalar reductions use at most 256 block atomics rather than one per element.
// The grid-stride loop requires no scratch allocations, including in capture.
extern "C" __global__ void reduce_scalar_f(const float* x,float* y,int n,float scale){
 __shared__ float sums[256];float sum=0;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)sum+=x[i];
 sums[threadIdx.x]=sum;__syncthreads();
 for(int step=128;step>0;step>>=1){if(threadIdx.x<step)sums[threadIdx.x]+=sums[threadIdx.x+step];__syncthreads();}
 if(threadIdx.x==0)atomicAdd(y,sums[0]*scale);
}
__device__ float warp_sum(float x){for(int delta=16;delta>0;delta>>=1)x+=__shfl_down_sync(0xffffffff,x,delta);return __shfl_sync(0xffffffff,x,0);}
__device__ float warp_max(float x){for(int delta=16;delta>0;delta>>=1){float o=__shfl_down_sync(0xffffffff,x,delta);x=(isnan(x)||isnan(o))?__int_as_float(0x7fc00000):fmaxf(x,o);}return __shfl_sync(0xffffffff,x,0);}
extern "C" __global__ void reduce_last_f(const float* x,float* y,int rows,int dim,float scale){
 int row=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5),lane=threadIdx.x&31;
 if(row>=rows)return;
 int base=row*dim;float sum=0.0f;
 for(int j=lane;j<dim;j+=32)sum+=x[base+j]*scale;
 sum=warp_sum(sum);
 if(lane==0)y[row]=sum;
}
extern "C" __global__ void split_heads_f(const float* x,float* y,int n,int t,int heads,int dim,int parts,int which){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int d=i%dim,tmp=i/dim,ti=tmp%t;tmp/=t;int h=tmp%heads,b=tmp/heads;
 y[i]=x[((b*t+ti)*parts+which)*heads*dim+h*dim+d];
}
extern "C" __global__ void split_heads_bf16_f(const float* x,float* y,unsigned short* y16,int n,int t,int heads,int dim,int parts,int which){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int d=i%dim,tmp=i/dim,ti=tmp%t;tmp/=t;int h=tmp%heads,b=tmp/heads;
 float v=x[((b*t+ti)*parts+which)*heads*dim+h*dim+d];
 y[i]=v;y16[i]=round_bf16(v);
}
extern "C" __global__ void split_heads_b(const float* g,float* dx,int n,int t,int heads,int dim,int parts,int which,int add){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int d=i%dim,tmp=i/dim,ti=tmp%t;tmp/=t;int h=tmp%heads,b=tmp/heads;
 int dst=((b*t+ti)*parts+which)*heads*dim+h*dim+d;
 if(add)dx[dst]+=g[i];else dx[dst]=g[i];
}
extern "C" __global__ void merge_heads_f(const float* x,float* y,int n,int t,int heads,int dim){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int d=i%dim,tmp=i/dim,h=tmp%heads;tmp/=heads;int ti=tmp%t,b=tmp/t;
 y[i]=x[((b*heads+h)*t+ti)*dim+d];
}
extern "C" __global__ void merge_heads_bf16_f(const float* x,float* y,unsigned short* y16,int n,int t,int heads,int dim){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int d=i%dim,tmp=i/dim,h=tmp%heads;tmp/=heads;int ti=tmp%t,b=tmp/t;
 float v=x[((b*heads+h)*t+ti)*dim+d];
 y[i]=v;y16[i]=round_bf16(v);
}
extern "C" __global__ void merge_heads_b(const float* g,float* dx,int n,int t,int heads,int dim,int add){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int d=i%dim,tmp=i/dim,h=tmp%heads;tmp/=heads;int ti=tmp%t,b=tmp/t;
 int dst=((b*heads+h)*t+ti)*dim+d;
 if(add)dx[dst]+=g[i];else dx[dst]=g[i];
}
extern "C" __global__ void softmax_f(const float* x,float* y,int rows,int dim,int inner,int logmode){int row=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5),lane=threadIdx.x&31;if(row>=rows)return;int o=row/inner,j=row%inner,base=o*dim*inner+j;float mx=-__int_as_float(0x7f800000);for(int k=lane;k<dim;k+=32){float v=x[base+k*inner];mx=isnan(v)?v:fmaxf(mx,v);}mx=warp_max(mx);float sum=0;for(int k=lane;k<dim;k+=32)sum+=expf(x[base+k*inner]-mx);sum=warp_sum(sum);float ls=logf(sum);for(int k=lane;k<dim;k+=32){int idx=base+k*inner;y[idx]=logmode?x[idx]-mx-ls:expf(x[idx]-mx)/sum;}}
extern "C" __global__ void softmax_b(const float* y,const float* g,float* dx,int rows,int dim,int inner,int logmode){int row=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5),lane=threadIdx.x&31;if(row>=rows)return;int o=row/inner,j=row%inner,base=o*dim*inner+j;float dot=0;for(int k=lane;k<dim;k+=32){int idx=base+k*inner;dot+=g[idx]*(logmode?1.0f:y[idx]);}dot=warp_sum(dot);for(int k=lane;k<dim;k+=32){int idx=base+k*inner;dx[idx]+=logmode?g[idx]-expf(y[idx])*dot:y[idx]*(g[idx]-dot);}}
extern "C" __global__ void attn_softmax_f(const float* scores,const float* mask,float* probs,unsigned short* probs16,int rows,int dim,float scale,Axes6 shape,Axes6 ms){
  int row=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5),lane=threadIdx.x&31;
  if(row>=rows)return;
  int base=row*dim;
  float mx=-__int_as_float(0x7f800000);
  for(int k=lane;k<dim;k+=32){int i=base+k;float z=scores[i]*scale+(mask?mask[view_index(i,shape,ms,0)]:0.0f);mx=isnan(z)?z:fmaxf(mx,z);}
  mx=warp_max(mx);
  float sum=0;
  for(int k=lane;k<dim;k+=32){int i=base+k;float z=scores[i]*scale+(mask?mask[view_index(i,shape,ms,0)]:0.0f);float e=expf(z-mx);probs[i]=e;sum+=e;}
  float inv=1.0f/warp_sum(sum);
  for(int k=lane;k<dim;k+=32){int i=base+k;float p=probs[i]*inv;probs[i]=p;if(probs16)probs16[i]=round_bf16(p);}
}
extern "C" __global__ void attn_softmax_b(const float* probs,const float* dp,float* ds,unsigned short* ds16,float* dmask,int rows,int dim,float scale,Axes6 shape,Axes6 ms){
  int row=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5),lane=threadIdx.x&31;
  if(row>=rows)return;
  int base=row*dim;
  float dot=0;
  for(int k=lane;k<dim;k+=32){int i=base+k;dot+=probs[i]*dp[i];}
  dot=warp_sum(dot);
  for(int k=lane;k<dim;k+=32){int i=base+k;float z=probs[i]*(dp[i]-dot);float dsv=z*scale;ds[i]=dsv;if(ds16)ds16[i]=round_bf16(dsv);if(dmask)atomicAdd(&dmask[view_index(i,shape,ms,0)],z);}
}
template<int BR,int BC,int ECOLS>
__device__ __forceinline__ void flash_fwd_impl(const float* q,const float* k,const float* v,const float* mask,float* out,float* stats,int nq,int nk,int d,int e,int qblocks,float scale,Axes6 shape,Axes6 ms){
  extern __shared__ float smem[];
  constexpr int RM=BR/8,CM=BC/16,SS=BC+1;
  int dp=d|1,kp=(d>e?d:e)|1;
  float* sq=smem;
  float* skv=sq+BR*dp;
  float* ss=skv+BC*kp;
  float* sm=ss+BR*SS;
  float* sl=sm+BR;
  float* sold=sl+BR;
  int batch=blockIdx.x/qblocks,qbase=(blockIdx.x%qblocks)*BR;
  int qtile=min(BR,nq-qbase);
  int tx=threadIdx.x&15,ty=threadIdx.x>>4;
  int r0=ty*RM,c0=tx*CM;
  for(int r=ty;r<BR;r+=8)for(int j=tx;j<d;j+=16)sq[r*dp+j]=r<qtile?q[(batch*nq+qbase+r)*d+j]:0.0f;
  float ninf=-__int_as_float(0x7f800000);
  if(threadIdx.x<BR){sm[threadIdx.x]=ninf;sl[threadIdx.x]=0.0f;sold[threadIdx.x]=1.0f;}
  __syncthreads();
  float acc[RM][ECOLS];
  #pragma unroll
  for(int ri=0;ri<RM;ri++){
    #pragma unroll
    for(int col=0;col<ECOLS;col++)acc[ri][col]=0.0f;}
  for(int base=0;base<nk;base+=BC){
    int ktile=min(BC,nk-base);
    for(int r=ty;r<BC;r+=8)for(int j=tx;j<d;j+=16)skv[r*kp+j]=r<ktile?k[(batch*nk+base+r)*d+j]:0.0f;
    __syncthreads();
    float s[RM][CM];
    #pragma unroll
    for(int ri=0;ri<RM;ri++){
      #pragma unroll
      for(int ci=0;ci<CM;ci++)s[ri][ci]=0.0f;}
    for(int j=0;j<d;j++){
      float qv[RM],kv[CM];
      #pragma unroll
      for(int ri=0;ri<RM;ri++)qv[ri]=sq[(r0+ri)*dp+j];
      #pragma unroll
      for(int ci=0;ci<CM;ci++)kv[ci]=skv[(c0+ci)*kp+j];
      #pragma unroll
      for(int ri=0;ri<RM;ri++){
      #pragma unroll
        for(int ci=0;ci<CM;ci++)s[ri][ci]=fmaf(qv[ri],kv[ci],s[ri][ci]);}
    }
    #pragma unroll
    for(int ri=0;ri<RM;ri++){
      int rr=r0+ri;
      #pragma unroll
      for(int ci=0;ci<CM;ci++){
        int cc=c0+ci;
        float val=s[ri][ci]*scale;
        if(mask&&rr<qtile&&cc<ktile)val+=mask[view_index((batch*nq+qbase+rr)*nk+base+cc,shape,ms,0)];
        ss[rr*SS+cc]=(cc<ktile)?val:ninf;
      }
    }
    __syncthreads();
    int sr=threadIdx.x>>2,slane=threadIdx.x&3;
    float tm=ninf;
    for(int c=slane;c<BC;c+=4){float v0=ss[sr*SS+c];tm=(isnan(tm)||isnan(v0))?__int_as_float(0x7fc00000):fmaxf(tm,v0);}
    for(int delta=2;delta>0;delta>>=1){float o=__shfl_xor_sync(0xffffffff,tm,delta);tm=(isnan(tm)||isnan(o))?__int_as_float(0x7fc00000):fmaxf(tm,o);}
    float mx=sm[sr];
    float next=(isnan(mx)||isnan(tm))?__int_as_float(0x7fc00000):fmaxf(mx,tm);
    float old=(next==mx)?1.0f:expf(mx-next);
    float sum=0.0f;
    for(int c=slane;c<BC;c+=4){
      float p=(c<ktile&&next!=ninf)?expf(ss[sr*SS+c]-next):0.0f;
      ss[sr*SS+c]=p;
      sum+=p;
    }
    for(int delta=2;delta>0;delta>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,delta);
    if(slane==0){sold[sr]=old;sm[sr]=next;sl[sr]=sl[sr]*old+sum;}
    for(int r=ty;r<ktile;r+=8)for(int c=tx;c<e;c+=16)skv[r*kp+c]=v[(batch*nk+base+r)*e+c];
    __syncthreads();
    #pragma unroll
    for(int ri=0;ri<RM;ri++){
      float sc=sold[r0+ri];
      #pragma unroll
      for(int col=0;col<ECOLS;col++)acc[ri][col]*=sc;
    }
    for(int t=0;t<ktile;t++){
      float pv[RM],vv[ECOLS];
      #pragma unroll
      for(int ri=0;ri<RM;ri++)pv[ri]=ss[(r0+ri)*SS+t];
      #pragma unroll
      for(int col=0;col<ECOLS;col++){int c=tx+col*16;vv[col]=(c<e)?skv[t*kp+c]:0.0f;}
      #pragma unroll
      for(int ri=0;ri<RM;ri++){
      #pragma unroll
        for(int col=0;col<ECOLS;col++)acc[ri][col]=fmaf(pv[ri],vv[col],acc[ri][col]);}
    }
    __syncthreads();
  }
  #pragma unroll
  for(int ri=0;ri<RM;ri++){
    int rr=r0+ri;
    if(rr<qtile){
      float inv=1.0f/sl[rr];
      #pragma unroll
      for(int col=0;col<ECOLS;col++){
        int c=tx+col*16;
        if(c<e)out[(batch*nq+qbase+rr)*e+c]=acc[ri][col]*inv;
      }
    }
  }
  if(threadIdx.x<qtile){int row=batch*nq+qbase+threadIdx.x;stats[row*2]=sm[threadIdx.x];stats[row*2+1]=sl[threadIdx.x];}
}
extern "C" __global__ void flash_attention_tiled_f(const float* q,const float* k,const float* v,const float* mask,float* out,float* stats,int nq,int nk,int d,int e,int qblocks,float scale,Axes6 shape,Axes6 ms){flash_fwd_impl<32,64,4>(q,k,v,mask,out,stats,nq,nk,d,e,qblocks,scale,shape,ms);}
extern "C" __global__ void flash_attention_tiled_f128(const float* q,const float* k,const float* v,const float* mask,float* out,float* stats,int nq,int nk,int d,int e,int qblocks,float scale,Axes6 shape,Axes6 ms){flash_fwd_impl<32,32,8>(q,k,v,mask,out,stats,nq,nk,d,e,qblocks,scale,shape,ms);}
extern "C" __global__ void flash_attention_delta(const float* out,const float* g,float* delta,int rows,int e){
  int row=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5),lane=threadIdx.x&31;
  if(row>=rows)return;
  float sum=0.0f;
  for(int c=lane;c<e;c+=32)sum=fmaf(g[row*e+c],out[row*e+c],sum);
  sum=warp_sum(sum);
  if(lane==0)delta[row]=sum;
}
template<int BR,int BC,int DCOLS>
__device__ __forceinline__ void flash_bwd_q_impl(const float* q,const float* k,const float* v,const float* mask,const float* stats,const float* deltaBuf,const float* g,float* dq,float* dmask,int nq,int nk,int d,int e,int qblocks,float scale,Axes6 shape,Axes6 ms){
  extern __shared__ float smem[];
  constexpr int RM=BR/8,CM=BC/16,SS=BC+1;
  int dp=d|1,ep=e|1,kp=(d>e?d:e)|1;
  float* sq=smem;
  float* sgo=sq+BR*dp;
  float* skv=sgo+BR*ep;
  float* sdp=skv+BC*kp;
  float* sm=sdp+BR*SS;
  float* sinv=sm+BR;
  float* sdelta=sinv+BR;
  int batch=blockIdx.x/qblocks,qbase=(blockIdx.x%qblocks)*BR;
  int qtile=min(BR,nq-qbase);
  int tx=threadIdx.x&15,ty=threadIdx.x>>4;
  int r0=ty*RM,c0=tx*CM;
  for(int r=ty;r<BR;r+=8){
    for(int j=tx;j<d;j+=16)sq[r*dp+j]=r<qtile?q[(batch*nq+qbase+r)*d+j]:0.0f;
    for(int c=tx;c<e;c+=16)sgo[r*ep+c]=r<qtile?g[(batch*nq+qbase+r)*e+c]:0.0f;
  }
  if(threadIdx.x<BR){
    if(threadIdx.x<qtile){
      int row=batch*nq+qbase+threadIdx.x;
      sm[threadIdx.x]=stats[row*2];
      sinv[threadIdx.x]=1.0f/stats[row*2+1];
      sdelta[threadIdx.x]=deltaBuf[row];
    }else{
      sm[threadIdx.x]=0.0f;sinv[threadIdx.x]=0.0f;sdelta[threadIdx.x]=0.0f;
    }
  }
  __syncthreads();
  float dqa[RM][DCOLS];
  #pragma unroll
  for(int ri=0;ri<RM;ri++){
    #pragma unroll
    for(int col=0;col<DCOLS;col++)dqa[ri][col]=0.0f;}
  for(int base=0;base<nk;base+=BC){
    int ktile=min(BC,nk-base);
    for(int r=ty;r<BC;r+=8)for(int c=tx;c<e;c+=16)skv[r*kp+c]=r<ktile?v[(batch*nk+base+r)*e+c]:0.0f;
    __syncthreads();
    float dpv[RM][CM];
    #pragma unroll
    for(int ri=0;ri<RM;ri++){
      #pragma unroll
      for(int ci=0;ci<CM;ci++)dpv[ri][ci]=0.0f;}
    for(int c=0;c<e;c++){
      float gv[RM],vv[CM];
      #pragma unroll
      for(int ri=0;ri<RM;ri++)gv[ri]=sgo[(r0+ri)*ep+c];
      #pragma unroll
      for(int ci=0;ci<CM;ci++)vv[ci]=skv[(c0+ci)*kp+c];
      #pragma unroll
      for(int ri=0;ri<RM;ri++){
      #pragma unroll
        for(int ci=0;ci<CM;ci++)dpv[ri][ci]=fmaf(gv[ri],vv[ci],dpv[ri][ci]);}
    }
    #pragma unroll
    for(int ri=0;ri<RM;ri++){
      #pragma unroll
      for(int ci=0;ci<CM;ci++)sdp[(r0+ri)*SS+c0+ci]=dpv[ri][ci];}
    __syncthreads();
    for(int r=ty;r<BC;r+=8)for(int j=tx;j<d;j+=16)skv[r*kp+j]=r<ktile?k[(batch*nk+base+r)*d+j]:0.0f;
    __syncthreads();
    float sv[RM][CM];
    #pragma unroll
    for(int ri=0;ri<RM;ri++){
      #pragma unroll
      for(int ci=0;ci<CM;ci++)sv[ri][ci]=0.0f;}
    for(int j=0;j<d;j++){
      float qv[RM],kv[CM];
      #pragma unroll
      for(int ri=0;ri<RM;ri++)qv[ri]=sq[(r0+ri)*dp+j];
      #pragma unroll
      for(int ci=0;ci<CM;ci++)kv[ci]=skv[(c0+ci)*kp+j];
      #pragma unroll
      for(int ri=0;ri<RM;ri++){
      #pragma unroll
        for(int ci=0;ci<CM;ci++)sv[ri][ci]=fmaf(qv[ri],kv[ci],sv[ri][ci]);}
    }
    #pragma unroll
    for(int ri=0;ri<RM;ri++){
      int rr=r0+ri;
      float m=sm[rr],inv=sinv[rr],dl=sdelta[rr];
      #pragma unroll
      for(int ci=0;ci<CM;ci++){
        int cc=c0+ci;
        float ds=0.0f;
        if(rr<qtile&&cc<ktile){
          float val=sv[ri][ci]*scale;
          if(mask)val+=mask[view_index((batch*nq+qbase+rr)*nk+base+cc,shape,ms,0)];
          ds=expf(val-m)*inv*(sdp[rr*SS+cc]-dl);
          if(dmask)atomicAdd(&dmask[view_index((batch*nq+qbase+rr)*nk+base+cc,shape,ms,0)],ds);
        }
        sdp[rr*SS+cc]=ds*scale;
      }
    }
    __syncthreads();
    if(dq){
      for(int t=0;t<ktile;t++){
        float av[RM],kv[DCOLS];
        #pragma unroll
        for(int ri=0;ri<RM;ri++)av[ri]=sdp[(r0+ri)*SS+t];
        #pragma unroll
        for(int col=0;col<DCOLS;col++){int j=tx+col*16;kv[col]=(j<d)?skv[t*kp+j]:0.0f;}
        #pragma unroll
        for(int ri=0;ri<RM;ri++){
      #pragma unroll
          for(int col=0;col<DCOLS;col++)dqa[ri][col]=fmaf(av[ri],kv[col],dqa[ri][col]);}
      }
    }
    __syncthreads();
  }
  if(dq){
    #pragma unroll
    for(int ri=0;ri<RM;ri++){
      int rr=r0+ri;
      if(rr<qtile){
        #pragma unroll
        for(int col=0;col<DCOLS;col++){
          int j=tx+col*16;
          if(j<d)dq[(batch*nq+qbase+rr)*d+j]+=dqa[ri][col];
        }
      }
    }
  }
}
extern "C" __global__ void flash_attention_tiled_b(const float* q,const float* k,const float* v,const float* mask,const float* stats,const float* deltaBuf,const float* g,float* dq,float* dmask,int nq,int nk,int d,int e,int qblocks,float scale,Axes6 shape,Axes6 ms){flash_bwd_q_impl<32,64,4>(q,k,v,mask,stats,deltaBuf,g,dq,dmask,nq,nk,d,e,qblocks,scale,shape,ms);}
extern "C" __global__ void flash_attention_tiled_b128(const float* q,const float* k,const float* v,const float* mask,const float* stats,const float* deltaBuf,const float* g,float* dq,float* dmask,int nq,int nk,int d,int e,int qblocks,float scale,Axes6 shape,Axes6 ms){flash_bwd_q_impl<16,32,8>(q,k,v,mask,stats,deltaBuf,g,dq,dmask,nq,nk,d,e,qblocks,scale,shape,ms);}
template<int BK,int BQ,int DCOLS,int ECOLS>
__device__ __forceinline__ void flash_bwd_kv_impl(const float* q,const float* k,const float* v,const float* mask,const float* stats,const float* deltaBuf,const float* g,float* dk,float* dv,int nq,int nk,int d,int e,int kvblocks,float scale,Axes6 shape,Axes6 ms){
  extern __shared__ float smem[];
  constexpr int RM=BK/8,CM=BQ/16,SS=BQ+1;
  int dp=d|1,ep=e|1;
  float* sk=smem;
  float* sv=sk+BK*dp;
  float* sq=sv+BK*ep;
  float* sgo=sq+BQ*dp;
  float* sp=sgo+BQ*ep;
  float* sds=sp+BK*SS;
  float* sm=sds+BK*SS;
  float* sinv=sm+BQ;
  float* sdelta=sinv+BQ;
  int batch=blockIdx.x/kvblocks,kbase=(blockIdx.x%kvblocks)*BK;
  int ktile=min(BK,nk-kbase);
  int tx=threadIdx.x&15,ty=threadIdx.x>>4;
  int r0=ty*RM,c0=tx*CM;
  for(int r=ty;r<BK;r+=8){
    for(int j=tx;j<d;j+=16)sk[r*dp+j]=r<ktile?k[(batch*nk+kbase+r)*d+j]:0.0f;
    for(int c=tx;c<e;c+=16)sv[r*ep+c]=r<ktile?v[(batch*nk+kbase+r)*e+c]:0.0f;
  }
  __syncthreads();
  float dka[RM][DCOLS],dva[RM][ECOLS];
  #pragma unroll
  for(int ri=0;ri<RM;ri++){
    #pragma unroll
    for(int col=0;col<DCOLS;col++)dka[ri][col]=0.0f;
    #pragma unroll
    for(int col=0;col<ECOLS;col++)dva[ri][col]=0.0f;
  }
  for(int base=0;base<nq;base+=BQ){
    int qtile=min(BQ,nq-base);
    for(int r=ty;r<BQ;r+=8){
      for(int j=tx;j<d;j+=16)sq[r*dp+j]=r<qtile?q[(batch*nq+base+r)*d+j]:0.0f;
      for(int c=tx;c<e;c+=16)sgo[r*ep+c]=r<qtile?g[(batch*nq+base+r)*e+c]:0.0f;
    }
    if(threadIdx.x<BQ){
      if(threadIdx.x<qtile){
        int qrow=batch*nq+base+threadIdx.x;
        sm[threadIdx.x]=stats[qrow*2];
        sinv[threadIdx.x]=1.0f/stats[qrow*2+1];
        sdelta[threadIdx.x]=deltaBuf[qrow];
      }else{
        sm[threadIdx.x]=0.0f;sinv[threadIdx.x]=0.0f;sdelta[threadIdx.x]=0.0f;
      }
    }
    __syncthreads();
    float s[RM][CM],dpv[RM][CM];
    #pragma unroll
    for(int ri=0;ri<RM;ri++){
      #pragma unroll
      for(int ci=0;ci<CM;ci++){s[ri][ci]=0.0f;dpv[ri][ci]=0.0f;}}
    for(int j=0;j<d;j++){
      float kv[RM],qv[CM];
      #pragma unroll
      for(int ri=0;ri<RM;ri++)kv[ri]=sk[(r0+ri)*dp+j];
      #pragma unroll
      for(int ci=0;ci<CM;ci++)qv[ci]=sq[(c0+ci)*dp+j];
      #pragma unroll
      for(int ri=0;ri<RM;ri++){
      #pragma unroll
        for(int ci=0;ci<CM;ci++)s[ri][ci]=fmaf(kv[ri],qv[ci],s[ri][ci]);}
    }
    for(int c=0;c<e;c++){
      float vv[RM],gv[CM];
      #pragma unroll
      for(int ri=0;ri<RM;ri++)vv[ri]=sv[(r0+ri)*ep+c];
      #pragma unroll
      for(int ci=0;ci<CM;ci++)gv[ci]=sgo[(c0+ci)*ep+c];
      #pragma unroll
      for(int ri=0;ri<RM;ri++){
      #pragma unroll
        for(int ci=0;ci<CM;ci++)dpv[ri][ci]=fmaf(vv[ri],gv[ci],dpv[ri][ci]);}
    }
    #pragma unroll
    for(int ri=0;ri<RM;ri++){
      int rr=r0+ri;
      #pragma unroll
      for(int ci=0;ci<CM;ci++){
        int cc=c0+ci;
        float p=0.0f,ds=0.0f;
        if(rr<ktile&&cc<qtile){
          float val=s[ri][ci]*scale;
          if(mask)val+=mask[view_index((batch*nq+base+cc)*nk+kbase+rr,shape,ms,0)];
          p=expf(val-sm[cc])*sinv[cc];
          ds=p*(dpv[ri][ci]-sdelta[cc])*scale;
        }
        sp[rr*SS+cc]=p;
        sds[rr*SS+cc]=ds;
      }
    }
    __syncthreads();
    if(dv){
      for(int t=0;t<qtile;t++){
        float pv[RM],gv[ECOLS];
        #pragma unroll
        for(int ri=0;ri<RM;ri++)pv[ri]=sp[(r0+ri)*SS+t];
        #pragma unroll
        for(int col=0;col<ECOLS;col++){int c=tx+col*16;gv[col]=(c<e)?sgo[t*ep+c]:0.0f;}
        #pragma unroll
        for(int ri=0;ri<RM;ri++){
      #pragma unroll
          for(int col=0;col<ECOLS;col++)dva[ri][col]=fmaf(pv[ri],gv[col],dva[ri][col]);}
      }
    }
    if(dk){
      for(int t=0;t<qtile;t++){
        float av[RM],qv[DCOLS];
        #pragma unroll
        for(int ri=0;ri<RM;ri++)av[ri]=sds[(r0+ri)*SS+t];
        #pragma unroll
        for(int col=0;col<DCOLS;col++){int j=tx+col*16;qv[col]=(j<d)?sq[t*dp+j]:0.0f;}
        #pragma unroll
        for(int ri=0;ri<RM;ri++){
      #pragma unroll
          for(int col=0;col<DCOLS;col++)dka[ri][col]=fmaf(av[ri],qv[col],dka[ri][col]);}
      }
    }
    __syncthreads();
  }
  if(dk){
    #pragma unroll
    for(int ri=0;ri<RM;ri++){
      int rr=r0+ri;
      if(rr<ktile){
        #pragma unroll
        for(int col=0;col<DCOLS;col++){
          int j=tx+col*16;
          if(j<d)dk[(batch*nk+kbase+rr)*d+j]+=dka[ri][col];
        }
      }
    }
  }
  if(dv){
    #pragma unroll
    for(int ri=0;ri<RM;ri++){
      int rr=r0+ri;
      if(rr<ktile){
        #pragma unroll
        for(int col=0;col<ECOLS;col++){
          int c=tx+col*16;
          if(c<e)dv[(batch*nk+kbase+rr)*e+c]+=dva[ri][col];
        }
      }
    }
  }
}
extern "C" __global__ void flash_attention_tiled_b_kv(const float* q,const float* k,const float* v,const float* mask,const float* stats,const float* deltaBuf,const float* g,float* dk,float* dv,int nq,int nk,int d,int e,int kvblocks,float scale,Axes6 shape,Axes6 ms){flash_bwd_kv_impl<32,32,4,4>(q,k,v,mask,stats,deltaBuf,g,dk,dv,nq,nk,d,e,kvblocks,scale,shape,ms);}
extern "C" __global__ void flash_attention_tiled_b_kv128(const float* q,const float* k,const float* v,const float* mask,const float* stats,const float* deltaBuf,const float* g,float* dk,float* dv,int nq,int nk,int d,int e,int kvblocks,float scale,Axes6 shape,Axes6 ms){flash_bwd_kv_impl<16,16,8,8>(q,k,v,mask,stats,deltaBuf,g,dk,dv,nq,nk,d,e,kvblocks,scale,shape,ms);}
extern "C" __global__ void layernorm_f(const float* x,const float* w,const float* b,float* y,float* stats,int rows,int dim,float eps){
  int row=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5),lane=threadIdx.x&31;
  if(row>=rows)return;
  int base=row*dim;
  float sum=0;for(int j=lane;j<dim;j+=32)sum+=x[base+j];
  float mean=warp_sum(sum)/(float)dim;
  float ss=0;for(int j=lane;j<dim;j+=32){float z=x[base+j]-mean;ss+=z*z;}
  float inv=rsqrtf(warp_sum(ss)/(float)dim+eps);
  if(lane==0){stats[row*2]=mean;stats[row*2+1]=inv;}
  for(int j=lane;j<dim;j+=32)y[base+j]=(x[base+j]-mean)*inv*w[j]+b[j];
}
extern "C" __global__ void layernorm_bf16_f(const float* x,const float* w,const float* b,float* y,unsigned short* y16,float* stats,int rows,int dim,float eps){
  int row=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5),lane=threadIdx.x&31;
  if(row>=rows)return;
  int base=row*dim;
  float sum=0;for(int j=lane;j<dim;j+=32)sum+=x[base+j];
  float mean=warp_sum(sum)/(float)dim;
  float ss=0;for(int j=lane;j<dim;j+=32){float z=x[base+j]-mean;ss+=z*z;}
  float inv=rsqrtf(warp_sum(ss)/(float)dim+eps);
  if(lane==0){stats[row*2]=mean;stats[row*2+1]=inv;}
  for(int j=lane;j<dim;j+=32){float z=(x[base+j]-mean)*inv*w[j]+b[j];y[base+j]=z;y16[base+j]=round_bf16(z);}
}
extern "C" __global__ void layernorm_b(const float* x,const float* w,const float* g,const float* stats,float* dx,float* dw,float* db,int rows,int dim){
  int warp=threadIdx.x>>5,lane=threadIdx.x&31;
  int row=blockIdx.x*(blockDim.x>>5)+warp;
  bool active=row<rows;
  __shared__ float sdw[8][32],sdb[8][32];
  int base=row*dim;
  float mean=active?stats[row*2]:0.0f,inv=active?stats[row*2+1]:0.0f;
  float sg=0,sgn=0;
  if(active){
    for(int j=lane;j<dim;j+=32){int i=base+j;float norm=(x[i]-mean)*inv,z=g[i]*w[j];sg+=z;sgn+=z*norm;}
  }
  sg=warp_sum(sg)/(float)dim;
  sgn=warp_sum(sgn)/(float)dim;
  int steps=(dim+31)>>5;
  for(int c=0;c<steps;c++){
    int j=lane+c*32;
    float dwv=0.0f,dbv=0.0f;
    if(active&&j<dim){
      int i=base+j;
      float norm=(x[i]-mean)*inv,gi=g[i];
      if(dx)dx[i]+=inv*(gi*w[j]-sg-norm*sgn);
      dwv=gi*norm;
      dbv=gi;
    }
    if(dw||db){
      sdw[warp][lane]=dwv;
      sdb[warp][lane]=dbv;
      __syncthreads();
      if(warp==0&&j<dim){
        if(dw){float sw=0;for(int r=0;r<8;r++)sw+=sdw[r][lane];atomicAdd(&dw[j],sw);}
        if(db){float sb=0;for(int r=0;r<8;r++)sb+=sdb[r][lane];atomicAdd(&db[j],sb);}
      }
      __syncthreads();
    }
  }
}
__device__ unsigned int dropout_hash(unsigned int i,unsigned int seed){unsigned int x=i^seed;x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x;}
extern "C" __global__ void dropout_f(const float* x,float* y,int n,float p,unsigned int seed,unsigned int cutoff){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=dropout_hash(i,seed)<cutoff?0.0f:x[i]/(1.0f-p);}
extern "C" __global__ void dropout_b(const float* g,float* dx,int n,float p,unsigned int seed,unsigned int cutoff){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dx[i]+=dropout_hash(i,seed)<cutoff?0.0f:g[i]/(1.0f-p);}
extern "C" __global__ void ce_f(const float* x,const int* target,float* acc,int rows,int classes,int ignore){
  int row=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5),lane=threadIdx.x&31;
  if(row>=rows)return;
  int tg=target[row];
  if(tg==ignore)return;
  int base=row*classes;
  float mx=-__int_as_float(0x7f800000);
  for(int j=lane;j<classes;j+=32){float v=x[base+j];mx=isnan(v)?v:fmaxf(mx,v);}
  mx=warp_max(mx);
  float sum=0.0f;
  for(int j=lane;j<classes;j+=32)sum+=expf(x[base+j]-mx);
  sum=warp_sum(sum);
  if(lane==0){atomicAdd(&acc[0],(mx-x[base+tg])+logf(sum));atomicAdd(&acc[1],1.0f);}
}
extern "C" __global__ void bce_logits_f(const float* x,const float* target,float* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=x[i];out[i]=fmaxf(z,0.0f)-z*target[i]+log1pf(expf(-fabsf(z)));}}
extern "C" __global__ void bce_logits_b(const float* x,const float* target,const float* g,float* dx,float* dt,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=x[i],e=expf(-fabsf(z)),sig=z>=0.0f?1.0f/(1.0f+e):e/(1.0f+e);if(dx)dx[i]+=g[i]*(sig-target[i]);if(dt)dt[i]-=g[i]*z;}}
extern "C" __global__ void smooth_l1_f(const float* x,const float* target,float* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float d=x[i]-target[i],a=fabsf(d);out[i]=a<1.0f?0.5f*d*d:a-0.5f;}}
extern "C" __global__ void smooth_l1_b(const float* x,const float* target,const float* g,float* dx,float* dt,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float d=x[i]-target[i],a=fabsf(d),s=a<1.0f?d:(d>0.0f?1.0f:(d<0.0f?-1.0f:d)),u=g[i]*s;if(dx)dx[i]+=u;if(dt)dt[i]-=u;}}
extern "C" __global__ void ce_finish(const float* acc,float* y){y[0]=acc[0]/acc[1];}
extern "C" __global__ void ce_b(const float* x,const int* target,const float* acc,const float* g,float* dx,int rows,int classes,int ignore){
  int row=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5),lane=threadIdx.x&31;
  if(row>=rows)return;
  int tg=target[row];
  if(tg==ignore||acc[1]==0)return;
  int base=row*classes;
  float mx=-__int_as_float(0x7f800000);
  for(int j=lane;j<classes;j+=32){float v=x[base+j];mx=isnan(v)?v:fmaxf(mx,v);}
  mx=warp_max(mx);
  float sum=0.0f;
  for(int j=lane;j<classes;j+=32)sum+=expf(x[base+j]-mx);
  sum=warp_sum(sum);
  float up=g[0]/acc[1],inv=1.0f/sum;
  for(int j=lane;j<classes;j+=32)dx[base+j]+=up*(expf(x[base+j]-mx)*inv-(j==tg?1.0f:0.0f));
}
extern "C" __global__ void graph_adam_inc(int* step){if(threadIdx.x==0&&blockIdx.x==0)step[0]++;}
// wd is already the effective learning rate times the group's weight decay.
extern "C" __global__ void graph_adamw(float* p,const float* g,float* m,float* v,const int* step,int n,float b1,float b2,float lr,float wd,float eps){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float grad=g[i];float mm=b1*m[i]+(1.0f-b1)*grad;float vv=b2*v[i]+(1.0f-b2)*grad*grad;m[i]=mm;v[i]=vv;float bc1=1.0f-powf(b1,(float)step[0]),bc2=1.0f-powf(b2,(float)step[0]);p[i]=p[i]*(1.0f-wd)-lr*(mm/bc1)/(sqrtf(vv/bc2)+eps);}
extern "C" __global__ void add_values(float* a,const float* b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)a[i]+=b[i];}
extern "C" __global__ void scalar_f(const float* a,float* out,int n,float value,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float x=a[i];out[i]=op==0?x+value:op==1?x-value:op==2?x*value:x/value;}}
extern "C" __global__ void scalar_b(const float* g,float* da,int n,float value,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)da[i]+=g[i]*(op==2?value:op==3?1.0f/value:1.0f);}
extern "C" __global__ void clamp_f(const float* a, float* out, int n, float lo, float hi) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = a[i];
  out[i] = (isnan(x) || isnan(lo) || isnan(hi)) ? __int_as_float(0x7fc00000) : fminf(fmaxf(x, lo), hi);
}
extern "C" __global__ void clamp_b(const float* a,const float* g,float* da,int n,float lo,float hi){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&a[i]>=lo&&a[i]<=hi)da[i]+=g[i];}
extern "C" __global__ void pow_scalar_f(const float* a,float* out,int n,float p){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=powf(a[i],p);}
extern "C" __global__ void pow_scalar_b(const float* a,const float* g,float* da,int n,float p){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)da[i]+=g[i]*(p==0.0f?0.0f:p*powf(a[i],p-1.0f));}
extern "C" __global__ void binary_same_f(const float* a,const float* b,float* out,int n,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float x=a[i],y=b[i];out[i]=op==0?x+y:op==1?x-y:op==2?x*y:op==3?x/y:op==4?fmaxf(x,y):fminf(x,y);}}
extern "C" __global__ void binary_same_bf16_f(const float* a,const float* b,float* out,unsigned short* out16,int n,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float x=a[i],y=b[i];float z=op==0?x+y:op==1?x-y:op==2?x*y:op==3?x/y:op==4?fmaxf(x,y):fminf(x,y);out[i]=z;out16[i]=round_bf16(z);}}
extern "C" __global__ void binary_same_b(const float* a,const float* b,const float* g,float* da,float* db,int n,int op,int addA,int addB){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float x=a[i],y=b[i],z=g[i];if(da){float ga=z*(op==2?y:op==3?1.0f/y:op==4?(x>y?1.0f:0.0f):op==5?(x<y?1.0f:0.0f):1.0f);if(addA)da[i]+=ga;else da[i]=ga;}if(db){float gb=z*(op==0?1.0f:op==1?-1.0f:op==2?x:op==4?(x>y?0.0f:1.0f):op==5?(x<y?0.0f:1.0f):-x/(y*y));if(addB)db[i]+=gb;else db[i]=gb;}}
extern "C" __global__ void bias_add_f(const float* a,const float* b,float* out,int n,int cols){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=a[i]+b[i%cols];}
extern "C" __global__ void bias_add_bf16_f(const float* a,const float* b,float* out,unsigned short* out16,int n,int cols){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=a[i]+b[i%cols];out[i]=z;out16[i]=round_bf16(z);}}
extern "C" __global__ void bias_add_b(const float* g,float* db,int rows,int cols){
  __shared__ float s[8][32];
  int col=blockIdx.x*32+threadIdx.x;
  float sum=0.0f;
  if(col<cols){for(int r=blockIdx.y*8+threadIdx.y;r<rows;r+=gridDim.y*8)sum+=g[r*cols+col];}
  s[threadIdx.y][threadIdx.x]=sum;
  __syncthreads();
  if(threadIdx.y<4)s[threadIdx.y][threadIdx.x]+=s[threadIdx.y+4][threadIdx.x];
  __syncthreads();
  if(threadIdx.y<2)s[threadIdx.y][threadIdx.x]+=s[threadIdx.y+2][threadIdx.x];
  __syncthreads();
  if(threadIdx.y==0&&col<cols){
    float total=s[0][threadIdx.x]+s[1][threadIdx.x];
    if(gridDim.y==1)db[col]+=total;else atomicAdd(&db[col],total);
  }
}
__device__ int index3(int i,int d1,int d2,int s0,int s1,int s2){int z=i%d2;int y=(i/d2)%d1;int x=i/(d1*d2);return (s0==1?0:x*s1*s2)+(s1==1?0:y*s2)+(s2==1?0:z);}
extern "C" __global__ void binary_f(const float* a,const float* b,float* out,int n,int d1,int d2,int a0,int a1,int a2,int b0,int b1,int b2,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float x=a[index3(i,d1,d2,a0,a1,a2)],y=b[index3(i,d1,d2,b0,b1,b2)];out[i]=op==0?x+y:op==1?x-y:op==2?x*y:op==3?x/y:op==4?fmaxf(x,y):fminf(x,y);}
extern "C" __global__ void binary_bf16_f(const float* a,const float* b,float* out,unsigned short* out16,int n,int d1,int d2,int a0,int a1,int a2,int b0,int b1,int b2,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float x=a[index3(i,d1,d2,a0,a1,a2)],y=b[index3(i,d1,d2,b0,b1,b2)];float z=op==0?x+y:op==1?x-y:op==2?x*y:op==3?x/y:op==4?fmaxf(x,y):fminf(x,y);out[i]=z;out16[i]=round_bf16(z);}
extern "C" __global__ void binary_b(const float* a,const float* b,const float* g,float* da,float* db,int n,int d1,int d2,int a0,int a1,int a2,int b0,int b1,int b2,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int ai=index3(i,d1,d2,a0,a1,a2),bi=index3(i,d1,d2,b0,b1,b2);float x=a[ai],y=b[bi],z=g[i];if(da)atomicAdd(&da[ai],z*(op==2?y:op==3?1.0f/y:op==4?(x>y?1.0f:0.0f):op==5?(x<y?1.0f:0.0f):1.0f));if(db)atomicAdd(&db[bi],z*(op==0?1.0f:op==1?-1.0f:op==2?x:op==4?(x>y?0.0f:1.0f):op==5?(x<y?0.0f:1.0f):-x/(y*y)));}
__device__ float sigmoid_f(float x){float e=expf(-fabsf(x));return x>=0.0f?1.0f/(1.0f+e):e/(1.0f+e);}
__device__ float fun(float x,int op){if(op==0)return expf(x);if(op==1)return logf(x);if(op==2)return fabsf(x);if(op==3)return tanhf(x);if(op==4)return fmaxf(x,0.0f);if(op==5)return .5f*x*(1.0f+erff(x*.7071067811865475f));if(op==7)return sinf(x);if(op==10)return cosf(x);if(op==11)return fmaxf(x,0.0f)+log1pf(expf(-fabsf(x)));if(op==12)return erff(x);if(op==13)return log1pf(x);if(op==14)return expm1f(x);if(op==8)return sqrtf(x);if(op==9)return sigmoid_f(x);float u=.7978845608028654f*(x+.044715f*x*x*x);return .5f*x*(1.0f+tanhf(u));}
__device__ float deriv(float x,int op){if(op==0)return expf(x);if(op==1)return 1.0f/x;if(op==2)return x>0?1.0f:x<0?-1.0f:0.0f;if(op==3){float y=tanhf(x);return 1-y*y;}if(op==4)return x>0?1.0f:0.0f;if(op==5)return .5f*(1.0f+erff(x*.7071067811865475f))+x*expf(-.5f*x*x)*.3989422804014327f;if(op==7)return cosf(x);if(op==10)return -sinf(x);if(op==11)return sigmoid_f(x);if(op==12)return 1.1283791670955126f*expf(-x*x);if(op==13)return 1.0f/(1.0f+x);if(op==14)return expf(x);if(op==8)return .5f/sqrtf(x);if(op==9){float s=sigmoid_f(x);return s*(1.0f-s);}float u=.7978845608028654f*(x+.044715f*x*x*x),y=tanhf(u);return .5f*(1+y)+.5f*x*(1-y*y)*.7978845608028654f*(1+3*.044715f*x*x);}
extern "C" __global__ void bias_gelu_f(const float* x,const float* b,float* y,int n,int c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=fun(x[i]+b[i%c],5);}
extern "C" __global__ void bias_gelu_bf16_f(const float* x,const float* b,float* y,unsigned short* y16,int n,int c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=fun(x[i]+b[i%c],5);y[i]=z;y16[i]=round_bf16(z);}}
extern "C" __global__ void bias_gelu_b(const float* x,const float* b,const float* g,float* dx,float* db,int n,int c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=g[i]*deriv(x[i]+b[i%c],5);if(dx)dx[i]+=z;if(db)atomicAdd(&db[i%c],z);}}
extern "C" __global__ void bias_gelu_b_2d(const float* x,const float* b,const float* g,float* dx,float* db,int rows,int c){
  __shared__ float sdb[8][32];
  int col=blockIdx.x*32+threadIdx.x;
  float acc=0.0f;
  if(col<c){
    float bv=b[col];
    for(int row=blockIdx.y*8+threadIdx.y;row<rows;row+=gridDim.y*8){
      int i=row*c+col;
      float z=g[i]*deriv(x[i]+bv,5);
      if(dx)dx[i]+=z;
      acc+=z;
    }
  }
  if(db){
    sdb[threadIdx.y][threadIdx.x]=acc;
    __syncthreads();
    if(threadIdx.y==0&&col<c){float sum=0.0f;for(int r=0;r<8;r++)sum+=sdb[r][threadIdx.x];atomicAdd(&db[col],sum);}
  }
}
extern "C" __global__ void bias_residual_f(const float* x,const float* b,const float* r,float* y,int n,int c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=x[i]+b[i%c]+r[i];}
extern "C" __global__ void bias_residual_bf16_f(const float* x,const float* b,const float* r,float* y,unsigned short* y16,int n,int c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=x[i]+b[i%c]+r[i];y[i]=z;y16[i]=round_bf16(z);}}
extern "C" __global__ void bias_residual_b(const float* g,float* dx,float* db,float* dr,int n,int c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=g[i];if(dx)dx[i]+=z;if(db)atomicAdd(&db[i%c],z);if(dr)dr[i]+=z;}}
extern "C" __global__ void bias_residual_b_2d(const float* g,float* dx,float* db,float* dr,int rows,int c){
  __shared__ float sdb[8][32];
  int col=blockIdx.x*32+threadIdx.x;
  float acc=0.0f;
  if(col<c){
    for(int row=blockIdx.y*8+threadIdx.y;row<rows;row+=gridDim.y*8){
      int i=row*c+col;
      float z=g[i];
      if(dx)dx[i]+=z;
      if(dr)dr[i]+=z;
      acc+=z;
    }
  }
  if(db){
    sdb[threadIdx.y][threadIdx.x]=acc;
    __syncthreads();
    if(threadIdx.y==0&&col<c){float sum=0.0f;for(int r=0;r<8;r++)sum+=sdb[r][threadIdx.x];atomicAdd(&db[col],sum);}
  }
}
extern "C" __global__ void unary_f(const float* a,float* out,int n,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=fun(a[i],op);}
extern "C" __global__ void unary_b(const float* a,const float* g,float* da,int n,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)da[i]+=g[i]*deriv(a[i],op);}
extern "C" __global__ void permute_f(const float* a,float* out,int n,int d0,int d1,int d2,int s0,int s1,int s2){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int j=(i/(d1*d2))*s0+((i/d2)%d1)*s1+(i%d2)*s2;out[i]=a[j];}}
extern "C" __global__ void permute_b(const float* g,float* da,int n,int d0,int d1,int d2,int s0,int s1,int s2){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int j=(i/(d1*d2))*s0+((i/d2)%d1)*s1+(i%d2)*s2;da[j]+=g[i];}}
extern "C" __global__ void concat_f(const float* a,const float* b,float* out,int outer,int ac,int bc,int inner){int i=blockIdx.x*blockDim.x+threadIdx.x;int n=outer*(ac+bc)*inner;if(i<n){int o=i/((ac+bc)*inner),r=i%((ac+bc)*inner);out[i]=r<ac*inner?a[o*ac*inner+r]:b[o*bc*inner+r-ac*inner];}}
extern "C" __global__ void concat_b(const float* g,float* da,float* db,int outer,int ac,int bc,int inner){int i=blockIdx.x*blockDim.x+threadIdx.x;int n=outer*(ac+bc)*inner;if(i<n){int o=i/((ac+bc)*inner),r=i%((ac+bc)*inner);if(r<ac*inner){if(da)da[o*ac*inner+r]+=g[i];}else if(db)db[o*bc*inner+r-ac*inner]+=g[i];}}
extern "C" __global__ void slice_f(const float* a,float* out,int n,int width,int start,int full,int inner){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int o=i/(width*inner),r=i%(width*inner);out[i]=a[o*full*inner+start*inner+r];}}
extern "C" __global__ void slice_b(const float* g,float* da,int n,int width,int start,int full,int inner){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int o=i/(width*inner),r=i%(width*inner);da[o*full*inner+start*inner+r]+=g[i];}}
extern "C" __global__ void embedding_f(const float* w,const int* ids,float* out,int n,int dim){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=w[ids[i/dim]*dim+i%dim];}
extern "C" __global__ void embedding_b(const float* g,const int* ids,float* dw,int n,int dim,int padding){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&ids[i/dim]!=padding)atomicAdd(&dw[ids[i/dim]*dim+i%dim],g[i]);}
extern "C" __global__ void reduce_f(const float* x,float* y,int n,int d1,int d2,int r0,int r1,int r2,float scale){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int j=(i/(d1*d2))*r0+((i/d2)%d1)*r1+(i%d2)*r2;atomicAdd(&y[j],x[i]*scale);}}
extern "C" __global__ void reduce_b(const float* g,float* dx,int n,int d1,int d2,int r0,int r1,int r2,float scale){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int j=(i/(d1*d2))*r0+((i/d2)%d1)*r1+(i%d2)*r2;dx[i]+=g[j]*scale;}}
extern "C" __global__ void group_stats(const float* x,float* stats,int batch,int time,int channels,int groups,float eps){
 int q=blockIdx.x;if(q>=batch*groups)return;int n=q/groups,gr=q%groups,cg=channels/groups,count=time*cg;
 __shared__ float sums[256];float sum=0;
 for(int k=threadIdx.x;k<count;k+=blockDim.x){int at=k/cg,ch=gr*cg+k%cg;sum+=x[(n*time+at)*channels+ch];}
 sums[threadIdx.x]=sum;__syncthreads();
 for(int step=128;step>0;step>>=1){if(threadIdx.x<step)sums[threadIdx.x]+=sums[threadIdx.x+step];__syncthreads();}
 float mean=sums[0]/count;__syncthreads();
 float ss=0;for(int k=threadIdx.x;k<count;k+=blockDim.x){int at=k/cg,ch=gr*cg+k%cg;float d=x[(n*time+at)*channels+ch]-mean;ss+=d*d;}
 sums[threadIdx.x]=ss;__syncthreads();
 for(int step=128;step>0;step>>=1){if(threadIdx.x<step)sums[threadIdx.x]+=sums[threadIdx.x+step];__syncthreads();}
 if(threadIdx.x==0){stats[q*4]=mean;stats[q*4+1]=rsqrtf(sums[0]/count+eps);}
}
extern "C" __global__ void group_f(const float* x,const float* w,const float* b,const float* stats,float* out,int n,int time,int channels,int groups){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int ch=i%channels,gr=ch/(channels/groups),sample=i/(time*channels),q=sample*groups+gr;out[i]=(x[i]-stats[q*4])*stats[q*4+1]*w[ch]+b[ch];}}
extern "C" __global__ void group_backstats(const float* x,const float* w,const float* g,float* stats,int time,int channels,int groups){int q=blockIdx.x;int sample=q/groups,gr=q%groups,cg=channels/groups,count=time*cg;__shared__ float sums[256],sumn[256];float a=0,b=0,mean=stats[q*4],inv=stats[q*4+1];for(int k=threadIdx.x;k<count;k+=blockDim.x){int at=k/cg,ch=gr*cg+k%cg,i=(sample*time+at)*channels+ch;float z=g[i]*w[ch];a+=z;b+=z*(x[i]-mean)*inv;}sums[threadIdx.x]=a;sumn[threadIdx.x]=b;__syncthreads();for(int step=128;step>0;step>>=1){if(threadIdx.x<step){sums[threadIdx.x]+=sums[threadIdx.x+step];sumn[threadIdx.x]+=sumn[threadIdx.x+step];}__syncthreads();}if(threadIdx.x==0){stats[q*4+2]=sums[0];stats[q*4+3]=sumn[0];}}
extern "C" __global__ void group_b(const float* x,const float* w,const float* g,const float* stats,float* dx,float* dw,float* db,int n,int time,int channels,int groups){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int ch=i%channels,gr=ch/(channels/groups),sample=i/(time*channels),q=sample*groups+gr,count=time*(channels/groups);float norm=(x[i]-stats[q*4])*stats[q*4+1];if(dx)dx[i]+=stats[q*4+1]*(count*g[i]*w[ch]-stats[q*4+2]-norm*stats[q*4+3])/count;if(dw)atomicAdd(&dw[ch],g[i]*norm);if(db)atomicAdd(&db[ch],g[i]);}}
extern "C" __global__ void masked_f(const float* pred,const float* target,float* accum,int n,int channels,int mse){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&!isnan(target[(i/channels)*channels])){float y=target[i];if(isnan(y))y=0;float d=pred[i]-y;atomicAdd(&accum[0],mse?d*d:fabsf(d));atomicAdd(&accum[1],1.0f);}}
extern "C" __global__ void masked_finish(const float* accum,float* loss){loss[0]=accum[1]>0?accum[0]/accum[1]:0.0f;}
extern "C" __global__ void masked_b(const float* pred,const float* target,const float* accum,const float* upstream,float* dx,int n,int channels,int mse){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&!isnan(target[(i/channels)*channels])&&accum[1]>0){float y=target[i];if(isnan(y))y=0;float d=pred[i]-y;dx[i]+=upstream[0]*(mse?2*d:d>0?1.0f:d<0?-1.0f:0.0f)/accum[1];}}
extern "C" __global__ void scale_grad(float* g,const float* norm,int n,float maxNorm){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float f=fminf(1.0f,maxNorm/(norm[0]+1e-6f));g[i]*=f;}}
extern "C" __global__ void global_norm_parts(const unsigned long long* ptrs,const int* sizes,float* norm,int count){int p=blockIdx.x;if(p>=count)return;const float* g=(const float*)ptrs[p];int n=sizes[p];__shared__ float sums[256];float sum=0;for(int i=threadIdx.x;i<n;i+=blockDim.x){float v=g[i];sum+=v*v;}sums[threadIdx.x]=sum;__syncthreads();for(int s=128;s>0;s>>=1){if(threadIdx.x<s)sums[threadIdx.x]+=sums[threadIdx.x+s];__syncthreads();}if(threadIdx.x==0)atomicAdd(norm,sums[0]);}
extern "C" __global__ void scale_all_grads(const unsigned long long* ptrs,const int* sizes,const float* norm,int count,float maxNorm){int p=blockIdx.x;if(p>=count)return;float* g=(float*)ptrs[p];int n=sizes[p];float f=fminf(1.0f,maxNorm/(norm[0]+1e-6f));for(int i=threadIdx.x;i<n;i+=blockDim.x)g[i]*=f;}
`

var graphCache cuda.KernelCache

func graphKernel(name string) *cuda.Kernel {
	functions, err := graphCache.Get(graphSource, "reduce_scalar_f", "reduce_last_f", "copy_values", "f32_to_bf16", "set_one", "seed_grad", "write_clip_entry", "view_gather", "view_gather_bf16", "view_scatter", "binary_nd_f", "binary_nd_bf16_f", "binary_nd_b", "reduce_nd_f", "reduce_nd_b", "split_heads_f", "split_heads_bf16_f", "split_heads_b", "merge_heads_f", "merge_heads_bf16_f", "merge_heads_b", "softmax_f", "softmax_b", "attn_softmax_f", "attn_softmax_b", "flash_attention_tiled_f", "flash_attention_tiled_f128", "flash_attention_delta", "flash_attention_tiled_b", "flash_attention_tiled_b128", "flash_attention_tiled_b_kv", "flash_attention_tiled_b_kv128", "layernorm_f", "layernorm_b", "layernorm_bf16_f", "bias_gelu_f", "bias_gelu_bf16_f", "bias_gelu_b", "bias_gelu_b_2d", "bias_residual_f", "bias_residual_bf16_f", "bias_residual_b", "bias_residual_b_2d", "dropout_f", "dropout_b", "ce_f", "ce_finish", "ce_b", "bce_logits_f", "bce_logits_b", "smooth_l1_f", "smooth_l1_b", "graph_adam_inc", "graph_adamw", "add_values", "scalar_f", "scalar_b", "clamp_f", "clamp_b", "pow_scalar_f", "pow_scalar_b", "binary_same_f", "binary_same_bf16_f", "binary_same_b", "bias_add_f", "bias_add_bf16_f", "bias_add_b", "binary_f", "binary_b", "binary_bf16_f", "unary_f", "unary_b", "permute_f", "permute_b", "concat_f", "concat_b", "slice_f", "slice_b", "embedding_f", "embedding_b", "reduce_f", "reduce_b", "group_stats", "group_f", "group_backstats", "group_b", "masked_f", "masked_finish", "masked_b", "scale_grad", "global_norm_parts", "scale_all_grads")
	if err != nil {
		panic(err)
	}
	return functions[name]
}
func launch(name string, n int, args ...unsafe.Pointer) {
	if n <= 0 {
		return
	}
	timedGPU(name, func() error {
		return graphKernel(name).Launch([3]int{(n + 255) / 256, 1, 1}, [3]int{256, 1, 1}, 0, nil, args)
	})
}
func ptr(b *cuda.Buffer) uintptr {
	if b == nil {
		return 0
	}
	return b.Pointer()
}
func copyDevice(dst, src *cuda.Buffer, n int) error {
	a, b, z := ptr(src), ptr(dst), int32(n)
	launch("copy_values", n, unsafe.Pointer(&a), unsafe.Pointer(&b), unsafe.Pointer(&z))
	return nil
}
func addDevice(dst, src *cuda.Buffer, n int) {
	a, b, z := ptr(dst), ptr(src), int32(n)
	launch("add_values", n, unsafe.Pointer(&a), unsafe.Pointer(&b), unsafe.Pointer(&z))
}
func setOne(dst *cuda.Buffer) { p := ptr(dst); launch("set_one", 1, unsafe.Pointer(&p)) }
func seedGrad(dst *cuda.Buffer, seed float32, accumulate bool) {
	p, add := ptr(dst), int32(0)
	if accumulate {
		add = 1
	}
	launch("seed_grad", 1, unsafe.Pointer(&p), unsafe.Pointer(&seed), unsafe.Pointer(&add))
}
