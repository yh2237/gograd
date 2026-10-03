package autograd

import (
	"github.com/yh2237/gograd/cuda"
	"sync"
	"time"
	"unsafe"
)

type OpTiming struct {
	Count    int
	Duration time.Duration
}

var gpuProfiling bool
var gpuTimings = map[string]OpTiming{}

func EnableGPUProfile(enabled bool) {
	gpuProfiling = enabled
	if enabled {
		gpuTimings = map[string]OpTiming{}
	}
}
func GPUProfile() map[string]OpTiming {
	out := map[string]OpTiming{}
	for k, v := range gpuTimings {
		out[k] = v
	}
	return out
}
func timedGPU(name string, fn func() error) {
	if !gpuProfiling {
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
	v := gpuTimings[name]
	v.Count++
	v.Duration += time.Duration(float64(ms) * float64(time.Millisecond))
	gpuTimings[name] = v
}

const graphSource = `
extern "C" __global__ void copy_values(const float* a,float* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i];}
extern "C" __global__ void f32_to_bf16(const float* x,unsigned short* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){unsigned int bits=__float_as_uint(x[i]);unsigned int round=0x7fffu+((bits>>16)&1u);y[i]=(unsigned short)((bits+round)>>16);}}
extern "C" __global__ void set_one(float* x){x[0]=1.0f;}
extern "C" __global__ void write_clip_entry(unsigned long long* ptrs,int* sizes,int index,unsigned long long address,int size){ptrs[index]=address;sizes[index]=size;}
struct Axes6 {int d[6];};
__device__ int view_index(int i,Axes6 shape,Axes6 stride,int offset){int j=offset;for(int axis=5;axis>=0;axis--){int dim=shape.d[axis];int c=i%dim;i/=dim;j+=c*stride.d[axis];}return j;}
extern "C" __global__ void view_gather(const float* x,float* y,int n,Axes6 shape,Axes6 stride,int offset){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=x[view_index(i,shape,stride,offset)];}
extern "C" __global__ void view_scatter(const float* g,float* dx,int n,Axes6 shape,Axes6 stride,int offset){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)atomicAdd(&dx[view_index(i,shape,stride,offset)],g[i]);}
extern "C" __global__ void binary_nd_f(const float* a,const float* b,float* y,int n,Axes6 shape,Axes6 as,Axes6 bs,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float x=a[view_index(i,shape,as,0)],z=b[view_index(i,shape,bs,0)];y[i]=op==0?x+z:op==1?x-z:op==2?x*z:x/z;}
extern "C" __global__ void binary_nd_b(const float* a,const float* b,const float* g,float* da,float* db,int n,Axes6 shape,Axes6 as,Axes6 bs,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int ai=view_index(i,shape,as,0),bi=view_index(i,shape,bs,0);float x=a[ai],z=b[bi],up=g[i];if(da)atomicAdd(&da[ai],up*(op==2?z:op==3?1.0f/z:1.0f));if(db)atomicAdd(&db[bi],up*(op==0?1.0f:op==1?-1.0f:op==2?x:-x/(z*z)));}
extern "C" __global__ void reduce_nd_f(const float* x,float* y,int n,Axes6 shape,Axes6 map,float scale){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)atomicAdd(&y[view_index(i,shape,map,0)],x[i]*scale);}
extern "C" __global__ void reduce_nd_b(const float* g,float* dx,int n,Axes6 shape,Axes6 map,float scale){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dx[i]+=g[view_index(i,shape,map,0)]*scale;}
__device__ float warp_sum(float x){for(int delta=16;delta>0;delta>>=1)x+=__shfl_down_sync(0xffffffff,x,delta);return __shfl_sync(0xffffffff,x,0);}
__device__ float warp_max(float x){for(int delta=16;delta>0;delta>>=1)x=fmaxf(x,__shfl_down_sync(0xffffffff,x,delta));return __shfl_sync(0xffffffff,x,0);}
extern "C" __global__ void softmax_f(const float* x,float* y,int rows,int dim,int inner,int logmode){int tid=blockIdx.x*blockDim.x+threadIdx.x,row=tid>>5,lane=tid&31;if(row>=rows)return;int o=row/inner,j=row%inner,base=o*dim*inner+j;float mx=-3.402823466e38f;for(int k=lane;k<dim;k+=32)mx=fmaxf(mx,x[base+k*inner]);mx=warp_max(mx);float sum=0;for(int k=lane;k<dim;k+=32)sum+=expf(x[base+k*inner]-mx);sum=warp_sum(sum);float ls=logf(sum);for(int k=lane;k<dim;k+=32){int idx=base+k*inner;y[idx]=logmode?x[idx]-mx-ls:expf(x[idx]-mx)/sum;}}
extern "C" __global__ void softmax_b(const float* y,const float* g,float* dx,int rows,int dim,int inner,int logmode){int tid=blockIdx.x*blockDim.x+threadIdx.x,row=tid>>5,lane=tid&31;if(row>=rows)return;int o=row/inner,j=row%inner,base=o*dim*inner+j;float dot=0;for(int k=lane;k<dim;k+=32){int idx=base+k*inner;dot+=g[idx]*(logmode?1.0f:y[idx]);}dot=warp_sum(dot);for(int k=lane;k<dim;k+=32){int idx=base+k*inner;dx[idx]+=logmode?g[idx]-expf(y[idx])*dot:y[idx]*(g[idx]-dot);}}
extern "C" __global__ void attn_softmax_f(const float* scores,const float* mask,float* probs,int rows,int dim,float scale,Axes6 shape,Axes6 ms){int tid=blockIdx.x*blockDim.x+threadIdx.x,row=tid>>5,lane=tid&31;if(row>=rows)return;int base=row*dim;float mx=-3.402823466e38f;for(int k=lane;k<dim;k+=32){int i=base+k;float z=scores[i]*scale+(mask?mask[view_index(i,shape,ms,0)]:0.0f);mx=fmaxf(mx,z);}mx=warp_max(mx);float sum=0;for(int k=lane;k<dim;k+=32){int i=base+k;float z=scores[i]*scale+(mask?mask[view_index(i,shape,ms,0)]:0.0f);sum+=expf(z-mx);}sum=warp_sum(sum);for(int k=lane;k<dim;k+=32){int i=base+k;float z=scores[i]*scale+(mask?mask[view_index(i,shape,ms,0)]:0.0f);probs[i]=expf(z-mx)/sum;}}
extern "C" __global__ void attn_softmax_b(const float* probs,const float* dp,float* ds,float* dmask,int rows,int dim,float scale,Axes6 shape,Axes6 ms){int tid=blockIdx.x*blockDim.x+threadIdx.x,row=tid>>5,lane=tid&31;if(row>=rows)return;int base=row*dim;float dot=0;for(int k=lane;k<dim;k+=32){int i=base+k;dot+=probs[i]*dp[i];}dot=warp_sum(dot);for(int k=lane;k<dim;k+=32){int i=base+k;float z=probs[i]*(dp[i]-dot);ds[i]=z*scale;if(dmask)atomicAdd(&dmask[view_index(i,shape,ms,0)],z);}}
extern "C" __global__ void layernorm_f(const float* x,const float* w,const float* b,float* y,float* stats,int rows,int dim,float eps){int tid=blockIdx.x*blockDim.x+threadIdx.x,row=tid>>5,lane=tid&31;if(row>=rows)return;int base=row*dim;float sum=0;for(int j=lane;j<dim;j+=32)sum+=x[base+j];float mean=warp_sum(sum)/(float)dim;float ss=0;for(int j=lane;j<dim;j+=32){float z=x[base+j]-mean;ss+=z*z;}float inv=rsqrtf(warp_sum(ss)/(float)dim+eps);if(lane==0){stats[row*2]=mean;stats[row*2+1]=inv;}for(int j=lane;j<dim;j+=32)y[base+j]=(x[base+j]-mean)*inv*w[j]+b[j];}
extern "C" __global__ void layernorm_b(const float* x,const float* w,const float* g,const float* stats,float* dx,float* dw,float* db,int rows,int dim){int tid=blockIdx.x*blockDim.x+threadIdx.x,row=tid>>5,lane=tid&31;if(row>=rows)return;int base=row*dim;float mean=stats[row*2],inv=stats[row*2+1];float sg=0,sgn=0;for(int j=lane;j<dim;j+=32){int i=base+j;float norm=(x[i]-mean)*inv,z=g[i]*w[j];sg+=z;sgn+=z*norm;}sg=warp_sum(sg)/(float)dim;sgn=warp_sum(sgn)/(float)dim;for(int j=lane;j<dim;j+=32){int i=base+j;float norm=(x[i]-mean)*inv;if(dx)dx[i]+=inv*(g[i]*w[j]-sg-norm*sgn);if(dw)atomicAdd(&dw[j],g[i]*norm);if(db)atomicAdd(&db[j],g[i]);}}
__device__ unsigned int dropout_hash(unsigned int i,unsigned int seed){unsigned int x=i^seed;x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x;}
extern "C" __global__ void dropout_f(const float* x,float* y,int n,float p,unsigned int seed,unsigned int cutoff){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=dropout_hash(i,seed)<cutoff?0.0f:x[i]/(1.0f-p);}
extern "C" __global__ void dropout_b(const float* g,float* dx,int n,float p,unsigned int seed,unsigned int cutoff){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dx[i]+=dropout_hash(i,seed)<cutoff?0.0f:g[i]/(1.0f-p);}
extern "C" __global__ void ce_f(const float* x,const int* target,float* acc,int rows,int classes,int ignore){int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows||target[row]==ignore)return;int base=row*classes;float mx=-3.402823466e38f;for(int j=0;j<classes;j++)mx=fmaxf(mx,x[base+j]);float sum=0;for(int j=0;j<classes;j++)sum+=expf(x[base+j]-mx);atomicAdd(&acc[0],mx+logf(sum)-x[base+target[row]]);atomicAdd(&acc[1],1.0f);}
extern "C" __global__ void ce_finish(const float* acc,float* y){y[0]=acc[0]/acc[1];}
extern "C" __global__ void ce_b(const float* x,const int* target,const float* acc,const float* g,float* dx,int rows,int classes,int ignore){int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows||target[row]==ignore||acc[1]==0)return;int base=row*classes;float mx=-3.402823466e38f;for(int j=0;j<classes;j++)mx=fmaxf(mx,x[base+j]);float sum=0;for(int j=0;j<classes;j++)sum+=expf(x[base+j]-mx);float up=g[0]/acc[1];for(int j=0;j<classes;j++)dx[base+j]+=up*(expf(x[base+j]-mx)/sum-(j==target[row]?1.0f:0.0f));}
extern "C" __global__ void graph_adam_inc(int* step){if(threadIdx.x==0&&blockIdx.x==0)step[0]++;}
extern "C" __global__ void graph_adamw(float* p,const float* g,float* m,float* v,const int* step,int n,float b1,float b2,float lr,float wd,float eps){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float grad=g[i];float mm=b1*m[i]+(1.0f-b1)*grad;float vv=b2*v[i]+(1.0f-b2)*grad*grad;m[i]=mm;v[i]=vv;float bc1=1.0f-powf(b1,(float)step[0]),bc2=1.0f-powf(b2,(float)step[0]);p[i]=p[i]*(1.0f-lr*wd)-lr*(mm/bc1)/(sqrtf(vv/bc2)+eps);}
extern "C" __global__ void add_values(float* a,const float* b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)a[i]+=b[i];}
extern "C" __global__ void scalar_f(const float* a,float* out,int n,float value,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float x=a[i];out[i]=op==0?x+value:op==1?x-value:op==2?x*value:x/value;}}
extern "C" __global__ void scalar_b(const float* g,float* da,int n,float value,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)da[i]+=g[i]*(op==2?value:op==3?1.0f/value:1.0f);}
__device__ int index3(int i,int d1,int d2,int s0,int s1,int s2){int z=i%d2;int y=(i/d2)%d1;int x=i/(d1*d2);return (s0==1?0:x*s1*s2)+(s1==1?0:y*s2)+(s2==1?0:z);}
extern "C" __global__ void binary_f(const float* a,const float* b,float* out,int n,int d1,int d2,int a0,int a1,int a2,int b0,int b1,int b2,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float x=a[index3(i,d1,d2,a0,a1,a2)],y=b[index3(i,d1,d2,b0,b1,b2)];out[i]=op==0?x+y:op==1?x-y:op==2?x*y:x/y;}
extern "C" __global__ void binary_b(const float* a,const float* b,const float* g,float* da,float* db,int n,int d1,int d2,int a0,int a1,int a2,int b0,int b1,int b2,int op){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int ai=index3(i,d1,d2,a0,a1,a2),bi=index3(i,d1,d2,b0,b1,b2);float x=a[ai],y=b[bi],z=g[i];if(da)atomicAdd(&da[ai],z*(op==2?y:op==3?1.0f/y:1.0f));if(db)atomicAdd(&db[bi],z*(op==0?1.0f:op==1?-1.0f:op==2?x:-x/(y*y)));}
__device__ float fun(float x,int op){if(op==0)return expf(x);if(op==1)return logf(x);if(op==2)return fabsf(x);if(op==3)return tanhf(x);if(op==4)return fmaxf(x,0.0f);if(op==5)return .5f*x*(1.0f+erff(x*.7071067811865475f));float u=.7978845608028654f*(x+.044715f*x*x*x);return .5f*x*(1.0f+tanhf(u));}
__device__ float deriv(float x,int op){if(op==0)return expf(x);if(op==1)return 1.0f/x;if(op==2)return x>0?1.0f:x<0?-1.0f:0.0f;if(op==3){float y=tanhf(x);return 1-y*y;}if(op==4)return x>0?1.0f:0.0f;if(op==5)return .5f*(1.0f+erff(x*.7071067811865475f))+x*expf(-.5f*x*x)*.3989422804014327f;float u=.7978845608028654f*(x+.044715f*x*x*x),y=tanhf(u);return .5f*(1+y)+.5f*x*(1-y*y)*.7978845608028654f*(1+3*.044715f*x*x);}
extern "C" __global__ void bias_gelu_f(const float* x,const float* b,float* y,int n,int c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=fun(x[i]+b[i%c],5);}
extern "C" __global__ void bias_gelu_b(const float* x,const float* b,const float* g,float* dx,float* db,int n,int c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=g[i]*deriv(x[i]+b[i%c],5);if(dx)dx[i]+=z;if(db)atomicAdd(&db[i%c],z);}}
extern "C" __global__ void bias_residual_f(const float* x,const float* b,const float* r,float* y,int n,int c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=x[i]+b[i%c]+r[i];}
extern "C" __global__ void bias_residual_b(const float* g,float* dx,float* db,float* dr,int n,int c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=g[i];if(dx)dx[i]+=z;if(db)atomicAdd(&db[i%c],z);if(dr)dr[i]+=z;}}
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
extern "C" __global__ void group_stats(const float* x,float* stats,int batch,int time,int channels,int groups,float eps){int q=blockIdx.x;if(q>=batch*groups)return;int n=q/groups,gr=q%groups,cg=channels/groups,count=time*cg;__shared__ float sums[256],squares[256];float sum=0,sq=0;for(int k=threadIdx.x;k<count;k+=blockDim.x){int at=k/cg,ch=gr*cg+k%cg;float v=x[(n*time+at)*channels+ch];sum+=v;sq+=v*v;}sums[threadIdx.x]=sum;squares[threadIdx.x]=sq;__syncthreads();for(int step=128;step>0;step>>=1){if(threadIdx.x<step){sums[threadIdx.x]+=sums[threadIdx.x+step];squares[threadIdx.x]+=squares[threadIdx.x+step];}__syncthreads();}if(threadIdx.x==0){float mean=sums[0]/count,var=fmaxf(squares[0]/count-mean*mean,0.0f);stats[q*4]=mean;stats[q*4+1]=rsqrtf(var+eps);}}
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

var graphOnce sync.Once
var graphProgram *cuda.Program
var graphKernels map[string]*cuda.Kernel
var graphErr error

func graphKernel(name string) *cuda.Kernel {
	graphOnce.Do(func() {
		graphProgram, graphErr = cuda.Compile(graphSource)
		if graphErr != nil {
			return
		}
		graphKernels = map[string]*cuda.Kernel{}
		for _, n := range []string{"copy_values", "f32_to_bf16", "set_one", "write_clip_entry", "view_gather", "view_scatter", "binary_nd_f", "binary_nd_b", "reduce_nd_f", "reduce_nd_b", "softmax_f", "softmax_b", "attn_softmax_f", "attn_softmax_b", "layernorm_f", "layernorm_b", "bias_gelu_f", "bias_gelu_b", "bias_residual_f", "bias_residual_b", "dropout_f", "dropout_b", "ce_f", "ce_finish", "ce_b", "graph_adam_inc", "graph_adamw", "add_values", "scalar_f", "scalar_b", "binary_f", "binary_b", "unary_f", "unary_b", "permute_f", "permute_b", "concat_f", "concat_b", "slice_f", "slice_b", "embedding_f", "embedding_b", "reduce_f", "reduce_b", "group_stats", "group_f", "group_backstats", "group_b", "masked_f", "masked_finish", "masked_b", "scale_grad", "global_norm_parts", "scale_all_grads"} {
			graphKernels[n], graphErr = graphProgram.Function(n)
			if graphErr != nil {
				return
			}
		}
	})
	if graphErr != nil {
		panic(graphErr)
	}
	return graphKernels[name]
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
