// SPDX-License-Identifier: Apache-2.0
// grid15-up producer and exact BF16 epilogue; PM opcode gaps logged in PM-GAPS.md.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ __launch_bounds__(256) void npu_trunk_pack_a_bf16(
    const __half* x, unsigned char* a, unsigned int* d, int rows, int k, int padded_rows) {
    const int t=blockIdx.x, tid=threadIdx.x, kc=k/64, waves=padded_rows/256;
    __shared__ float red[256];
    float am=0;
    if(t<rows) for(int i=tid;i<k;i+=256) am=fmaxf(am,fabsf(__half2float(x[(size_t)t*k+i])));
    red[tid]=am; __syncthreads();
    for(int s=128;s;s>>=1){ if(tid<s)red[tid]=fmaxf(red[tid],red[tid+s]); __syncthreads(); }
    am=red[0]; int m=0,e=0; float inv=0;
    if(am!=0){
        const float raw=__fdiv_rn(am,127.0f); int ex;
        const float f=frexpf(raw,&ex); e=ex-15;
        m=(int)ceilf(__fmul_rn(f,32768.0f));
        if(m==32768){m=16384;e++;}
        inv=__fdiv_rn(1.0f,ldexpf((float)m,e));
    }
    if(tid==0 && t<rows)d[t]=(unsigned int)m|((unsigned int)(unsigned short)e<<16);
    const int wave=t/256, rt=t%128, pair=(t%256)/128;
    for(int i=tid;i<k;i+=256){
        int q=0; if(t<rows && am!=0)q=__float2int_rn(__fmul_rn(__half2float(x[(size_t)t*k+i]),inv));
        q=q>127?127:(q< -127?-127:q);
        const int col=pair*2+(rt/8)%2, ch=i/64, kb=(i%64)/8;
        const size_t off=((size_t)col*waves*kc+(size_t)wave*kc+ch)*4608
            +(rt/16)*512+kb*64+(rt%8)*8+i%8;
        a[off]=(unsigned char)(signed char)q;
    }
    // Each token's scale is repeated in E and O halves for every K64.
    for(int ch=tid;ch<kc;ch+=256)for(int h=0;h<2;h++){
        const size_t off=((size_t)(pair*2+h)*waves*kc+(size_t)wave*kc+ch)*4608+4096;
        ((unsigned short*)(a+off))[rt]=(unsigned short)m;
        ((short*)(a+off+256))[rt]=(short)e;
    }
}
static __device__ __forceinline__ unsigned short direct(int isum,int sm,int se,int dm,int de){
    const long long j=((long long)isum*sm)*dm;
    if(j==0)return 0;
    const unsigned int sign=j<0?0x8000u:0u;
    const unsigned long long mag=j<0?(0ull-(unsigned long long)j):(unsigned long long)j;
    const int sh=se+de,l=63-(int)__clzll((long long)mag);
    int q=l+sh-7;if(q< -133)q=-133;
    if(q+133>255)return (unsigned short)(sign|0x7f80u);
    const int rs=q-sh; unsigned long long r;
    if(rs<=0)r=mag<<(-rs);
    else if(rs>=64)r=0;
    else{r=mag>>rs;const unsigned long long rem=mag&((1ull<<rs)-1),half=1ull<<(rs-1);
        if(rem>half||(rem==half&&(r&1)))r++;}
    unsigned int bits=((unsigned int)(q+133)<<7)+(unsigned int)r;
    if(bits>=0x7f80u)bits=0x7f80u;
    return (unsigned short)(sign|bits);
}
extern "C" __global__ __launch_bounds__(256) void npu_trunk_epi_bf16(
    const int* sums,const unsigned int* scales,const unsigned int* d,void* out,
    int rows,int n,int col0,int cols,int bf16){
    const int col=blockIdx.x*256+threadIdx.x,t=blockIdx.y;
    if(col>=cols||t>=rows)return;
    const unsigned int s=scales[col0+col],dt=d[t];
    const unsigned short bits=direct(sums[(size_t)t*n+col0+col],s&65535,(short)(s>>16),dt&65535,(short)(dt>>16));
    const size_t o=(size_t)t*cols+col;
    if(bf16)((unsigned short*)out)[o]=bits;
    else ((unsigned int*)out)[o]=(unsigned int)bits<<16;
}
