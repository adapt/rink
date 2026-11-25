// X[batch,cols] * W[rows,cols]^T. Dequantize only the current on-chip tile.
// FP16 inputs and weights feed FP32 SIMD matrix accumulators.
#include <metal_simdgroup_matrix>

template<uint M,bool Symmetric>
void bonsai_matrix(device const uint *w,device const half *scales,device const half *biases,
    device const half *x,device half *out,uint cols,uint rows,uint batch,
    uint group,uint tid,threadgroup float *scratch) {
    constexpr uint N=64,K=32,S=40;
    uint tiles=(rows+N-1)/N,row=(group%tiles)*N,token=(group/tiles)*M,sg=tid/32;
    threadgroup half *a=(threadgroup half*)scratch,*b=a+M*S;
    simdgroup_float8x8 sums[M/16][4];
    for(uint mi=0;mi<M/16;mi++)for(uint ni=0;ni<4;ni++)
        sums[mi][ni]=make_filled_simdgroup_matrix<float,8,8>(0.f);
    for(uint base=0;base<cols;base+=K) {
        for(uint i=tid;i<M*K/8;i+=128) {
            uint r=i/(K/8),c=(i%(K/8))*8;half4 left=0,right=0;
            if(token+r<batch){left=*((device const half4*)(x+(token+r)*cols+base+c));right=*((device const half4*)(x+(token+r)*cols+base+c+4));}
            *((threadgroup half4*)(a+r*S+c))=left;*((threadgroup half4*)(a+r*S+c+4))=right;
        }
        for(uint i=tid;i<N*K/16;i+=128) {
            uint r=i/(K/16),c=(i%(K/16))*16,index=(row+r)*cols+base+c;
            uint codes=0;float scale=0,bias=0;
            if(row+r<rows){codes=w[index/16];scale=float(scales[index/128]);bias=float(biases[index/128]);}
            for(uint j=0;j<4;j++){
                half4 value=0;
                uint4 unpacked=(uint4(codes)>>(uint4(0,2,4,6)+8*j))&3;
                if(Symmetric)value=(half4(unpacked)-half4(1))*half(scale);
                else value=half4(float4(unpacked)*scale+bias);
                *((threadgroup half4*)(b+r*S+c+4*j))=value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint k=0;k<K;k+=8) {
            simdgroup_half8x8 bv[4];
            for(uint ni=0;ni<4;ni++)
                simdgroup_load(bv[ni],b+((sg%2)*8+ni*16)*S+k,S,ulong2(0),true);
            for(uint mi=0;mi<M/16;mi++) {
                simdgroup_half8x8 av;simdgroup_load(av,a+((sg/2)*8+mi*16)*S+k,S);
                for(uint ni=0;ni<4;ni++)
                    simdgroup_multiply_accumulate(sums[mi][ni],av,bv[ni],sums[mi][ni]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Reuse input storage for 16 output rows at a time. This bounds threadgroup
    // storage without changing the per-output FP32 accumulation order.
    for(uint mi=0;mi<M/16;mi++) {
        for(uint ni=0;ni<4;ni++)
            simdgroup_store(sums[mi][ni],scratch+(sg/2)*8*N+(sg%2)*8+ni*16,N);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<16*N;i+=128) {
            uint t=token+mi*16+i/N,r=row+i%N;
            if(t<batch && r<rows)out[t*rows+r]=half(scratch[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define BONSAI_PACKED_MM(NAME,M,SYMMETRIC) \
kernel void NAME(device const uint *w [[buffer(0)]],device const half *s [[buffer(1)]], \
    device const half *b [[buffer(2)]],device const half *x [[buffer(3)]],device half *out [[buffer(4)]], \
    constant Params &p [[buffer(15)]],uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float scratch[(M+64)*40/2]; \
    bonsai_matrix<M,SYMMETRIC>(w,s,b,x,out,p.u[0],p.u[1],p.u[2],group,tid,scratch); \
}
BONSAI_PACKED_MM(packed_mm32,32,false)
BONSAI_PACKED_MM(packed_mm64,64,false)
// Only selected after the loader proves bias == -scale and |scale| <= 32752.
// All four codes then have exactly representable FP16 weights, including code 3.
BONSAI_PACKED_MM(packed_mm32_symmetric,32,true)
BONSAI_PACKED_MM(packed_mm64_symmetric,64,true)

// Gate matrices have only a few output rows. Small tiles expose enough
// workgroups to fill the GPU instead of running eight oversized workgroups.
kernel void dense_mm(device const half *w [[buffer(0)]],device const half *x [[buffer(1)]],
    device half *out [[buffer(2)]],constant Params &p [[buffer(15)]],
    uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint M=16,N=16,K=64,S=72;
    uint cols=p.u[0],rows=p.u[1],batch=p.u[2],tiles=(rows+N-1)/N;
    uint row=(group%tiles)*N,token=(group/tiles)*M,sg=tid/32;
    threadgroup float storage[1152];
    threadgroup half *a=(threadgroup half*)storage,*b=a+M*S;
    simdgroup_float8x8 sum=make_filled_simdgroup_matrix<float,8,8>(0.f);
    for(uint base=0;base<cols;base+=K) {
        for(uint i=tid;i<M*K/4;i+=128) {
            uint r=i/(K/4),c=(i%(K/4))*4;
            *((threadgroup half4*)(a+r*S+c))=token+r<batch?*((device const half4*)(x+(token+r)*cols+base+c)):half4(0);
            *((threadgroup half4*)(b+r*S+c))=row+r<rows?*((device const half4*)(w+(row+r)*cols+base+c)):half4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint k=0;k<K;k+=8) {
            simdgroup_half8x8 av,bv;
            simdgroup_load(av,a+(sg/2)*8*S+k,S);
            simdgroup_load(bv,b+(sg%2)*8*S+k,S,ulong2(0),true);
            simdgroup_multiply_accumulate(sum,av,bv,sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(sum,storage+(sg/2)*8*N+(sg%2)*8,N);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<M*N;i+=128) {
        uint t=token+i/N,r=row+i%N;
        if(t<batch && r<rows)out[t*rows+r]=half(storage[i]);
    }
}

// Gate/up fusion retains both FP16 projection boundaries before SiLU. Only
// matrices with exactly matching rotations and validated symmetric scales enter.
template<uint M>
void bonsai_mlp(device const uint *wg,device const half *sg,
    device const uint *wu,device const half *su,device const half *x,device half *out,
    constant Params &p,uint group,uint tid,threadgroup float *storage) {
    constexpr uint N=32,K=32,S=40,RM=M/16,RN=N/16;
    uint cols=p.u[0],rows=p.u[1],batch=p.u[2],tiles=(rows+N-1)/N;
    uint row=(group%tiles)*N,token=(group/tiles)*M,simd=tid/32;
    threadgroup half *a=(threadgroup half*)storage,*g=a+M*S,*u=g+N*S;
    simdgroup_float8x8 gs[RM][RN],us[RM][RN];
    for(uint mi=0;mi<RM;mi++)for(uint ni=0;ni<RN;ni++) {
        gs[mi][ni]=make_filled_simdgroup_matrix<float,8,8>(0.f);
        us[mi][ni]=make_filled_simdgroup_matrix<float,8,8>(0.f);
    }
    for(uint base=0;base<cols;base+=K) {
        for(uint i=tid;i<M*K/8;i+=128) {
            uint r=i/(K/8),c=(i%(K/8))*8;half4 left=0,right=0;
            if(token+r<batch) {
                left=*((device const half4*)(x+(token+r)*cols+base+c));
                right=*((device const half4*)(x+(token+r)*cols+base+c+4));
            }
            *((threadgroup half4*)(a+r*S+c))=left;
            *((threadgroup half4*)(a+r*S+c+4))=right;
        }
        for(uint i=tid;i<2*N*K/16;i+=128) {
            uint r=(i/(K/16))%N,c=(i%(K/16))*16;
            bool up=i>=N*K/16;uint codes=0;half scale=0;
            if(row+r<rows) {
                uint index=(row+r)*cols+base+c;
                codes=up?wu[index/16]:wg[index/16];
                scale=up?su[index/128]:sg[index/128];
            }
            for(uint j=0;j<4;j++) {
                half4 value=(half4((uint4(codes)>>(uint4(0,2,4,6)+8*j))&3)-half4(1))*scale;
                *((threadgroup half4*)((up?u:g)+r*S+c+4*j))=value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint k=0;k<K;k+=8) {
            simdgroup_half8x8 gv[RN],uv[RN];
            for(uint ni=0;ni<RN;ni++) {
                simdgroup_load(gv[ni],g+((simd%2)*8+ni*16)*S+k,S,ulong2(0),true);
                simdgroup_load(uv[ni],u+((simd%2)*8+ni*16)*S+k,S,ulong2(0),true);
            }
            for(uint mi=0;mi<RM;mi++) {
                simdgroup_half8x8 av;simdgroup_load(av,a+((simd/2)*8+mi*16)*S+k,S);
                for(uint ni=0;ni<RN;ni++) {
                    simdgroup_multiply_accumulate(gs[mi][ni],av,gv[ni],gs[mi][ni]);
                    simdgroup_multiply_accumulate(us[mi][ni],av,uv[ni],us[mi][ni]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint mi=0;mi<RM;mi++) {
        for(uint ni=0;ni<RN;ni++) {
            uint index=(simd/2)*8*N+(simd%2)*8+ni*16;
            simdgroup_store(gs[mi][ni],storage+index,N);
            simdgroup_store(us[mi][ni],storage+16*N+index,N);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<16*N;i+=128) {
            uint t=token+mi*16+i/N,r=row+i%N;
            if(t<batch && r<rows) {
                float gate=float(half(storage[i])),up=float(half(storage[16*N+i]));
                out[t*rows+r]=half((gate/(1.f+exp(-gate)))*up);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
#define BONSAI_MLP_MM(NAME,M) \
kernel void NAME(device const uint *wg [[buffer(0)]],device const half *sg [[buffer(1)]], \
    device const half *bg [[buffer(2)]],device const uint *wu [[buffer(3)]],device const half *su [[buffer(4)]], \
    device const half *bu [[buffer(5)]],device const half *x [[buffer(6)]],device half *y [[buffer(7)]], \
    constant Params &p [[buffer(15)]],uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float scratch[(M+64)*40/2]; \
    bonsai_mlp<M>(wg,sg,wu,su,x,y,p,group,tid,scratch); \
}
BONSAI_MLP_MM(mlp_mm32,32)
BONSAI_MLP_MM(mlp_mm,64)
