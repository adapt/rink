// X[batch,cols] * W[rows,cols]^T. Dequantize only the current on-chip tile.
// FP16 inputs and weights feed FP32 SIMD matrix accumulators.
#include <metal_simdgroup_matrix>

template<uint M,bool Packed>
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
            if(Packed && row+r<rows){codes=w[index/16];scale=float(scales[index/128]);bias=float(biases[index/128]);}
            for(uint j=0;j<4;j++){
                half4 value=0;
                if(Packed)value=half4(float4((uint4(codes)>>(uint4(0,2,4,6)+8*j))&3)*scale+bias);
                else if(row+r<rows)value=*((device const half4*)(((device const half*)w)+index+4*j));
                *((threadgroup half4*)(b+r*S+c+4*j))=value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint k=0;k<K;k+=8)for(uint mi=0;mi<M/16;mi++) {
            simdgroup_half8x8 av;simdgroup_load(av,a+((sg/2)*8+mi*16)*S+k,S);
            for(uint ni=0;ni<4;ni++) {
                simdgroup_half8x8 bv;simdgroup_load(bv,b+((sg%2)*8+ni*16)*S+k,S,ulong2(0),true);
                simdgroup_multiply_accumulate(sums[mi][ni],av,bv,sums[mi][ni]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint mi=0;mi<M/16;mi++)for(uint ni=0;ni<4;ni++)
        simdgroup_store(sums[mi][ni],scratch+((sg/2)*8+mi*16)*N+(sg%2)*8+ni*16,N);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<M*N;i+=128) {
        uint t=token+i/N,r=row+i%N;
        if(t<batch && r<rows)out[t*rows+r]=half(scratch[i]);
    }
}

#define BONSAI_PACKED_MM(NAME,M) \
kernel void NAME(device const uint *w [[buffer(0)]],device const half *s [[buffer(1)]], \
    device const half *b [[buffer(2)]],device const half *x [[buffer(3)]],device half *out [[buffer(4)]], \
    constant Params &p [[buffer(15)]],uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float scratch[M*64]; \
    bonsai_matrix<M,true>(w,s,b,x,out,p.u[0],p.u[1],p.u[2],group,tid,scratch); \
}
BONSAI_PACKED_MM(packed_mm32,32)
BONSAI_PACKED_MM(packed_mm64,64)

kernel void dense_mm(device const uint *w [[buffer(0)]],device const half *x [[buffer(1)]],
    device half *out [[buffer(2)]],constant Params &p [[buffer(15)]],
    uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[4096];
    bonsai_matrix<64,false>(w,nullptr,nullptr,x,out,p.u[0],p.u[1],p.u[2],group,tid,scratch);
}
