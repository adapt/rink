// Benchmark-only GEMM tiling experiment. Uses the production decode helpers and
// the same ordered FP32 8x8 matrix operations. No extra device buffers.
template<uint M, uint N, uint K, bool Expert>
void probe_gemm(device float* y, device const uchar* w, device const uchar* scales,
    device const float* factors, device const float* x, device const uint* groups,
    device const uint* tiles, device const float* table, constant Params& p,
    uint group, uint tid, uint lane, threadgroup float* scratch) {
    uint rows=p.u[0],cols=p.u[1],row_tiles=(rows+N-1)/N,tile=group/row_tiles;
    if(Expert && tile>=tiles[0])return;
    uint row=(group%row_tiles)*N,expert=Expert?tiles[1+2*tile]:0;
    uint first=Expert?tiles[2+2*tile]:tile*M,stride=p.u[5]+1;
    uint count=Expert?groups[expert*stride]:p.u[5],sg=tid/32;
    float factor=Expert && p.u[3]==0?factors[expert]:1.f;
    threadgroup float* a=scratch;
    threadgroup float* b=scratch+M*K;
    simdgroup_float8x8 sum[M/16][N/32];
    for(uint mi=0;mi<M/16;mi++)for(uint ni=0;ni<N/32;ni++)
        sum[mi][ni]=make_filled_simdgroup_matrix<float,8,8>(0.f);
    uint padded=((cols+31)/32)*32;
    for(uint base=0;base<padded;base+=K){
        for(uint i=tid;i<M*K/4;i+=256){
            uint r=i/(K/4),c=(i%(K/4))*4;float4 av=0;
            if(first+r<count){
                uint slot=Expert?groups[expert*stride+1+first+r]:first+r;
                uint src=(Expert && !p.u[4]?slot/p.u[2]:slot)*cols;
                if(base+c+4<=cols)av=float4(*((device const packed_float4*)(x+src+base+c)));
                else for(uint j=0;j<4;j++)if(base+c+j<cols)av[j]=x[src+base+c+j];
            }
            for(uint j=0;j<4;j++)a[r*K+c+j]=av[j];
        }
        for(uint i=tid;i<N*K/4;i+=256){
            uint r=i/(K/4),c=(i%(K/4))*4;float4 bv=0;
            if(row+r<rows){
                uint idx=(expert*rows+row+r)*cols+base+c;
                if(base+c+4<=cols){
                    if(!Expert)bv=weight4(w,idx,p.u[3]);
                    else bv=p.u[3]==0?expert_lookup4(w,scales,table,idx,expert):expert_weight4(w,scales,idx,p.u[3],factor);
                }else for(uint j=0;j<4;j++)if(base+c+j<cols){
                    if(!Expert)bv[j]=weight(w,idx+j,p.u[3]);
                    else if(p.u[3]==2)bv[j]=((device const float*)w)[idx+j];
                }
            }
            for(uint j=0;j<4;j++)b[(c+j)*N+r]=bv[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint k=0;k<min(K,padded-base);k+=8){
            for(uint mi=0;mi<M/16;mi++){
                simdgroup_float8x8 av;
                simdgroup_load(av,a+((sg/4)*8+mi*16)*K+k,K);
                for(uint ni=0;ni<N/32;ni++){
                    simdgroup_float8x8 bv;
                    simdgroup_load(bv,b+k*N+(sg%4)*8+ni*32,N);
                    simdgroup_multiply_accumulate(sum[mi][ni],av,bv,sum[mi][ni]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Staging is dead after the last barrier; reuse it for the accumulator store.
    for(uint mi=0;mi<M/16;mi++)for(uint ni=0;ni<N/32;ni++)
        simdgroup_store(sum[mi][ni],scratch+((sg/4)*8+mi*16)*N+(sg%4)*8+ni*32,N);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<M*N;i+=256){
        uint t=first+i/N,r=row+i%N;
        if(t<count && r<rows){uint slot=Expert?groups[expert*stride+1+t]:t;y[slot*rows+r]=scratch[i];}
    }
}

#define PROBE(NAME,M,N,K,E) \
kernel void NAME(device float* y [[buffer(0)]],device const uchar* w [[buffer(1)]],device const uchar* scales [[buffer(2)]],device const float* factors [[buffer(3)]],device const float* x [[buffer(4)]],device const uint* groups [[buffer(5)]],device const uint* tiles [[buffer(6)]],device const float* table [[buffer(7)]],constant Params& p [[buffer(15)]],uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float scratch[((M+N)*K>M*N)?(M+N)*K:M*N]; \
    probe_gemm<M,N,K,E>(y,w,scales,factors,x,groups,tiles,table,p,group,tid,lane,scratch); \
}

PROBE(dense_16_32_32,16,32,32,false)
PROBE(dense_32_32_32,32,32,32,false)
PROBE(dense_32_32_64,32,32,64,false)
PROBE(dense_32_32_128,32,32,128,false)
PROBE(dense_32_64_32,32,64,32,false)
PROBE(dense_32_64_64,32,64,64,false)
PROBE(dense_64_32_32,64,32,32,false)
PROBE(dense_64_64_32,64,64,32,false)
PROBE(dense_64_64_64,64,64,64,false)
PROBE(expert_16_32_32,16,32,32,true)
PROBE(expert_32_32_32,32,32,32,true)
PROBE(expert_32_32_64,32,32,64,true)
PROBE(expert_32_32_128,32,32,128,true)
PROBE(expert_32_64_32,32,64,32,true)
PROBE(expert_32_64_64,32,64,64,true)
PROBE(expert_64_32_32,64,32,32,true)
PROBE(expert_64_64_32,64,64,32,true)
PROBE(expert_64_64_64,64,64,64,true)
#undef PROBE
