// Large prefill tiles keep the existing ordered FP32 8x8 matrix operations.
// Scratch is on-chip and reused for output; no new device buffers.
template<uint M, uint N, uint K, bool Expert>
void prefill_matrix_tile(device float* y, device const uchar* w, device const uchar* scales,
    device const float* factors, device const float* x, device const uint* groups,
    device const uint* tiles, device const float* table,
    uint rows, uint cols, uint top, uint format, bool down, uint batch, uint input_offset,
    uint group, uint tid, uint lane, threadgroup float* scratch) {
    uint row_tiles=(rows+N-1)/N,tile=group/row_tiles;
    if(Expert && tile>=tiles[0])return;
    uint row=(group%row_tiles)*N,expert=Expert?tiles[1+2*tile]:0;
    uint first=Expert?tiles[2+2*tile]:tile*M,stride=batch+1;
    uint count=Expert?groups[expert*stride]:batch,sg=tid/32;
    float factor=Expert && format==0?factors[expert]:1.f;
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
                uint src=(Expert && !down?slot/top:slot)*cols+input_offset;
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
                    if(!Expert)bv=weight4(w,idx,format);
                    else bv=format==0?expert_lookup4(w,scales,table,idx,expert):expert_weight4(w,scales,idx,format,factor);
                }else for(uint j=0;j<4;j++)if(base+c+j<cols){
                    if(!Expert)bv[j]=weight(w,idx+j,format);
                    else if(format==2)bv[j]=((device const float*)w)[idx+j];
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


// Register prefetch for 64 routed slots by 64 output channels. Keep one shared tile, issue the next tile's
// global loads before computing the current tile, then reuse shared storage.
template<uint M,uint N,uint K,bool Expert>
void prefill_expert_pipeline(device float* y,device const uchar* w,device const uchar* scales,device const float* factors,device const float* x,device const uint* groups,device const uint* tiles,device const float* table,uint rows,uint cols,uint top,uint format,bool down,uint batch,uint group,uint tid,threadgroup float* scratch){
    constexpr uint A=M*K/1024,B=N*K/1024;
    uint row_tiles=(rows+N-1)/N,tile=group/row_tiles;
    if(Expert && tile>=tiles[0])return;
    uint row=(group%row_tiles)*N,expert=Expert?tiles[1+2*tile]:0,first=Expert?tiles[2+2*tile]:tile*M;
    uint stride=batch+1,count=Expert?groups[expert*stride]:batch,sg=tid/32,padded=((cols+31)/32)*32;
    float factor=Expert && format==0?factors[expert]:1.f;
    uint ai[A],bi[B];bool av[A],bv[B];
    for(uint t=0;t<A;t++){
        uint i=tid+t*256,r=i/(K/4),c=(i%(K/4))*4;av[t]=first+r<count;
        uint slot=av[t]?(Expert?groups[expert*stride+1+first+r]:first+r):0;
        ai[t]=(Expert && !down?slot/top:slot)*cols+c;
    }
    for(uint t=0;t<B;t++){uint i=tid+t*256,r=i/(K/4),c=(i%(K/4))*4;bv[t]=row+r<rows;bi[t]=(expert*rows+row+r)*cols+c;}
    threadgroup float* a=scratch;threadgroup float* b=scratch+M*K;
    simdgroup_float8x8 sum[M/16][N/32];
    for(uint mi=0;mi<M/16;mi++)for(uint ni=0;ni<N/32;ni++)sum[mi][ni]=make_filled_simdgroup_matrix<float,8,8>(0.f);
    float4 next_a[A],next_b[B];
    // Initial load, followed by one prefetched tile per loop iteration.
    for(uint iteration=0;iteration<=padded/K+(padded%K!=0);iteration++){
        uint base=iteration*K;
        if(base<padded){
            for(uint t=0;t<A;t++){
                uint c=((tid+t*256)%(K/4))*4;float4 value=0;
                if(av[t]){
                    if(base+c+4<=cols)value=float4(*((device const packed_float4*)(x+ai[t]+base)));
                    else for(uint j=0;j<4;j++)if(base+c+j<cols)value[j]=x[ai[t]+base+j];
                }
                next_a[t]=value;
            }
            for(uint t=0;t<B;t++){
                uint c=((tid+t*256)%(K/4))*4,idx=bi[t]+base;float4 value=0;
                if(bv[t]){
                    if(base+c+4<=cols){
                        if(!Expert)value=weight4(w,idx,format);
                        else value=format==0?expert_lookup4(w,scales,table,idx,expert):expert_weight4(w,scales,idx,format,factor);
                    }else for(uint j=0;j<4;j++)if(base+c+j<cols){
                        if(!Expert)value[j]=weight(w,idx+j,format);
                        else if(format==2)value[j]=((device const float*)w)[idx+j];
                    }
                }
                next_b[t]=value;
            }
        }
        if(iteration){
            for(uint k=0;k<min(K,padded-(base-K));k+=8){
                for(uint mi=0;mi<M/16;mi++){
                    simdgroup_float8x8 ax;simdgroup_load(ax,a+((sg/4)*8+mi*16)*K+k,K);
                    for(uint ni=0;ni<N/32;ni++){
                        simdgroup_float8x8 bx;simdgroup_load(bx,b+k*N+(sg%4)*8+ni*32,N);
                        simdgroup_multiply_accumulate(sum[mi][ni],ax,bx,sum[mi][ni]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if(base<padded){
            for(uint t=0;t<A;t++){uint i=tid+t*256,r=i/(K/4),c=(i%(K/4))*4;for(uint j=0;j<4;j++)a[r*K+c+j]=next_a[t][j];}
            for(uint t=0;t<B;t++){uint i=tid+t*256,r=i/(K/4),c=(i%(K/4))*4;for(uint j=0;j<4;j++)b[(c+j)*N+r]=next_b[t][j];}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for(uint mi=0;mi<M/16;mi++)for(uint ni=0;ni<N/32;ni++)simdgroup_store(sum[mi][ni],scratch+((sg/4)*8+mi*16)*N+(sg%4)*8+ni*32,N);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<M*N;i+=256){uint t=first+i/N,r=row+i%N;if(t<count && r<rows){uint slot=Expert?groups[expert*stride+1+t]:t;y[slot*rows+r]=scratch[i];}}
}

// Same three-buffer ABI and geometry as matmul_packed. M tokens by 64 outputs.
#define PREFILL_DENSE(NAME,M) \
kernel void NAME(device float* y [[buffer(0)]],device const uchar* w [[buffer(1)]],device const float* x [[buffer(2)]],constant Params& p [[buffer(15)]],uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float scratch[(M+64)*32>M*64?(M+64)*32:M*64]; \
    prefill_matrix_tile<M,64,32,false>(y,w,nullptr,nullptr,x,nullptr,nullptr,nullptr,p.u[0],p.u[1],1,p.u[2],false,p.u[3],p.u[4],group,tid,lane,scratch); \
}
PREFILL_DENSE(matmul_prefill_m32,32)
PREFILL_DENSE(matmul_prefill_m64,64)
#undef PREFILL_DENSE

// 64-slot expert tile list. Small final tiles compute only 16 or 32 slots;
// each 8x8 accumulator retains the same K order and 32-column zero padding.
kernel void expert_matmul_prefill(device float* y [[buffer(0)]],device const uchar* w [[buffer(1)]],device const uchar* scales [[buffer(2)]],device const float* factors [[buffer(3)]],device const float* x [[buffer(4)]],device const uint* groups [[buffer(5)]],device const uint* tiles [[buffer(6)]],device const float* table [[buffer(7)]],constant Params& p [[buffer(15)]],uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[4096];
    uint tile=group/((p.u[0]+63)/64);if(tile>=tiles[0])return;
    uint expert=tiles[1+2*tile],first=tiles[2+2*tile],remaining=groups[expert*(p.u[5]+1)]-first;
    if(remaining<=16)prefill_matrix_tile<16,64,32,true>(y,w,scales,factors,x,groups,tiles,table,p.u[0],p.u[1],p.u[2],p.u[3],p.u[4],p.u[5],0,group,tid,lane,scratch);
    else if(remaining<=32)prefill_matrix_tile<32,64,32,true>(y,w,scales,factors,x,groups,tiles,table,p.u[0],p.u[1],p.u[2],p.u[3],p.u[4],p.u[5],0,group,tid,lane,scratch);
    else prefill_expert_pipeline<64,64,16,true>(y,w,scales,factors,x,groups,tiles,table,p.u[0],p.u[1],p.u[2],p.u[3],p.u[4],p.u[5],group,tid,scratch);
}

// Native dimensions and format are compile-time constants; the arithmetic and
// adaptive 16/32/64-slot tiles match expert_matmul_prefill.
template<uint Cols,uint Format,bool Down>
void prefill_native_expert_tile(device float* y,device const uchar* w,device const uchar* scales,device const float* factors,device const float* x,device const uint* groups,device const uint* tiles,device const float* table,constant Params& p,uint group,uint tid,uint lane,threadgroup float* scratch){
    uint tile=group/36;if(tile>=tiles[0])return;
    uint expert=tiles[1+2*tile],first=tiles[2+2*tile],remaining=groups[expert*(p.u[5]+1)]-first;
    if(remaining<=16)prefill_matrix_tile<16,64,32,true>(y,w,scales,factors,x,groups,tiles,table,2304,Cols,8,Format,Down,p.u[5],0,group,tid,lane,scratch);
    else if(remaining<=32)prefill_matrix_tile<32,64,32,true>(y,w,scales,factors,x,groups,tiles,table,2304,Cols,8,Format,Down,p.u[5],0,group,tid,lane,scratch);
    else prefill_expert_pipeline<64,64,16,true>(y,w,scales,factors,x,groups,tiles,table,2304,Cols,8,Format,Down,p.u[5],group,tid,scratch);
}
#define NATIVE_EXPERT(NAME,COLS,FORMAT,DOWN) \
kernel void NAME(device float* y [[buffer(0)]],device const uchar* w [[buffer(1)]],device const uchar* scales [[buffer(2)]],device const float* factors [[buffer(3)]],device const float* x [[buffer(4)]],device const uint* groups [[buffer(5)]],device const uint* tiles [[buffer(6)]],device const float* table [[buffer(7)]],constant Params& p [[buffer(15)]],uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float scratch[4096]; \
    prefill_native_expert_tile<COLS,FORMAT,DOWN>(y,w,scales,factors,x,groups,tiles,table,p,group,tid,lane,scratch); \
}
NATIVE_EXPERT(expert_prefill_nvfp4_up,2304,0,false)
NATIVE_EXPERT(expert_prefill_nvfp4_down,1152,0,true)
NATIVE_EXPERT(expert_prefill_mxfp8_up,2304,1,false)
NATIVE_EXPERT(expert_prefill_mxfp8_down,1152,1,true)
#undef NATIVE_EXPERT
