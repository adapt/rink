// Native 64-channel image attention. Reuse K/V and score staging for the output
// tile after all matrix readers finish. Complete query tiles load directly into
// matrix fragments, avoiding the register pressure of keeping every Q fragment
// live. The final partial tile pads Q in scratch before loading its fragments.
// FP32 dot products, 32-key softmax updates and P*V accumulation are unchanged.
template<uint Q, bool Direct>
void vision_attention_shared(device float* y, device const float* q, device const float* k, device const float* v, constant Params& p, uint group, uint tid, uint lane, threadgroup float* scratch) {
    uint n=p.u[0],heads=p.u[1],head=group%heads,first=(group/heads)*Q,stride=heads*64,sg=tid/32,part=sg%4;
    threadgroup float* stage=scratch;
    threadgroup float* scores=scratch+2048;
    constexpr uint Storage=(Q*64>2048+Q*32?Q*64:2048+Q*32);
    threadgroup float* mx=scratch+Storage;
    threadgroup float* den=mx+Q;
    threadgroup float* scale=den+Q;
    simdgroup_float8x8 queries[Q/16][8];
    if(!Direct){
        for(uint i=tid;i<Q*64;i+=256){uint row=i/64,c=i%64;scratch[i]=first+row<n?q[(first+row)*stride+head*64+c]:0.f;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint r=0;r<Q/16;r++)for(uint c=0;c<8;c++)simdgroup_load(queries[r][c],scratch+((sg/4)*8+r*16)*64+c*8,64);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid<Q){mx[tid]=-INFINITY;den[tid]=0;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float accum[Q/4]={0.f};
    for(uint base=0;base<n;base+=32){
        for(uint j=0;j<8;j++){uint i=tid+j*256,c=i/32,t=i%32;stage[i]=base+t<n?k[(base+t)*stride+head*64+c]:0.f;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint r=0;r<Q/16;r++){
            simdgroup_float8x8 sum=make_filled_simdgroup_matrix<float,8,8>(0.f);
            for(uint c=0;c<8;c++){simdgroup_float8x8 b;simdgroup_load(b,stage+c*8*32+part*8,32);if(Direct){simdgroup_float8x8 a;simdgroup_load(a,q+(first+(sg/4)*8+r*16)*stride+head*64+c*8,stride);simdgroup_multiply_accumulate(sum,a,b,sum);}else simdgroup_multiply_accumulate(sum,queries[r][c],b,sum);}
            simdgroup_store(sum,scores+((sg/4)*8+r*16)*32+part*8,32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint row=sg;row<Q;row+=8){
            float logit=base+lane<n?scores[row*32+lane]*rsqrt(64.f):-INFINITY;
            float peak=max(mx[row],simd_max(logit)),prob=exp(logit-peak),old_scale=exp(mx[row]-peak);
            float total=den[row]*old_scale+simd_sum(prob);
            scores[row*32+lane]=prob;
            if(lane==0){mx[row]=peak;den[row]=total;scale[row]=old_scale;}
        }
        for(uint i=tid;i<2048;i+=256){uint t=i/64,c=i%64;stage[i]=base+t<n?v[(base+t)*stride+head*64+c]:0.f;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 products[Q/16][2];
        for(uint r=0;r<Q/16;r++)for(uint c=0;c<2;c++){
            products[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.f);
            for(uint t=0;t<32;t+=8){simdgroup_float8x8 a,b;simdgroup_load(a,scores+((sg/4)*8+r*16)*32+t,32);simdgroup_load(b,stage+t*64+part*8+c*32,64);simdgroup_multiply_accumulate(products[r][c],a,b,products[r][c]);}
        }
        // All readers must finish before K/V and score storage becomes output.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint r=0;r<Q/16;r++)for(uint c=0;c<2;c++)simdgroup_store(products[r][c],scratch+((sg/4)*8+r*16)*64+part*8+c*32,64);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<Q/4;j++){uint i=tid+j*256;accum[j]=accum[j]*scale[i/64]+scratch[i];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint j=0;j<Q/4;j++){uint i=tid+j*256,row=i/64;if(first+row<n)y[(first+row)*stride+head*64+i%64]=accum[j]/den[row];}
}

// 64 queries share every KV tile. 16.75 KiB shared storage, no device workspace.
kernel void vision_attention_64x64(device float* y [[buffer(0)]],device const float* q [[buffer(1)]],device const float* k [[buffer(2)]],device const float* v [[buffer(3)]],constant Params& p [[buffer(15)]],uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[4096+3*64];
    if((group/p.u[1])*64+64<=p.u[0])vision_attention_shared<64,true>(y,q,k,v,p,group,tid,lane,scratch);
    else vision_attention_shared<64,false>(y,q,k,v,p,group,tid,lane,scratch);
}
