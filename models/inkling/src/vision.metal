// Still-image encoder. Uses the shared Params and weight helpers in kernels.metal.
// width, native patch rows/columns, learned grid side, position weight dtype.
kernel void vision_positions(device float* y [[buffer(0)]], device const float* patches [[buffer(1)]], device const uchar* positions [[buffer(2)]], device const float* cls [[buffer(3)]], device const float* cls_pos [[buffer(4)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    uint h=p.u[0],token=i/h,c=i%h,count=p.u[1]*p.u[2];if(token>count)return;
    if(token==0){y[i]=cls[c]+cls_pos[c];return;}
    uint patch=token-1,block=patch/4,row=(block/(p.u[2]/2))*2+(patch%4)/2,col=(block%(p.u[2]/2))*2+patch%2;
    float fy=(float(row)+0.5f)*float(p.u[3])/float(p.u[1])-0.5f;
    float fx=(float(col)+0.5f)*float(p.u[3])/float(p.u[2])-0.5f;
    int iy=int(floor(fy)),ix=int(floor(fx));float dy=fy-float(iy),dx=fx-float(ix),v=0;
    for(uint a=0;a<2;a++)for(uint b=0;b<2;b++){
        uint r=uint(clamp(iy+int(a),0,int(p.u[3])-1)),s=uint(clamp(ix+int(b),0,int(p.u[3])-1));
        v+=weight(positions,(r*p.u[3]+s)*h+c,p.u[4])*((a?dy:1.f-dy)*(b?dx:1.f-dx));
    }
    y[i]=patches[patch*h+c]+v;
}
// width, rows; epsilon. Mean subtraction is required: this is LayerNorm, not RMSNorm.
kernel void vision_norm(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], device const float* w [[buffer(2)]], device const float* bias [[buffer(3)]], constant Params& p [[buffer(15)]], uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=t/32,n=p.u[0];if(row>=p.u[1])return;x+=row*n;y+=row*n;
    float mean=0;for(uint c=lane;c<n;c+=32)mean+=x[c];mean=simd_sum(mean)/float(n);
    float var=0;for(uint c=lane;c<n;c+=32){float v=x[c]-mean;var+=v*v;}
    float inv=rsqrt(simd_sum(var)/float(n)+p.f[0]);
    for(uint c=lane;c<n;c+=32)y[c]=((x[c]-mean)*inv)*w[c]+bias[c];
}
// width, rows, GELU, bias-present. The bias is F32 and shared by rows.
// MSL has no erf. Evaluate its convergent, positive-term series:
// erf(x) = 2*x*exp(-x*x)/sqrt(pi) * sum_n (2*x*x)^n/(2n+1)!!.
// At |x| >= 4 the result rounds to +/-1 in F32. Below 4, 64 terms
// suffice; compensated summation limits rounding error without a tanh GELU.
float vision_erf(float x) {
    float a=abs(x);if(a>=4.f)return copysign(1.f,x);
    float z=2.f*a*a,term=1.f,sum=1.f,correction=0.f;
    for(uint n=1;n<64;n++){
        term*=z/float(2*n+1);
        float v=term-correction,next=sum+v;correction=(next-sum)-v;sum=next;
    }
    return copysign(min(1.f,1.1283791670955126f*a*exp(-a*a)*sum),x);
}
kernel void vision_bias(device float* y [[buffer(0)]], device const float* bias [[buffer(1)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if(i>=p.u[0]*p.u[1])return;float v=y[i]+(p.u[3]?bias[i%p.u[0]]:0.f);
    if(p.u[2])v=0.5f*v*(1.f+vision_erf(v*0.7071067811865475f));y[i]=v;
}
// width, rows. Projection bias precedes the layer scale and residual addition.
kernel void vision_residual(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], device const float* projected [[buffer(2)]], device const float* bias [[buffer(3)]], device const float* scale [[buffer(4)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if(i<p.u[0]*p.u[1])y[i]=x[i]+(projected[i]+bias[i%p.u[0]])*scale[i%p.u[0]];
}
// sequence, heads, head width, native patch columns; axial RoPE theta.
// Each thread owns a complete interleaved pair, so in-place rotation is race free.
kernel void vision_rope(device float* q [[buffer(0)]], device float* k [[buffer(1)]], device const float* qb [[buffer(2)]], device const float* kb [[buffer(3)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    uint d=p.u[2],h=p.u[1]*d,token=(2*i)/h,pair=i%(d/2),c=(2*i)%h;if(token>=p.u[0])return;
    uint patch=token?token-1:0,block=patch/4;
    uint row=token?(block/(p.u[3]/2))*2+(patch%4)/2+1:0;
    uint col=token?(block%(p.u[3]/2))*2+patch%2+1:0;
    float angle=float(pair<d/4?col:row)*pow(p.f[0],-float(pair%(d/4))*4.f/float(d));
    float co=cos(angle),si=sin(angle),q0=q[2*i]+qb[c],q1=q[2*i+1]+qb[c+1],k0=k[2*i]+kb[c],k1=k[2*i+1]+kb[c+1];
    q[2*i]=q0*co-q1*si;q[2*i+1]=q1*co+q0*si;
    k[2*i]=k0*co-k1*si;k[2*i+1]=k1*co+k0*si;
}
// sequence, heads, head width (multiple of 8, <=128). Noncausal attention for
// one image, including its class token. Eight queries share each 32-key tile.
// Online softmax bounds workspace independently of image size; all math is F32.
kernel void vision_attention(device float* y [[buffer(0)]], device const float* q [[buffer(1)]], device const float* k [[buffer(2)]], device const float* v [[buffer(3)]], constant Params& p [[buffer(15)]], uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float queries[1024],stage[4096],scores[256],tile[1024],out[1024],mx[8],den[8],scale[8];
    uint n=p.u[0],heads=p.u[1],d=p.u[2],head=group%heads,first=(group/heads)*8,stride=heads*d;if(first>=n)return;
    for(uint i=lane;i<8*d;i+=32){uint row=i/d,c=i%d;queries[row*128+c]=first+row<n?q[(first+row)*stride+head*d+c]:0.f;out[i]=0.f;}
    if(lane<8){mx[lane]=-INFINITY;den[lane]=0;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint base=0;base<n;base+=32){
        for(uint i=lane;i<d*32;i+=32){uint c=i/32,t=i%32;stage[i]=base+t<n?k[(base+t)*stride+head*d+c]:0.f;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint part=0;part<4;part++){
            simdgroup_float8x8 sum=make_filled_simdgroup_matrix<float,8,8>(0.f);
            for(uint c=0;c<d;c+=8){simdgroup_float8x8 a,b;simdgroup_load(a,queries+c,128);simdgroup_load(b,stage+c*32+part*8,32);simdgroup_multiply_accumulate(sum,a,b,sum);}
            simdgroup_store(sum,scores+part*8,32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint row=0;row<8;row++){
            float logit=base+lane<n?scores[row*32+lane]*rsqrt(float(d)):-INFINITY;
            float peak=max(mx[row],simd_max(logit)),prob=exp(logit-peak),old_scale=exp(mx[row]-peak);
            float total=den[row]*old_scale+simd_sum(prob);
            scores[row*32+lane]=prob;
            if(lane==0){mx[row]=peak;den[row]=total;scale[row]=old_scale;}
        }
        for(uint i=lane;i<32*d;i+=32){uint t=i/d,c=i%d;stage[t*128+c]=base+t<n?v[(base+t)*stride+head*d+c]:0.f;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint c=0;c<d;c+=8){
            simdgroup_float8x8 sum=make_filled_simdgroup_matrix<float,8,8>(0.f);
            for(uint t=0;t<32;t+=8){simdgroup_float8x8 a,b;simdgroup_load(a,scores+t,32);simdgroup_load(b,stage+t*128+c,128);simdgroup_multiply_accumulate(sum,a,b,sum);}
            simdgroup_store(sum,tile+c,128);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=lane;i<8*d;i+=32)out[i]=out[i]*scale[i/d]+tile[(i/d)*128+i%d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=lane;i<8*d;i+=32){uint row=i/d;if(first+row<n)y[(first+row)*stride+head*d+i%d]=out[i]/den[row];}
}

// Native 64-channel heads: eight SIMD groups share 16 queries and 32 keys.
// Two independent 8-query matrix tiles reuse each K/V load. F32 matrix
// operations, key tiles and softmax reductions retain the reference order.
// Per-thread output accumulators and compact strides bound threadgroup memory
// to 18.2 KiB, with no extra device buffers.
kernel void vision_attention_64x16(device float* y [[buffer(0)]], device const float* q [[buffer(1)]], device const float* k [[buffer(2)]], device const float* v [[buffer(3)]], constant Params& p [[buffer(15)]], uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float queries[1024],stage[2048],scores[512],tile[1024],mx[16],den[16],scale[16];
    uint n=p.u[0],heads=p.u[1],head=group%heads,first=(group/heads)*16,stride=heads*64,sg=tid/32,qr=(sg/4)*8,part=sg%4;if(first>=n)return;
    float accum[4]={0.f,0.f,0.f,0.f};
    for(uint i=tid;i<1024;i+=256){uint row=i/64,c=i%64;queries[i]=first+row<n?q[(first+row)*stride+head*64+c]:0.f;}
    if(tid<16){mx[tid]=-INFINITY;den[tid]=0;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint base=0;base<n;base+=32){
        for(uint i=tid;i<2048;i+=256){uint c=i/32,t=i%32;stage[i]=base+t<n?k[(base+t)*stride+head*64+c]:0.f;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 sum=make_filled_simdgroup_matrix<float,8,8>(0.f);
        for(uint c=0;c<64;c+=8){simdgroup_float8x8 a,b;simdgroup_load(a,queries+qr*64+c,64);simdgroup_load(b,stage+c*32+part*8,32);simdgroup_multiply_accumulate(sum,a,b,sum);}
        simdgroup_store(sum,scores+qr*32+part*8,32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint row=sg;row<16;row+=8){
            float logit=base+lane<n?scores[row*32+lane]*rsqrt(64.f):-INFINITY;
            float peak=max(mx[row],simd_max(logit)),prob=exp(logit-peak),old_scale=exp(mx[row]-peak);
            float total=den[row]*old_scale+simd_sum(prob);
            scores[row*32+lane]=prob;
            if(lane==0){mx[row]=peak;den[row]=total;scale[row]=old_scale;}
        }
        for(uint i=tid;i<2048;i+=256){uint t=i/64,c=i%64;stage[i]=base+t<n?v[(base+t)*stride+head*64+c]:0.f;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint c=part*8;c<64;c+=32){
            simdgroup_float8x8 product=make_filled_simdgroup_matrix<float,8,8>(0.f);
            for(uint t=0;t<32;t+=8){simdgroup_float8x8 a,b;simdgroup_load(a,scores+qr*32+t,32);simdgroup_load(b,stage+t*64+c,64);simdgroup_multiply_accumulate(product,a,b,product);}
            simdgroup_store(product,tile+qr*64+c,64);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<4;j++){uint i=tid+j*256;accum[j]=accum[j]*scale[i/64]+tile[i];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint j=0;j<4;j++){uint i=tid+j*256,row=i/64;if(first+row<n)y[(first+row)*stride+head*64+i%64]=accum[j]/den[row];}
}
