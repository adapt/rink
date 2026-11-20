// Inkling 2 Mini text decoder. Float32 accumulation and activations.
// ABI: output at 0, inputs at 1..14, Params at 15. Each kernel states its dimensions.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
struct Params { uint u[16]; float f[16]; };
float weight(device const uchar* w, uint i, uint dtype) {
    if (dtype == 0) return ((device const float*)w)[i];
    if (dtype == 1) return as_type<float>(uint(((device const ushort*)w)[i]) << 16);
    return float(((device const half*)w)[i]);
}
float fp8(uchar v) {
    uint e=(v>>3)&15, m=v&7;
    float a = e == 0 ? float(m)*0.001953125f : as_type<float>(((e+120)<<23)|(m<<20));
    if(e==15 && m==7) a=NAN;
    return v&128 ? -a : a;
}
float scale8(uchar v) { return as_type<float>(v ? uint(v)<<23 : 0x00400000u); }
float4 weight4(device const uchar* w, uint i, uint dtype) {
    if(dtype==0)return float4(*((device const packed_float4*)(w+4*i)));
    if(dtype==1)return as_type<float4>(uint4(*((device const packed_ushort4*)(w+2*i)))<<16);
    return float4(*((device const packed_half4*)(w+2*i)));
}
kernel void copy_vec(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if(i<p.u[0]) y[p.u[1]+i]=x[p.u[2]+i];
}
// n, row, dtype
kernel void embedding(device float* y [[buffer(0)]], device const uchar* w [[buffer(1)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if(i<p.u[0]) y[i]=weight(w,p.u[1]*p.u[0]+i,p.u[2]);
}
kernel void embedding_batch(device float* y [[buffer(0)]], device const uchar* w [[buffer(1)]], device const uint* ids [[buffer(2)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if(i<p.u[0]*p.u[1])y[i]=weight(w,ids[i/p.u[0]]*p.u[0]+i%p.u[0],p.u[2]);
}
// rows, cols, dtype; one SIMD group per row
kernel void matvec(device float* y [[buffer(0)]], device const uchar* w [[buffer(1)]], device const float* x [[buffer(2)]], constant Params& p [[buffer(15)]], uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=t/32; if(row>=p.u[0]) return;
    float v=0; for(uint c=lane;c<p.u[1];c+=32) v+=weight(w,row*p.u[1]+c,p.u[2])*x[p.u[4]+c];
    v=simd_sum(v); if(lane==0) y[row]=v;
}
// Four output rows per SIMD group, four contiguous weights per lane. cols % 4 == 0.
kernel void matvec4(device float* y [[buffer(0)]], device const uchar* w [[buffer(1)]], device const float* x [[buffer(2)]], constant Params& p [[buffer(15)]], uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=(t/32)*4,cols=p.u[1];if(row>=p.u[0])return;float4 sum=0;
    for(uint c=lane*4;c<cols;c+=128){float4 v=*((device const float4*)(x+p.u[4]+c));
        for(uint r=0;r<4 && row+r<p.u[0];r++)sum[r]+=dot(weight4(w,(row+r)*cols+c,p.u[2]),v);}
    for(uint r=0;r<4;r++){float v=simd_sum(sum[r]);if(lane==0 && row+r<p.u[0])y[row+r]=v;}
}
// X[batch,cols] * W[rows,cols]^T. Each SIMD group computes an 8x8 tile,
// staging 32 columns at a time and accumulating in float32 matrix registers.
kernel void matmul(device float* y [[buffer(0)]], device const uchar* w [[buffer(1)]], device const float* x [[buffer(2)]], constant Params& p [[buffer(15)]], uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float a[8*32],b[32*8],result[64];
    uint row_tiles=(p.u[0]+7)/8,row=(group%row_tiles)*8,token=(group/row_tiles)*8;
    simdgroup_float8x8 sum=make_filled_simdgroup_matrix<float,8,8>(0.f);
    for(uint base=0;base<p.u[1];base+=32){
        for(uint i=lane;i<256;i+=32){uint r=i/32,c=i%32;
            a[i]=token+r<p.u[3] && base+c<p.u[1]?x[p.u[4]+(token+r)*p.u[1]+base+c]:0.f;
            b[c*8+r]=row+r<p.u[0] && base+c<p.u[1]?weight(w,(row+r)*p.u[1]+base+c,p.u[2]):0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint k=0;k<32;k+=8){simdgroup_float8x8 av,bv;simdgroup_load(av,a+k,32);simdgroup_load(bv,b+k*8,8);simdgroup_multiply_accumulate(sum,av,bv,sum);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(sum,result,8);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=lane;i<64;i+=32){uint t=token+i/8,r=row+i%8;if(t<p.u[3] && r<p.u[0])y[t*p.u[0]+r]=result[i];}
}
// width, rows, weighted; epsilon
kernel void rms(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], device const float* w [[buffer(2)]], constant Params& p [[buffer(15)]], uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=t/32, n=p.u[0]; if(row>=p.u[1]) return;
    float s=0; for(uint i=lane;i<n;i+=32) s+=x[row*n+i]*x[row*n+i];
    float r=rsqrt(simd_sum(s)/float(n)+p.f[0]);
    for(uint i=lane;i<n;i+=32) y[row*n+i]=x[row*n+i]*r*(p.u[2]?w[i]:1.f);
}
// head_dim, heads, learned_scale; threshold, log-scale multiplier
kernel void clip(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], device const float* scale [[buffer(2)]], constant Params& p [[buffer(15)]], uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=t/32,n=p.u[0]; if(row>=p.u[1]*p.u[3]) return;uint h=row%p.u[1],token=row/p.u[1];
    float s=0; for(uint i=lane;i<n;i+=32){ uint j=row*n+i; float a=x[j]*(p.u[2]?scale[h*n+i]:1.f); s+=a*a; }
    float tau=1.f+p.f[1]*log(max(float(p.u[4]+token+1)/float(p.u[5]),1.f));
    float r=min(1.f,p.f[0]/(sqrt(simd_sum(s))+1.e-6f))*tau;
    for(uint i=lane;i<n;i+=32){ uint j=row*n+i; y[j]=x[j]*(p.u[2]?scale[h*n+i]:1.f)*r; }
}
kernel void add(device float* y [[buffer(0)]], device const float* a [[buffer(1)]], device const float* b [[buffer(2)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) { if(i<p.u[0]) y[i]=a[i]+b[i]; }
// width, blocks (<=32), query offset. One SIMD group, normalized values and keys.
kernel void block_read(device float* y [[buffer(0)]], device const float* blocks [[buffer(1)]], device const float* live [[buffer(2)]], device const float* query [[buffer(3)]], constant Params& p [[buffer(15)]], uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    float r[32],score[32],mx=-INFINITY,den=0; uint n=p.u[0], b=p.u[1];
    uint token=t/32;if(token>=p.u[4])return;y+=token*n;live+=token*n;blocks+=token*n;
    for(uint j=0;j<b;j++) {
        float ss=0,dot=0;
        device const float* block = j+1==b ? live : blocks+(j+p.u[3])*n*p.u[4];
        for(uint i=lane;i<n;i+=32) { float v=block[i]; ss+=v*v; dot+=v*query[p.u[2]+i]; }
        r[j]=rsqrt(simd_sum(ss)/float(n)+1.e-6f);
        score[j]=simd_sum(dot)*r[j]*rsqrt(float(n)); mx=max(mx,score[j]);
    }
    for(uint j=0;j<b;j++){score[j]=exp(score[j]-mx); den+=score[j];}
    for(uint i=lane;i<n;i+=32){float v=0;for(uint j=0;j<b;j++){device const float* block=j+1==b?live:blocks+(j+p.u[3])*n*p.u[4];v+=block[i]*r[j]*score[j]/den;}y[i]=v;}
}
// Same ABI and arithmetic as block_read. Eight SIMD groups divide the block
// reductions; all 256 threads share the weighted output across hidden channels.
kernel void block_read_parallel(device float* y [[buffer(0)]], device const float* blocks [[buffer(1)]], device const float* live [[buffer(2)]], device const float* query [[buffer(3)]], constant Params& p [[buffer(15)]], uint token [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float r[32],score[32],den;
    uint n=p.u[0],b=p.u[1],sg=tid/32;
    if(token>=p.u[4])return;y+=token*n;live+=token*n;blocks+=token*n;
    for(uint j=sg;j<b;j+=8){
        device const float* block=j+1==b?live:blocks+(j+p.u[3])*n*p.u[4];
        float ss=0,dot=0;for(uint i=lane;i<n;i+=32){float v=block[i];ss+=v*v;dot+=v*query[p.u[2]+i];}
        float norm=rsqrt(simd_sum(ss)/float(n)+1.e-6f),s=simd_sum(dot)*norm*rsqrt(float(n));
        if(lane==0){r[j]=norm;score[j]=s;}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0){float mx=-INFINITY,total=0;for(uint j=0;j<b;j++)mx=max(mx,score[j]);for(uint j=0;j<b;j++){score[j]=exp(score[j]-mx);total+=score[j];}den=total;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<n;i+=256){float v=0;for(uint j=0;j<b;j++){device const float* block=j+1==b?live:blocks+(j+p.u[3])*n*p.u[4];v+=block[i]*r[j]*score[j]/den;}y[i]=v;}
}
// Concatenated unweighted RMS-normalized controller inputs. width; epsilon.
kernel void concat_norm(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], device const float* msg [[buffer(2)]], constant Params& p [[buffer(15)]], uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint n=p.u[0];float a=0,b=0;
    uint token=t/32;if(token>=p.u[1])return;x+=token*n;msg+=token*n;y+=token*n*2;
    for(uint i=lane;i<n;i+=32){a+=x[i]*x[i];b+=msg[i]*msg[i];}
    a=rsqrt(simd_sum(a)/float(n)+p.f[0]);b=rsqrt(simd_sum(b)/float(n)+p.f[0]);
    for(uint i=lane;i<n;i+=32){y[i]=x[i]*a;y[n+i]=msg[i]*b;}
}
kernel void controller(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], device const float* msg [[buffer(2)]], device const float* forget [[buffer(3)]], device const float* gain [[buffer(4)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if(i<p.u[0]) y[i]=(1.f-tanh(forget[i]))*x[i]+tanh(gain[i])*msg[i];
}
// channels, taps, absolute position, dynamic, head_dim. State is [tap,channel].
kernel void convolution(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], device const float* w [[buffer(2)]], device float* state [[buffer(3)]], device const float* dyn [[buffer(4)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    uint c=p.u[0],k=p.u[1];if(i>=c)return;
    for(uint token=0;token<p.u[5];token++){uint pos=p.u[2]+token;
        state[(pos%k)*c+i]=x[token*c+i]; float v=x[token*c+i];
        for(uint lag=0;lag<k && lag<=pos;lag++) {
            float tap=p.u[3] ? w[lag*c+i]+dyn[(token*(c/p.u[4])+i/p.u[4])*k+lag] : w[i*k+(k-1-lag)];
            v+=tap*state[((pos+k-lag)%k)*c+i];
        }
        y[token*c+i]=v*rsqrt(float(1+k*(p.u[3]?2:1)));
    }
}
// Commit only the last cache-capacity entries after attention reads the old ring.
kernel void cache_write(device float* keys [[buffer(0)]], device float* values [[buffer(1)]], device const float* k [[buffer(2)]], device const float* v [[buffer(3)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    uint token=i/p.u[0],c=i%p.u[0];if(token>=p.u[3] || token+p.u[1]<p.u[3])return;
    uint dest=((p.u[2]+token)%p.u[1])*p.u[0]+c;keys[dest]=k[i];values[dest]=v[i];
}
// q_heads, kv_heads, head_dim, position, cache_capacity, visible_len, d_rel, rel_extent.
// One SIMD group per (head,key); KV cache is token-major with ring addressing.
kernel void attn_scores(device float* y [[buffer(0)]], device const float* q [[buffer(1)]], device const float* keys [[buffer(2)]], device const float* rel [[buffer(3)]], device const float* profiles [[buffer(4)]], constant Params& p [[buffer(15)]], uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=t/32,n=p.u[5],h=row/n,j=row%n;if(h>=p.u[0])return;
    uint kp=p.u[3]+1-n+j, kh=h/(p.u[0]/p.u[1]),d=p.u[2],dist=p.u[3]-kp;
    float v=0;for(uint i=lane;i<d;i+=32)v+=q[h*d+i]*keys[((kp%p.u[4])*p.u[1]+kh)*d+i];
    v=simd_sum(v)*rsqrt(float(d));
    float bias=0;if(dist<p.u[7])for(uint i=lane;i<p.u[6];i+=32)bias+=rel[h*p.u[6]+i]*profiles[i*p.u[7]+dist];
    v+=simd_sum(bias);if(lane==0)y[row]=v;
}
// Same geometry as attn_scores. Sink adds probability mass with zero value.
kernel void attn_values(device float* y [[buffer(0)]], device const float* scores [[buffer(1)]], device const float* values [[buffer(2)]], device const float* sinks [[buffer(3)]], constant Params& p [[buffer(15)]], uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint h=t/32,n=p.u[5],d=p.u[2];if(h>=p.u[0])return;
    float mx=sinks[h];for(uint j=lane;j<n;j+=32)mx=max(mx,scores[h*n+j]);mx=simd_max(mx);
    float den=0;for(uint j=lane;j<n;j+=32)den+=exp(scores[h*n+j]-mx);den=simd_sum(den)+exp(sinks[h]-mx);
    uint kh=h/(p.u[0]/p.u[1]);
    for(uint i=lane;i<d;i+=32){float v=0;for(uint j=0;j<n;j++){uint kp=p.u[3]+1-n+j;v+=exp(scores[h*n+j]-mx)/den*values[((kp%p.u[4])*p.u[1]+kh)*d+i];}y[h*d+i]=v;}
}
kernel void gate(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], device const float* g [[buffer(2)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) { if(i<p.u[0])y[i]=x[i]/(1.f+exp(-g[i])); }
// experts<=256, top_k<=32; route_scale. One thread. Output [expert, weight] pairs.
kernel void route(device float* y [[buffer(0)]], device const float* logits [[buffer(1)]], device const float* bias [[buffer(2)]], constant Params& p [[buffer(15)]], uint t [[thread_position_in_grid]]) {
    if(t>=p.u[2])return;logits+=t*p.u[0];y+=t*p.u[1]*2;float s[256],rank[256],den=0;
    for(uint i=0;i<p.u[0];i++){s[i]=1.f/(1.f+exp(-logits[i]));rank[i]=s[i]+bias[i];}
    for(uint j=0;j<p.u[1];j++){uint best=0;for(uint i=1;i<p.u[0];i++)if(rank[i]>rank[best])best=i; y[2*j]=float(best);y[2*j+1]=s[best];den+=s[best];rank[best]=-INFINITY;}
    for(uint j=0;j<p.u[1];j++)y[2*j+1]*=p.f[0]/(den+1.e-20f);
}
// rows, cols, top_k, format (0 NVFP4, 1 MXFP8, 2 F32), per-expert-input.
kernel void expert_matvec(device float* y [[buffer(0)]], device const uchar* w [[buffer(1)]], device const uchar* scales [[buffer(2)]], device const float* factors [[buffer(3)]], device const float* x [[buffer(4)]], device const float* routing [[buffer(5)]], constant Params& p [[buffer(15)]], uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint out=t/32,slot=out/p.u[0],row=out%p.u[0],token=slot/p.u[2];if(token>=p.u[5])return;
    uint expert=uint(routing[2*slot]),cols=p.u[1];float sum=0;x+=(p.u[4]?slot:token)*cols;
    const float lut[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
    if(p.u[3]<2){
        float factor=p.u[3]==0?factors[expert]:1.f;
        for(uint c=lane*4;c<cols;c+=128){uint idx=(expert*p.u[0]+row)*cols+c;float4 v;
            if(p.u[3]==0){uint codes=*((device const ushort*)(w+idx/2));float scale=fp8(scales[idx/16]);
                for(uint j=0;j<4;j++){uint code=(codes>>(j*4))&15;v[j]=lut[code&7]*(code&8?-1.f:1.f)*scale*factor;}}
            else{uchar4 codes=*((device const uchar4*)(w+idx));float scale=scale8(scales[idx/32]);for(uint j=0;j<4;j++)v[j]=fp8(codes[j])*scale;}
            float4 input=*((device const float4*)(x+c));sum+=dot(v,input);
        }
    }else{for(uint c=lane;c<cols;c+=32){uint idx=(expert*p.u[0]+row)*cols+c;sum+=((device const float*)w)[idx]*x[c];}}
    sum=simd_sum(sum);if(lane==0)y[out]=sum;
}
// intermediate width, top_k; gate softcap, up softcap. Native w13 rows interleave gate/up.
kernel void swiglu(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if(i<p.u[0]*p.u[1]*p.u[2]){float g=x[2*i],u=x[2*i+1];y[i]=p.f[0]*tanh(g/p.f[0])/(1.f+exp(-g))*p.f[1]*tanh(u/p.f[1]);}
}
kernel void expert_sum(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], device const float* routing [[buffer(2)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if(i<p.u[0]*p.u[2]){uint token=i/p.u[0],c=i%p.u[0];float v=0;for(uint j=0;j<p.u[1];j++)v+=x[(token*p.u[1]+j)*p.u[0]+c]*routing[2*(token*p.u[1]+j)+1];y[i]=v;}
}
kernel void softcap(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) { if(i<p.u[0]) y[i]=p.f[0]*tanh(x[i]/p.f[0]); }

struct Choice { float value; uint index; uint invalid; };
// 128 keys per partition, four SIMD groups per threadgroup. Compute softmax
// weights once per key and retain only a partial weighted value plus max/sum.
// Geometry: q_heads,kv_heads,dim,base_position,capacity,visible,d_rel,extent,parts,batch.
kernel void attention_partition(device float* out [[buffer(0)]], device const float* q [[buffer(1)]], device const float* keys [[buffer(2)]], device const float* values [[buffer(3)]], device const float* new_keys [[buffer(4)]], device const float* new_values [[buffer(5)]], device const float* rel [[buffer(6)]], device const float* profiles [[buffer(7)]], constant Params& p [[buffer(15)]], uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[128], maxima[4], totals[4];
    uint parts=p.u[8], query=group/parts, part=group%parts, h=query%p.u[0], token=query/p.u[0], d=p.u[2];
    uint pos=p.u[3]+token,n=min(pos+1,p.u[4]),start=part*128,kh=h/(p.u[0]/p.u[1]),sg=tid/32;
    uint dest=(query*parts+part)*(d+2);
    if(start>=n){if(tid==0){out[dest]=-INFINITY;out[dest+1]=0;}for(uint i=tid;i<d;i+=128)out[dest+2+i]=0;return;}
    for(uint j=sg;j<128;j+=4){uint at=start+j;float score=-INFINITY;
        if(at<n){uint kp=pos+1-n+at,dist=pos-kp;
            device const float* key=kp>=p.u[3]?new_keys+((kp-p.u[3])*p.u[1]+kh)*d:keys+((kp%p.u[4])*p.u[1]+kh)*d;
            float dot=0;for(uint i=lane;i<d;i+=32)dot+=q[query*d+i]*key[i];score=simd_sum(dot)*rsqrt(float(d));
            float bias=0;if(dist<p.u[7])for(uint i=lane;i<p.u[6];i+=32)bias+=rel[query*p.u[6]+i]*profiles[i*p.u[7]+dist];score+=simd_sum(bias);
        }
        if(lane==0)scores[j]=score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx=simd_max(scores[tid]);if(lane==0)maxima[sg]=mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx=max(max(maxima[0],maxima[1]),max(maxima[2],maxima[3]));
    float weight=exp(scores[tid]-mx);scores[tid]=weight;float den=simd_sum(weight);if(lane==0)totals[sg]=den;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0){out[dest]=mx;out[dest+1]=totals[0]+totals[1]+totals[2]+totals[3];}
    for(uint i=tid;i<d;i+=128){float sum=0;for(uint j=0;j<min(128u,n-start);j++){uint kp=pos+1-n+start+j;
        device const float* value=kp>=p.u[3]?new_values+((kp-p.u[3])*p.u[1]+kh)*d:values+((kp%p.u[4])*p.u[1]+kh)*d;
        sum+=scores[j]*value[i];}out[dest+2+i]=sum;}
}
kernel void attention_finish(device float* y [[buffer(0)]], device const float* parts [[buffer(1)]], device const float* sinks [[buffer(2)]], device const float* gate [[buffer(3)]], constant Params& p [[buffer(15)]], uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint query=t/32,d=p.u[2],n=p.u[8];if(query>=p.u[0]*p.u[9])return;
    uint base=query*n*(d+2);float mx=sinks[query];for(uint j=lane;j<n;j+=32)mx=max(mx,parts[base+j*(d+2)]);mx=simd_max(mx);
    float den=0;for(uint j=lane;j<n;j+=32)den+=parts[base+j*(d+2)+1]*exp(parts[base+j*(d+2)]-mx);den=simd_sum(den)+exp(sinks[query]-mx);
    for(uint i=lane;i<d;i+=32){float sum=0;for(uint j=0;j<n;j++)sum+=parts[base+j*(d+2)+2+i]*exp(parts[base+j*(d+2)]-mx);
        y[query*d+i]=(sum/den)/(1.f+exp(-gate[query*d+i]));}
}
kernel void argmax_partial(device Choice* y [[buffer(0)]], device const float* x [[buffer(1)]], constant Params& p [[buffer(15)]], uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint group=t/32, best=0xffffffff; float value=-INFINITY; bool bad=false;
    for(uint i=group*1024+lane;i<min((group+1)*1024,p.u[0]);i+=32){float v=x[i];bad|=!isfinite(v);if(v>value || (v==value && i<best)){value=v;best=i;}}
    float maximum=simd_max(value);uint index=simd_min(value==maximum?best:0xffffffff);bool invalid=simd_any(bad);
    if(lane==0)y[group]={maximum,index,uint(invalid)};
}
kernel void argmax_finish(device uint* y [[buffer(0)]], device const Choice* x [[buffer(1)]], constant Params& p [[buffer(15)]], uint lane [[thread_index_in_simdgroup]]) {
    uint best=0xffffffff;float value=-INFINITY;bool bad=false;
    for(uint i=lane;i<p.u[0];i+=32){Choice v=x[i];bad|=v.invalid!=0;if(v.value>value || (v.value==value && v.index<best)){value=v.value;best=v.index;}}
    float maximum=simd_max(value);uint index=simd_min(value==maximum?best:0xffffffff);bool invalid=simd_any(bad);
    if(lane==0){y[0]=index;y[1]=uint(invalid);}
}
