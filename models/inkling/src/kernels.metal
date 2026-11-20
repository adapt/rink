// Inkling 2 Mini text decoder. Float32 accumulation and activations.
// ABI: output at 0, inputs at 1..14, Params at 15. Each kernel states its dimensions.
#include <metal_stdlib>
using namespace metal;
struct Params { uint u[16]; float f[16]; };
float weight(device const uchar* w, uint i, uint dtype) {
    if (dtype == 0) return ((device const float*)w)[i];
    if (dtype == 1) return as_type<float>(uint(((device const ushort*)w)[i]) << 16);
    return float(((device const half*)w)[i]);
}
float fp8(uchar v) {
    uint e=(v>>3)&15, m=v&7;
    float a = e == 0 ? float(m)*0.001953125f : exp2(float(int(e)-7))*(1.f+float(m)*0.125f);
    if(e==15 && m==7) a=NAN;
    return v&128 ? -a : a;
}
kernel void copy_vec(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if(i<p.u[0]) y[p.u[1]+i]=x[p.u[2]+i];
}
// n, row, dtype
kernel void embedding(device float* y [[buffer(0)]], device const uchar* w [[buffer(1)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if(i<p.u[0]) y[i]=weight(w,p.u[1]*p.u[0]+i,p.u[2]);
}
// rows, cols, dtype; one SIMD group per row
kernel void matvec(device float* y [[buffer(0)]], device const uchar* w [[buffer(1)]], device const float* x [[buffer(2)]], constant Params& p [[buffer(15)]], uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=t/32; if(row>=p.u[0]) return;
    float v=0; for(uint c=lane;c<p.u[1];c+=32) v+=weight(w,row*p.u[1]+c,p.u[2])*x[c];
    v=simd_sum(v); if(lane==0) y[row]=v;
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
    uint h=t/32,n=p.u[0]; if(h>=p.u[1]) return;
    float s=0; for(uint i=lane;i<n;i+=32){ uint j=h*n+i; float a=x[j]*(p.u[2]?scale[j]:1.f); s+=a*a; }
    float r=min(1.f,p.f[0]/(sqrt(simd_sum(s))+1.e-6f))*p.f[1];
    for(uint i=lane;i<n;i+=32){ uint j=h*n+i; y[j]=x[j]*(p.u[2]?scale[j]:1.f)*r; }
}
kernel void add(device float* y [[buffer(0)]], device const float* a [[buffer(1)]], device const float* b [[buffer(2)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) { if(i<p.u[0]) y[i]=a[i]+b[i]; }
// width, blocks (<=32), query offset. One SIMD group, normalized values and keys.
kernel void block_read(device float* y [[buffer(0)]], device const float* blocks [[buffer(1)]], device const float* query [[buffer(2)]], constant Params& p [[buffer(15)]], uint lane [[thread_index_in_simdgroup]]) {
    float r[32],score[32],mx=-INFINITY,den=0; uint n=p.u[0], b=p.u[1];
    for(uint j=0;j<b;j++) {
        float ss=0,dot=0;
        for(uint i=lane;i<n;i+=32) { float v=blocks[j*n+i]; ss+=v*v; dot+=v*query[p.u[2]+i]; }
        r[j]=rsqrt(simd_sum(ss)/float(n)+1.e-6f);
        score[j]=simd_sum(dot)*r[j]*rsqrt(float(n)); mx=max(mx,score[j]);
    }
    for(uint j=0;j<b;j++){score[j]=exp(score[j]-mx); den+=score[j];}
    for(uint i=lane;i<n;i+=32){float v=0;for(uint j=0;j<b;j++) v+=blocks[j*n+i]*r[j]*score[j]/den; y[i]=v;}
}
// Concatenated unweighted RMS-normalized controller inputs. width; epsilon.
kernel void concat_norm(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], device const float* msg [[buffer(2)]], constant Params& p [[buffer(15)]], uint lane [[thread_index_in_simdgroup]]) {
    uint n=p.u[0];float a=0,b=0;
    for(uint i=lane;i<n;i+=32){a+=x[i]*x[i];b+=msg[i]*msg[i];}
    a=rsqrt(simd_sum(a)/float(n)+p.f[0]);b=rsqrt(simd_sum(b)/float(n)+p.f[0]);
    for(uint i=lane;i<n;i+=32){y[i]=x[i]*a;y[n+i]=msg[i]*b;}
}
kernel void controller(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], device const float* msg [[buffer(2)]], device const float* forget [[buffer(3)]], device const float* gain [[buffer(4)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if(i<p.u[0]) y[i]=(1.f-tanh(forget[i]))*x[i]+tanh(gain[i])*msg[i];
}
// channels, taps, absolute position, dynamic, head_dim. State is [tap,channel].
kernel void convolution(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], device const float* w [[buffer(2)]], device float* state [[buffer(3)]], device const float* dyn [[buffer(4)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    uint c=p.u[0],k=p.u[1],pos=p.u[2];if(i>=c)return;
    state[(pos%k)*c+i]=x[i]; float v=x[i];
    for(uint lag=0;lag<k && lag<=pos;lag++) {
        float tap=p.u[3] ? w[lag*c+i]+dyn[(i/p.u[4])*k+lag] : w[i*k+(k-1-lag)];
        v+=tap*state[((pos+k-lag)%k)*c+i];
    }
    y[i]=v*rsqrt(float(1+k*(p.u[3]?2:1)));
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
    if(t)return;float s[256],rank[256],den=0;
    for(uint i=0;i<p.u[0];i++){s[i]=1.f/(1.f+exp(-logits[i]));rank[i]=s[i]+bias[i];}
    for(uint j=0;j<p.u[1];j++){uint best=0;for(uint i=1;i<p.u[0];i++)if(rank[i]>rank[best])best=i; y[2*j]=float(best);y[2*j+1]=s[best];den+=s[best];rank[best]=-INFINITY;}
    for(uint j=0;j<p.u[1];j++)y[2*j+1]*=p.f[0]/(den+1.e-20f);
}
// rows, cols, top_k, format (0 NVFP4, 1 MXFP8, 2 F32), per-expert-input.
kernel void expert_matvec(device float* y [[buffer(0)]], device const uchar* w [[buffer(1)]], device const uchar* scales [[buffer(2)]], device const float* factors [[buffer(3)]], device const float* x [[buffer(4)]], device const float* routing [[buffer(5)]], constant Params& p [[buffer(15)]], uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint out=t/32,slot=out/p.u[0],row=out%p.u[0];if(slot>=p.u[2])return;
    uint expert=uint(routing[2*slot]),cols=p.u[1];float sum=0;
    const float lut[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
    for(uint c=lane;c<cols;c+=32){
        uint idx=(expert*p.u[0]+row)*cols+c;float v;
        if(p.u[3]==0){uint code=(w[idx/2]>>((idx%2)*4))&15;v=lut[code&7]*(code&8?-1.f:1.f)*fp8(scales[idx/16])*factors[expert];}
        else if(p.u[3]==1)v=fp8(w[idx])*exp2(float(int(scales[idx/32])-127));
        else v=((device const float*)w)[idx];
        sum+=v*x[(p.u[4]?slot*cols:0)+c];
    }
    sum=simd_sum(sum);if(lane==0)y[out]=sum;
}
// intermediate width, top_k; gate softcap, up softcap. Native w13 rows interleave gate/up.
kernel void swiglu(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if(i<p.u[0]*p.u[1]){float g=x[2*i],u=x[2*i+1];y[i]=p.f[0]*tanh(g/p.f[0])/(1.f+exp(-g))*p.f[1]*tanh(u/p.f[1]);}
}
kernel void expert_sum(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], device const float* routing [[buffer(2)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    if(i<p.u[0]){float v=0;for(uint j=0;j<p.u[1];j++)v+=x[j*p.u[0]+i]*routing[2*j+1];y[i]=v;}
}
kernel void softcap(device float* y [[buffer(0)]], device const float* x [[buffer(1)]], constant Params& p [[buffer(15)]], uint i [[thread_position_in_grid]]) { if(i<p.u[0]) y[i]=p.f[0]*tanh(x[i]/p.f[0]); }
