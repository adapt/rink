// Bonsai's ABI uses FP16 activations and affine parameters, FP32 reductions and
// recurrent state. Packed matrices remain 2-bit throughout inference.
#include <metal_stdlib>
using namespace metal;
struct Params { uint u[16]; float f[16]; };

kernel void embedding(device const uint *w [[buffer(0)]], device const half *s [[buffer(1)]],
    device const half *b [[buffer(2)]], device half *out [[buffer(3)]],
    constant Params &p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    uint width=p.u[0], row=p.u[1];
    if(i<width) { uint j=row*width+i; out[i]=half(float((w[j/16]>>(2*(j%16)))&3)*float(s[j/128])+float(b[j/128])); }
}

// Normalized Sylvester transform: forward H(x*sign), inverse H(x)*sign.
// One threadgroup owns a block; no intergroup dependencies or in-place aliasing.
kernel void hadamard(device const half *x [[buffer(0)]], device const half *sign [[buffer(1)]],
    device half *out [[buffer(2)]], constant Params &p [[buffer(15)]],
    uint tid [[thread_index_in_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
    threadgroup float v[4096];
    uint n=p.u[0], base=block*n; bool inverse=p.u[1];
    for(uint i=tid;i<n;i+=256) v[i]=float(x[base+i])*(inverse?1.0f:float(sign[base+i]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=1;stride<n;stride*=2) {
        for(uint j=tid;j<n/2;j+=256) {
            uint a=(j/stride)*(2*stride)+j%stride, b=a+stride;
            float left=v[a],right=v[b]; v[a]=left+right;v[b]=left-right;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale=rsqrt(float(n));
    for(uint i=tid;i<n;i+=256) out[base+i]=half(v[i]*scale*(inverse?float(sign[base+i]):1.0f));
}

kernel void packed_mv(device const uint *w [[buffer(0)]], device const half *s [[buffer(1)]],
    device const half *b [[buffer(2)]], device const half *x [[buffer(3)]], device half *out [[buffer(4)]],
    constant Params &p [[buffer(15)]], uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=gid/32, width=p.u[0]; if(row>=p.u[1])return;
    float sum=0.0f;
    for(uint col=lane;col<width;col+=32) {
        uint j=row*width+col, code=(w[j/16]>>(2*(j%16)))&3;
        float weight=float(half(float(code)*float(s[j/128])+float(b[j/128])));
        sum+=weight*float(x[col]);
    }
    sum=simd_sum(sum);if(lane==0)out[row]=half(sum);
}

kernel void dense_mv(device const half *w [[buffer(0)]], device const half *x [[buffer(1)]],
    device half *out [[buffer(2)]], constant Params &p [[buffer(15)]], uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint row=gid/32,width=p.u[0];if(row>=p.u[1])return;
    float sum=0;for(uint i=lane;i<width;i+=32)sum+=float(w[row*width+i])*float(x[i]);
    sum=simd_sum(sum);if(lane==0)out[row]=half(sum);
}

// p: width, heads, input head stride, use weight; f: epsilon, post scale.
kernel void rms(device const half *x [[buffer(0)]], device const half *w [[buffer(1)]],
    device half *out [[buffer(2)]], constant Params &p [[buffer(15)]],
    uint tid [[thread_index_in_threadgroup]], uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];uint n=p.u[0],base=head*p.u[2]+p.u[4];
    float v=0;for(uint i=tid;i<n;i+=256)v+=float(x[base+i])*float(x[base+i]);
    v=simd_sum(v);if(lane==0)sums[simd]=v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=0;for(uint j=0;j<8;j++)total+=sums[j];
    float inv=rsqrt(total/float(n)+p.f[0]);
    for(uint i=tid;i<n;i+=256){half normalized=half(float(x[base+i])*inv*(p.u[3]?float(w[i]):1.0f));
        out[head*n+i]=half(float(normalized)*p.f[1]);}
}

kernel void rope(device half *x [[buffer(0)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint d=p.u[0],rot=p.u[1],halfrot=rot/2;if(i>=p.u[2]*halfrot)return;
    uint head=i/halfrot,j=i%halfrot,offset=head*d+j;
    float angle=float(p.u[3])*pow(p.f[0],-float(j)/float(halfrot));
    float c=cos(angle),s=sin(angle),a=float(x[offset]),b=float(x[offset+halfrot]);
    x[offset]=half(a*c-b*s);x[offset+halfrot]=half(b*c+a*s);
}

kernel void cache_write(device const half *k [[buffer(0)]],device const half *v [[buffer(1)]],
    device half *kc [[buffer(2)]],device half *vc [[buffer(3)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i<p.u[0]){kc[p.u[1]*p.u[0]+i]=k[i];vc[p.u[1]*p.u[0]+i]=v[i];}
}

// A SIMD group computes one head / 256-position partition with online softmax.
kernel void attention_part(device const half *q [[buffer(0)]],device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]],device float *partials [[buffer(3)]],constant Params &p [[buffer(15)]],
    uint gid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint item=gid/32,parts=p.u[0],head=item/parts,part=item%parts,d=p.u[1];
    if(head>=p.u[2])return;
    uint kvhead=head/(p.u[2]/p.u[3]),stride=p.u[3]*d;
    float acc[16];for(uint j=0;j<16;j++)acc[j]=0;
    float maximum=-INFINITY,denom=0;
    for(uint t=part*256;t<min(p.u[4],(part+1)*256);t++) {
        float score=0;for(uint j=lane;j<d;j+=32)score+=float(q[head*d+j])*float(k[t*stride+kvhead*d+j]);
        score=simd_sum(score)*p.f[0];float m=max(maximum,score),rescale=exp(maximum-m),prob=exp(score-m);
        for(uint j=lane;j<d;j+=32)acc[j/32]=acc[j/32]*rescale+prob*float(v[t*stride+kvhead*d+j]);
        denom=denom*rescale+prob;maximum=m;
    }
    uint base=item*(d+2);for(uint j=lane;j<d;j+=32)partials[base+j]=acc[j/32];
    if(lane==0){partials[base+d]=maximum;partials[base+d+1]=denom;}
}

kernel void attention_finish(device const float *partials [[buffer(0)]],device const half *qgate [[buffer(1)]],
    device half *out [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint d=p.u[1],head=i/d,j=i%d,parts=p.u[0];if(head>=p.u[2])return;
    float m=-INFINITY;for(uint t=0;t<parts;t++)m=max(m,partials[(head*parts+t)*(d+2)+d]);
    float sum=0,denom=0;for(uint t=0;t<parts;t++){
        uint base=(head*parts+t)*(d+2);float a=exp(partials[base+d]-m);
        sum+=a*partials[base+j];denom+=a*partials[base+d+1];}
    half value=half(sum/denom),gate=half(1.0f/(1.0f+exp(-float(qgate[head*2*d+d+j]))));
    out[i]=value*gate;
}

kernel void conv(device const half *x [[buffer(0)]],device const half *w [[buffer(1)]],
    device half *history [[buffer(2)]],device half *out [[buffer(3)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint n=p.u[0],taps=p.u[1];if(i>=n)return;
    float sum=float(x[i])*float(w[i*taps+taps-1]);
    for(uint t=0;t+1<taps;t++)sum+=float(history[t*n+i])*float(w[i*taps+t]);
    for(uint t=0;t+2<taps;t++)history[t*n+i]=history[(t+1)*n+i];
    if(taps>1)history[(taps-2)*n+i]=x[i];
    float a=float(half(sum));out[i]=half(a/(1.0f+exp(-a)));
}

// q/k are normalized separately. A SIMD group owns one value channel and its
// entire FP32 state row, so recurrence updates are independent and race-free.
kernel void delta(device const half *q [[buffer(0)]],device const half *k [[buffer(1)]],
    device const half *qkv [[buffer(2)]],device const half *a [[buffer(3)]],device const half *b [[buffer(4)]],
    device const half *alog [[buffer(5)]],device const half *dt [[buffer(6)]],device float *state [[buffer(7)]],
    device half *out [[buffer(8)]],constant Params &p [[buffer(15)]],uint gid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=gid/32,dk=p.u[0],dv=p.u[1],hv=p.u[2],hk=p.u[3];if(row>=hv*dv)return;
    uint head=row/dv,kh=head/(hv/hk),base=row*dk;
    float val=float(half(a[head]+dt[head]));
    float e=exp(-abs(val)), u=1.0f+e;
    float logplus=(u==1.0f)?e:log(u)*(e/(u-1.0f));
    float softplus=float(half(max(val,0.0f)+logplus));
    float decay=exp(-exp(float(alog[head]))*softplus);
    float beta=float(half(1.0f/(1.0f+exp(-float(b[head])))));
    float st[8],memory=0;
    for(uint j=lane;j<dk;j+=32){st[j/32]=state[base+j]*decay;memory+=st[j/32]*float(k[kh*dk+j]);}
    memory=simd_sum(memory);float correction=(float(qkv[2*hk*dk+row])-memory)*beta,output=0;
    for(uint j=lane;j<dk;j+=32){float v=st[j/32]+float(k[kh*dk+j])*correction;state[base+j]=v;output+=v*float(q[kh*dk+j]);}
    output=simd_sum(output);if(lane==0)out[row]=half(output);
}

kernel void swiglu(device const half *gate [[buffer(0)]],device const half *x [[buffer(1)]],
    device half *out [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i<p.u[0]){float g=float(gate[i]);out[i]=half((g/(1.0f+exp(-g)))*float(x[i]));}
}
kernel void add(device half *x [[buffer(0)]],device const half *r [[buffer(1)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i<p.u[0])x[i]+=r[i];
}
kernel void zeros(device uint *x [[buffer(0)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i<p.u[0])x[i]=0;
}
