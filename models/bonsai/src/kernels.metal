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

kernel void embedding_batch(device const uint *w [[buffer(0)]], device const half *s [[buffer(1)]],
    device const half *b [[buffer(2)]], device half *out [[buffer(3)]], device const uint *tokens [[buffer(4)]],
    constant Params &p [[buffer(15)]], uint i [[thread_position_in_grid]]) {
    uint width=p.u[0];
    if(i<width*p.u[1]) { uint j=tokens[i/width]*width+i%width; out[i]=half(float((w[j/16]>>(2*(j%16)))&3)*float(s[j/128])+float(b[j/128])); }
}

// Normalized Sylvester transform: forward H(x*sign), inverse H(x)*sign.
// One threadgroup owns a block; no intergroup dependencies or in-place aliasing.
kernel void hadamard(device const half *x [[buffer(0)]], device const half *sign [[buffer(1)]],
    device half *out [[buffer(2)]], constant Params &p [[buffer(15)]],
    uint tid [[thread_index_in_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
    threadgroup float v[4096];
    uint n=p.u[0], base=block*n,signbase=p.u[2]?base%p.u[2]:base; bool inverse=p.u[1];
    for(uint i=tid;i<n;i+=256) v[i]=float(x[base+i])*(inverse?1.0f:float(sign[signbase+i]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=1;stride<n;stride*=2) {
        for(uint j=tid;j<n/2;j+=256) {
            uint a=(j/stride)*(2*stride)+j%stride, b=a+stride;
            float left=v[a],right=v[b]; v[a]=left+right;v[b]=left-right;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale=rsqrt(float(n));
    for(uint i=tid;i<n;i+=256) out[base+i]=half(v[i]*scale*(inverse?float(sign[signbase+i]):1.0f));
}

// The native block keeps the five within-SIMD butterfly stages in registers.
kernel void hadamard1024(device const half *x [[buffer(0)]],device const half *sign [[buffer(1)]],
 device half *out [[buffer(2)]],constant Params &p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],
 uint block [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 threadgroup float values[1024];uint base=block*1024,signbase=base%p.u[2];bool inverse=p.u[1];
 for(uint part=0;part<4;part++){
  uint i=tid+part*256;float value=float(x[base+i])*(inverse?1.f:float(sign[signbase+i]));
  for(uint stride=1;stride<32;stride*=2){float other=simd_shuffle_xor(value,stride);value=(lane&stride)?other-value:value+other;}
  values[i]=value;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint stride=32;stride<1024;stride*=2){for(uint j=tid;j<512;j+=256){uint a=(j/stride)*(2*stride)+j%stride,b=a+stride;float left=values[a],right=values[b];values[a]=left+right;values[b]=left-right;}threadgroup_barrier(mem_flags::mem_threadgroup);}
 for(uint i=tid;i<1024;i+=256)out[base+i]=half(values[i]*(1.f/32.f)*(inverse?float(sign[signbase+i]):1.f));
}

// Width is divisible by 128. Each SIMD group owns one output row.
kernel void packed_mv(device const uint *w [[buffer(0)]], device const half *s [[buffer(1)]],
    device const half *b [[buffer(2)]], device const half *x [[buffer(3)]], device half *out [[buffer(4)]],
    constant Params &p [[buffer(15)]], uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=gid/32, width=p.u[0]; if(row>=p.u[1])return;
    float4 total[4];for(uint i=0;i<4;i++)total[i]=0;
    // Two complete words per lane share their scale and bias. Independent
    // vector sums remove the dependency chain between consecutive dot products.
    for(uint word=lane*2;word<width/16;word+=64) {
        uint j=row*(width/16)+word;uint2 codes=*((device const uint2*)(w+j));
        float scale=float(s[j/8]),bias=float(b[j/8]);
        #pragma unroll
        for(uint z=0;z<2;z++) {
            #pragma unroll
            for(uint i=0;i<4;i++) {
                float4 weights=float4(half4(float4((uint4(codes[z])>>(uint4(0,2,4,6)+8*i))&3)*scale+bias));
                float4 inputs=float4(*((device const half4*)(x+(word+z)*16+i*4)));
                total[i]+=weights*inputs;
            }
        }
    }
    float4 values=(total[0]+total[1])+(total[2]+total[3]);
    float sum=simd_sum((values.x+values.y)+(values.z+values.w));if(lane==0)out[row]=half(sum);
}

// Fast path only for finite FP16 scales with bias == -scale and 2*scale
// representable in FP16. All four code values then dequantize exactly to
// (code-1)*scale. Factor the common scale after each sixteen-coefficient sum.
// The generic kernel retains independently rounded affine weights.
kernel void packed_mv_symmetric(device const uint*w [[buffer(0)]],device const half*s [[buffer(1)]],
 device const half*b [[buffer(2)]],device const half*x [[buffer(3)]],device half*y [[buffer(4)]],
 constant Params&p [[buffer(15)]],uint gid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 uint row=(gid/32)*2,width=p.u[0];float sums[2]={0,0};
 for(uint word=lane;word<width/16;word+=32){float4 inputs[4];float input_sum=0;
  for(uint j=0;j<4;j++){float4 v=float4(*((device const half4*)(x+word*16+4*j)));input_sum+=(v.x+v.y)+(v.z+v.w);inputs[j]=v*float4(1.f,0.25f,0.0625f,0.015625f);}
  for(uint r=0;r<2;r++)if(row+r<p.u[1]){uint idx=(row+r)*(width/16)+word;uchar4 bytes=as_type<uchar4>(w[idx]);float dot=0;
   for(uint j=0;j<4;j++)dot+=metal::dot(float4(uint4(bytes[j])&uint4(3,12,48,192)),inputs[j]);
   sums[r]+=(dot-input_sum)*float(s[idx/8]);}
 }
 for(uint r=0;r<2;r++){float sum=simd_sum(sums[r]);if(lane==0 && row+r<p.u[1])y[row+r]=half(sum);}
}

kernel void dense_mv(device const half *w [[buffer(0)]], device const half *x [[buffer(1)]],
    device half *out [[buffer(2)]], constant Params &p [[buffer(15)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];uint width=p.u[0];if(row>=p.u[1])return;
    float sum=0;for(uint i=tid*4;i<width;i+=256*4)sum+=dot(float4(*((device const half4*)(w+row*width+i))),float4(*((device const half4*)(x+i))));
    sum=simd_sum(sum);if(lane==0)partial[tid/32]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0){float total=0;for(uint i=0;i<8;i++)total+=partial[i];out[row]=half(total);}
}

// p: width, heads, input head stride, use weight; f: epsilon, post scale.
kernel void rms(device const half *x [[buffer(0)]], device const half *w [[buffer(1)]],
    device half *out [[buffer(2)]], constant Params &p [[buffer(15)]],
    uint tid [[thread_index_in_threadgroup]], uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];uint n=p.u[0],token=head/p.u[1],localhead=head%p.u[1];
    uint base=token*p.u[6]+localhead*p.u[2]+p.u[4];
    float v=0;for(uint i=tid;i<n;i+=256)v+=float(x[base+i])*float(x[base+i]);
    v=simd_sum(v);if(lane==0)sums[simd]=v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=0;for(uint j=0;j<8;j++)total+=sums[j];
    float inv=rsqrt(total/float(n)+p.f[0]);
    for(uint i=tid;i<n;i+=256){half normalized=half(float(x[base+i])*inv*(p.u[3]?float(w[i]):1.0f));
        out[head*n+i]=half(float(normalized)*p.f[1]);}
}

// Small heads need one SIMD reduction, without inter-SIMD barriers.
kernel void rms_small(device const half*x [[buffer(0)]],device const half*w [[buffer(1)]],device half*out [[buffer(2)]],
 constant Params&p [[buffer(15)]],uint gid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 uint head=gid/32,n=p.u[0];if(head>=p.u[1]*p.u[5])return;uint token=head/p.u[1],localhead=head%p.u[1],base=token*p.u[6]+localhead*p.u[2]+p.u[4];
 float sum=0;for(uint i=lane;i<n;i+=32)sum+=float(x[base+i])*float(x[base+i]);
 float inv=rsqrt(simd_sum(sum)/float(n)+p.f[0]);
 for(uint i=lane;i<n;i+=32){half normalized=half(float(x[base+i])*inv*(p.u[3]?float(w[i]):1.f));out[head*n+i]=half(float(normalized)*p.f[1]);}
}

kernel void rope(device half *x [[buffer(0)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint d=p.u[0],rot=p.u[1],halfrot=rot/2;if(i>=max(1u,p.u[4])*p.u[2]*halfrot)return;
    uint head=i/halfrot,j=i%halfrot,offset=head*d+j;
    float angle=float(p.u[3]+head/p.u[2])*pow(p.f[0],-float(j)/float(halfrot));
    float c=cos(angle),s=sin(angle),a=float(x[offset]),b=float(x[offset+halfrot]);
    x[offset]=half(a*c-b*s);x[offset+halfrot]=half(b*c+a*s);
}

kernel void cache_write(device const half *k [[buffer(0)]],device const half *v [[buffer(1)]],
    device half *kc [[buffer(2)]],device half *vc [[buffer(3)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i<p.u[0]*max(1u,p.u[2])){kc[p.u[1]*p.u[0]+i]=k[i];vc[p.u[1]*p.u[0]+i]=v[i];}
}

// Parallel score reduction and value accumulation within each bounded KV tile.
kernel void attention_part(device const half *q [[buffer(0)]],device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]],device float *partials [[buffer(3)]],constant Params &p [[buffer(15)]],
    uint item [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[256],reduction[8],denominators[8];
    uint parts=p.u[0],head=(item/parts)%p.u[2],part=item%parts,d=p.u[1];
    uint query=item/(parts*p.u[2]),token=p.u[5]+query,size=p.u[7];
    if(query>=max(1u,p.u[6]))return;
    uint kh=head/(p.u[2]/p.u[3]),stride=p.u[3]*d,first=part*size;
    uint length=first<p.u[4]+token?min(size,p.u[4]+token-first):0;
    for(uint t=tid/32;t<size;t+=4){
        float score=-INFINITY;
        if(t<length){score=0;for(uint j=lane;j<d;j+=32)score+=float(q[(token*p.u[2]+head)*d+j])*float(k[(first+t)*stride+kh*d+j]);score=simd_sum(score)*p.f[0];}
        if(lane==0)scores[t]=score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(uint t=tid;t<size;t+=128)maximum=max(maximum,scores[t]);
    maximum=simd_max(maximum);if(lane==0)reduction[tid/32]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=max(max(reduction[0],reduction[1]),max(reduction[2],reduction[3]));
    float denom=0;for(uint t=tid;t<size;t+=128){float prob=t<length?exp(scores[t]-maximum):0;scores[t]=prob;denom+=prob;}
    denom=simd_sum(denom);if(lane==0)denominators[tid/32]=denom;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint dest=item*(d+2);
    for(uint j=tid;j<d;j+=128){float sum=0;for(uint t=0;t<length;t++)sum+=scores[t]*float(v[(first+t)*stride+kh*d+j]);partials[dest+j]=sum;}
    if(tid==0){partials[dest+d]=maximum;partials[dest+d+1]=(denominators[0]+denominators[1])+(denominators[2]+denominators[3]);}
}

kernel void attention_finish(device const float *partials [[buffer(0)]],device const half *qgate [[buffer(1)]],
    device half *out [[buffer(2)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint d=p.u[1],head=(i/d)%p.u[2],j=i%d,parts=p.u[0],query=i/(p.u[2]*d);
    if(query>=max(1u,p.u[4]))return;
    uint item=query*p.u[2]+head,token=p.u[3]+query;
    float m=-INFINITY;for(uint t=0;t<parts;t++)m=max(m,partials[(item*parts+t)*(d+2)+d]);
    float sum=0,denom=0;for(uint t=0;t<parts;t++){
        uint base=(item*parts+t)*(d+2);float a=exp(partials[base+d]-m);
        sum+=a*partials[base+j];denom+=a*partials[base+d+1];}
    half value=half(sum/denom),gate=half(1.0f/(1.0f+exp(-float(qgate[(token*p.u[2]+head)*2*d+d+j]))));
    out[(token*p.u[2]+head)*d+j]=value*gate;
}

kernel void conv(device const half *x [[buffer(0)]],device const half *w [[buffer(1)]],
    device half *history [[buffer(2)]],device half *out [[buffer(3)]],constant Params &p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint n=p.u[0],taps=p.u[1];if(i>=n)return;
    half h[15];for(uint t=0;t+1<taps;t++)h[t]=history[t*n+i];
    for(uint token=0;token<max(1u,p.u[2]);token++) {
        float sum=float(x[token*n+i])*float(w[i*taps+taps-1]);
        for(uint t=0;t+1<taps;t++)sum+=float(h[t])*float(w[i*taps+t]);
        for(uint t=0;t+2<taps;t++)h[t]=h[t+1];
        if(taps>1)h[taps-2]=x[token*n+i];
        float a=float(half(sum));out[token*n+i]=half(a/(1.0f+exp(-a)));
    }
    for(uint t=0;t+1<taps;t++)history[t*n+i]=h[t];
}

kernel void conv_batch(device const half*x [[buffer(0)]],device const half*w [[buffer(1)]],device const half*history [[buffer(2)]],device half*out [[buffer(3)]],constant Params&p [[buffer(15)]],uint gid [[thread_position_in_grid]]) {
 uint n=p.u[0],taps=p.u[1];if(gid>=n*p.u[2])return;uint token=gid/n,i=gid%n;
 float sum=float(x[gid])*float(w[i*taps+taps-1]);
 for(uint t=0;t+1<taps;t++){int position=int(token)+int(t)-int(taps-1);half value=position<0?history[(position+int(taps-1))*n+i]:x[uint(position)*n+i];sum+=float(value)*float(w[i*taps+t]);}
 float v=float(half(sum));out[gid]=half(v/(1.f+exp(-v)));
}
kernel void conv_save(device const half*x [[buffer(0)]],device half*history [[buffer(1)]],constant Params&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
 uint n=p.u[0],taps=p.u[1],batch=p.u[2];if(i>=n)return;half old[15];for(uint t=0;t+1<taps;t++)old[t]=history[t*n+i];
 for(uint t=0;t+1<taps;t++){int position=int(batch)+int(t)-int(taps-1);history[t*n+i]=position<0?old[position+int(taps-1)]:x[uint(position)*n+i];}
}

kernel void delta_gates(device const half*a [[buffer(0)]],device const half*b [[buffer(1)]],device const half*alog [[buffer(2)]],device const half*dt [[buffer(3)]],device float2*gates [[buffer(4)]],constant Params&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
 if(i>=p.u[0]*p.u[1])return;uint head=i%p.u[0];
 float val=float(half(a[i]+dt[head]));float e=exp(-abs(val)),u=1.f+e;
 float logplus=(u==1.f)?e:log(u)*(e/(u-1.f));
 float softplus=float(half(max(val,0.f)+logplus));
 float beta=float(half(1.f/(1.f+exp(-float(b[i])))));
 gates[i]=float2(exp(-exp(float(alog[head]))*softplus),beta);
}

// q/k are normalized separately. A SIMD group owns one value channel and its
// entire FP32 state row, so recurrence updates are independent and race-free.
kernel void delta(device const half *q [[buffer(0)]],device const half *k [[buffer(1)]],
    device const half *qkv [[buffer(2)]],device const float2 *gates [[buffer(3)]],device float *state [[buffer(4)]],
    device half *out [[buffer(5)]],constant Params &p [[buffer(15)]],uint gid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=gid/32,dk=p.u[0],dv=p.u[1],hv=p.u[2],hk=p.u[3];if(row>=hv*dv)return;
    uint head=row/dv,kh=head/(hv/hk),base=row*dk;
    float st[8];for(uint j=lane;j<dk;j+=32)st[j/32]=state[base+j];
    for(uint token=0;token<max(1u,p.u[4]);token++) {
        float2 gate=gates[token*hv+head];float decay=gate.x,beta=gate.y;
        float memory=0;
        for(uint j=lane;j<dk;j+=32){st[j/32]*=decay;memory+=st[j/32]*float(k[(token*hk+kh)*dk+j]);}
        memory=simd_sum(memory);
        float correction=(float(qkv[token*(2*hk*dk+hv*dv)+2*hk*dk+row])-memory)*beta,output=0;
        for(uint j=lane;j<dk;j+=32){st[j/32]+=float(k[(token*hk+kh)*dk+j])*correction;output+=st[j/32]*float(q[(token*hk+kh)*dk+j]);}
        output=simd_sum(output);if(lane==0)out[token*hv*dv+row]=half(output);
    }
    for(uint j=lane;j<dk;j+=32)state[base+j]=st[j/32];
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

struct Choice { float value; uint index; };
inline Choice choose_simd(Choice value) {
    float maximum=simd_max(value.value);
    uint index=simd_min(value.value==maximum?value.index:0xffffffffu);
    return Choice{maximum,index};
}
kernel void greedy_part(device const half*logits [[buffer(0)]],device Choice*partial [[buffer(1)]],
 constant Params&p [[buffer(15)]],uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    threadgroup Choice groups[8];Choice best={-INFINITY,0xffffffffu};
    for(uint i=group*1024+tid;i<min(p.u[0],(group+1)*1024);i+=256){float v=float(logits[i]);Choice next={isfinite(v)?v:INFINITY,isfinite(v)?i:0xffffffffu};if(next.value>best.value || (next.value==best.value && next.index<best.index))best=next;}
    best=choose_simd(best);if(lane==0)groups[tid/32]=best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<32){best=tid<8?groups[tid]:Choice{-INFINITY,0xffffffffu};best=choose_simd(best);if(tid==0)partial[group]=best;}
}
kernel void greedy_finish(device const Choice*partial [[buffer(0)]],device uint*token [[buffer(1)]],
 constant Params&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    threadgroup Choice groups[8];Choice best={-INFINITY,0xffffffffu};
    for(uint i=tid;i<p.u[0];i+=256){Choice next=partial[i];if(next.value>best.value || (next.value==best.value && next.index<best.index))best=next;}
    best=choose_simd(best);if(lane==0)groups[tid/32]=best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<32){best=tid<8?groups[tid]:Choice{-INFINITY,0xffffffffu};best=choose_simd(best);if(tid==0)token[0]=best.index;}
}
