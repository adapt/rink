// Benchmark-only full attention. Retains the production 128-key partitions,
// per-lane dot products, softmax reductions and ordered value sums.
template<uint Q, uint H>
void probe_attention(device float* out,device const float* q,device const uchar* keys,device const uchar* values,device const float* new_keys,device const float* new_values,device const float* rel,device const float* profiles,constant Params& p,uint group,uint tid,uint lane,threadgroup float4* scores,threadgroup float4* maxima,threadgroup float4* totals) {
    constexpr uint V=Q*H/4;
    uint parts=p.u[8],part=group%parts,head=(group/parts)%(p.u[0]/H),first=(group/(parts*(p.u[0]/H)))*Q;
    uint kh=head*H/(p.u[0]/p.u[1]),sg=tid/32,pos=p.u[3]+p.u[11]+first,start=part*128;
    uint4 query[V],count[V];uint length=0;
    for(uint v=0;v<V;v++)for(uint c=0;c<4;c++){
        uint u=v*4+c,token=u/H,h=head*H+u%H;
        query[v][c]=(p.u[11]+min(first+token,p.u[9]-1))*p.u[0]+h;
        count[v][c]=first+token<p.u[9] && start<=pos+token?min(128u,pos+token+1-start):0;
        length=max(length,count[v][c]);
    }
    if(length==0){
        for(uint u=0;u<Q*H;u++)if(first+u/H<p.u[9]){
            uint dest=((first+u/H)*p.u[0]+head*H+u%H)*parts*130+part*130;
            if(tid==0){out[dest]=-INFINITY;out[dest+1]=0;}if(tid<128)out[dest+2+tid]=0;
        }
        return;
    }
    for(uint j=sg;j<128;j+=8){
        float4 score[V];for(uint v=0;v<V;v++)score[v]=-INFINITY;
        if(j<length){
            uint kp=start+j;bool fresh=kp>=p.u[3];uint offset=((fresh?kp-p.u[3]:kp)*p.u[1]+kh)*128;
            float4 dot[V];for(uint v=0;v<V;v++)dot[v]=0;
            for(uint i=lane;i<128;i+=32){
                float key=kv_read(keys,new_keys,offset+i,fresh,p.u[10]);
                for(uint v=0;v<V;v++)dot[v]+=float4(q[query[v].x*128+i],q[query[v].y*128+i],q[query[v].z*128+i],q[query[v].w*128+i])*key;
            }
            for(uint v=0;v<V;v++){
                score[v]=simd_sum(dot[v])*rsqrt(float(p.u[2]));float4 bias=0;
                for(uint c=0;c<4;c++){
                    uint dist=pos+(v*4+c)/H-kp;
                    if(j<count[v][c] && dist<p.u[7])for(uint i=lane;i<p.u[6];i+=32)bias[c]+=rel[query[v][c]*p.u[6]+i]*profiles[i*p.u[7]+dist];
                }
                score[v]+=simd_sum(bias);
                score[v]=select(float4(-INFINITY),score[v],uint4(j)<count[v]);
            }
        }
        if(lane==0)for(uint v=0;v<V;v++)scores[j*V+v]=score[v];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<128)for(uint v=0;v<V;v++){float4 mx=simd_max(scores[tid*V+v]);if(lane==0)maxima[sg*V+v]=mx;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 mx[V];for(uint v=0;v<V;v++)mx[v]=max(max(maxima[v],maxima[V+v]),max(maxima[2*V+v],maxima[3*V+v]));
    if(tid<128)for(uint v=0;v<V;v++){
        float4 weight=select(float4(0),exp(scores[tid*V+v]-mx[v]),count[v]>uint4(0));scores[tid*V+v]=weight;
        float4 den=simd_sum(weight);if(lane==0)totals[sg*V+v]=den;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0)for(uint v=0;v<V;v++){
        float4 den=totals[v]+totals[V+v]+totals[2*V+v]+totals[3*V+v];
        for(uint c=0;c<4;c++){uint u=v*4+c;if(first+u/H<p.u[9]){
            uint dest=((first+u/H)*p.u[0]+head*H+u%H)*parts*130+part*130;out[dest]=mx[v][c];out[dest+1]=den[c];
        }}
    }
    if(tid<128){
        float4 sum[V];for(uint v=0;v<V;v++)sum[v]=0;uint j=0;
        for(;j+3<length;j+=4){
            uint kp0=start+j+0;bool f0=kp0>=p.u[3];uint o0=((f0?kp0-p.u[3]:kp0)*p.u[1]+kh)*128+tid;float v0=kv_read(values,new_values,o0,f0,p.u[10]);
            uint kp1=start+j+1;bool f1=kp1>=p.u[3];uint o1=((f1?kp1-p.u[3]:kp1)*p.u[1]+kh)*128+tid;float v1=kv_read(values,new_values,o1,f1,p.u[10]);
            uint kp2=start+j+2;bool f2=kp2>=p.u[3];uint o2=((f2?kp2-p.u[3]:kp2)*p.u[1]+kh)*128+tid;float v2=kv_read(values,new_values,o2,f2,p.u[10]);
            uint kp3=start+j+3;bool f3=kp3>=p.u[3];uint o3=((f3?kp3-p.u[3]:kp3)*p.u[1]+kh)*128+tid;float v3=kv_read(values,new_values,o3,f3,p.u[10]);
            for(uint v=0;v<V;v++){
                if(all(count[v]==128)){sum[v]+=scores[(j+0)*V+v]*v0;sum[v]+=scores[(j+1)*V+v]*v1;sum[v]+=scores[(j+2)*V+v]*v2;sum[v]+=scores[(j+3)*V+v]*v3;}
                else{sum[v]=select(sum[v],sum[v]+scores[(j+0)*V+v]*v0,uint4(j+0)<count[v]);sum[v]=select(sum[v],sum[v]+scores[(j+1)*V+v]*v1,uint4(j+1)<count[v]);sum[v]=select(sum[v],sum[v]+scores[(j+2)*V+v]*v2,uint4(j+2)<count[v]);sum[v]=select(sum[v],sum[v]+scores[(j+3)*V+v]*v3,uint4(j+3)<count[v]);}
            }
        }
        for(;j<length;j++){
            uint kp=start+j;bool fresh=kp>=p.u[3];uint offset=((fresh?kp-p.u[3]:kp)*p.u[1]+kh)*128+tid;float value=kv_read(values,new_values,offset,fresh,p.u[10]);
            for(uint v=0;v<V;v++)sum[v]=select(sum[v],sum[v]+scores[j*V+v]*value,uint4(j)<count[v]);
        }
        for(uint v=0;v<V;v++)for(uint c=0;c<4;c++){uint u=v*4+c;if(first+u/H<p.u[9]){
            uint dest=((first+u/H)*p.u[0]+head*H+u%H)*parts*130+part*130;out[dest+2+tid]=sum[v][c];
        }}
    }
}
#define ATTENTION(NAME,Q,H) \
kernel void NAME(device float* out [[buffer(0)]],device const float* q [[buffer(1)]],device const uchar* keys [[buffer(2)]],device const uchar* values [[buffer(3)]],device const float* new_keys [[buffer(4)]],device const float* new_values [[buffer(5)]],device const float* rel [[buffer(6)]],device const float* profiles [[buffer(7)]],constant Params& p [[buffer(15)]],uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float4 scores[128*Q*H/4],maxima[Q*H],totals[Q*H]; \
    probe_attention<Q,H>(out,q,keys,values,new_keys,new_values,rel,profiles,p,group,tid,lane,scores,maxima,totals); \
}
ATTENTION(attention_q4h1,4,1)
ATTENTION(attention_q8h1,8,1)
ATTENTION(attention_q16h1,16,1)
ATTENTION(attention_q4h3,4,3)
ATTENTION(attention_q8h3,8,3)
#undef ATTENTION
