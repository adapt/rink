// Benchmark reference: native entry points from a0e92fb.
// These call the unchanged generic matrix helpers from the production library.
// Update this comparison if those helpers or the model's dispatch rules change.
// Native dimensions and format are compile-time constants; the arithmetic and
// adaptive 16/32/64-slot tiles match expert_matmul_prefill.
template<uint Cols,uint Format,bool Down>
void previous_native_expert_tile(device float* y,device const uchar* w,device const uchar* scales,device const float* factors,device const float* x,device const uint* groups,device const uint* tiles,device const float* table,constant Params& p,uint group,uint tid,uint lane,threadgroup float* scratch){
    uint tile=group/36;if(tile>=tiles[0])return;
    uint expert=tiles[1+2*tile],first=tiles[2+2*tile],remaining=groups[expert*(p.u[5]+1)]-first;
    if(remaining<=16)prefill_matrix_tile<16,64,32,true>(y,w,scales,factors,x,groups,tiles,table,2304,Cols,8,Format,Down,p.u[5],0,group,tid,lane,scratch);
    else if(remaining<=32)prefill_matrix_tile<32,64,32,true>(y,w,scales,factors,x,groups,tiles,table,2304,Cols,8,Format,Down,p.u[5],0,group,tid,lane,scratch);
    else prefill_expert_pipeline<64,64,16,true>(y,w,scales,factors,x,groups,tiles,table,2304,Cols,8,Format,Down,p.u[5],group,tid,scratch);
}
#define NATIVE_EXPERT(NAME,COLS,FORMAT,DOWN) \
kernel void NAME(device float* y [[buffer(0)]],device const uchar* w [[buffer(1)]],device const uchar* scales [[buffer(2)]],device const float* factors [[buffer(3)]],device const float* x [[buffer(4)]],device const uint* groups [[buffer(5)]],device const uint* tiles [[buffer(6)]],device const float* table [[buffer(7)]],constant Params& p [[buffer(15)]],uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float scratch[4096]; \
    previous_native_expert_tile<COLS,FORMAT,DOWN>(y,w,scales,factors,x,groups,tiles,table,p,group,tid,lane,scratch); \
}
NATIVE_EXPERT(previous_expert_prefill_nvfp4_up,2304,0,false)
NATIVE_EXPERT(previous_expert_prefill_nvfp4_down,1152,0,true)
NATIVE_EXPERT(previous_expert_prefill_mxfp8_up,2304,1,false)
NATIVE_EXPERT(previous_expert_prefill_mxfp8_down,1152,1,true)
#undef NATIVE_EXPERT
