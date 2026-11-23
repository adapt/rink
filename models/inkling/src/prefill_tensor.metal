// Optional Metal 4 implementation. Keep this library separate so unsupported
// SDKs/devices can compile and use the original projection kernels.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct Params { uint u[16]; float f[16]; };

// BF16 weights, FP32 inputs/output, 64x64 output tiles and 128 threads.
// Precision relaxation is explicitly disabled. Tensor views share the existing
// buffers; this kernel never stores a dequantized weight matrix.
kernel void matmul_prefill_tensor(device float* y [[buffer(0)]],
    device const uchar* w [[buffer(1)]], device const float* x [[buffer(2)]],
    constant Params& p [[buffer(15)]], uint group [[threadgroup_position_in_grid]]) {
    int rows=p.u[0], cols=p.u[1], batch=p.u[3], tiles=(rows+63)/64;
    int row=(group%tiles)*64, first=(group/tiles)*64;
    // MPP's tensor element types must be unqualified. Both operands are read
    // only; the casts do not permit any writes through the input views here.
    auto a=tensor(const_cast<device float*>(x)+p.u[4],
        dextents<int,2>(cols,batch),array<int,2>{1,cols});
    auto b=tensor((device bfloat*)w,
        dextents<int,2>(cols,rows),array<int,2>{1,cols});
    auto c=tensor(y,dextents<int,2>(rows,batch),array<int,2>{1,rows});
    auto av=a.slice(0,first);
    auto bv=b.slice(0,row);
    auto cv=c.slice(row,first);
    constexpr auto desc=matmul2d_descriptor(64,64,int(dynamic_extent),false,true,false);
    matmul2d<desc,execution_simdgroups<4>> op;
    op.run(av,bv,cv);
}
