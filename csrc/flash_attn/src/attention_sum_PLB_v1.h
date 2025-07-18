#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "philox.cuh"
#include "utils.h"

#include <cstdio>

namespace flash {

using namespace cute;


template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_aws_reduce_PLB_(Tensor<Engine0, Layout0> const &acc_s, Tensor<Engine1, Layout1> &row_sum, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(row_sum) == size<0>(acc_s));
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_s); mi++) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_s); ni++) {
            row_sum(mi) = op(row_sum(mi), acc_s(mi, ni));
        }
    }
}


template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_aws_allreduce_PLB_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}


template<int KMRows, int MaxPages>
struct AttentionSumPLB_V1 {
    using TensorT = decltype(make_tensor<float>(Shape<Int<KMRows>>{}));
    using TensorAws = decltype(make_tensor<float>(Shape<Int<KMRows>, Int<MaxPages>>{}));
    
    TensorAws aws;
    TensorT row_sum_aw;

    __forceinline__ __device__ AttentionSumPLB_V1() {};
    // 这里从后往前遍历的块
    template<typename Tensor0>
    __forceinline__ __device__ void update_sum_aw(Tensor0 &acc_s, int n_block, int page_block_size, int kBlockN){

        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == KMRows);

        // 这个是page_size更大的场景，每个block只对应一个page, Page_size larger than kBlockN
        int page_idx = n_block * kBlockN / page_block_size;

        AbsSumOp<float> abs_sum_op;
        
        flash::template thread_aws_reduce_PLB_(scores, row_sum_aw, abs_sum_op);
        
        flash::template quad_aws_allreduce_PLB_(row_sum_aw, row_sum_aw, abs_sum_op);
        
        #pragma unroll
        for (int mi = 0; mi < size<0>(row_sum_aw); mi++) {
            aws(mi, page_idx) += row_sum_aw(mi);
        }

        clear(row_sum_aw);
    };

    template<bool Split=false>
    __forceinline__ __device__ TensorAws get_attention_weights_sum()  {
        return aws;
    }

};

}  // namespace flash