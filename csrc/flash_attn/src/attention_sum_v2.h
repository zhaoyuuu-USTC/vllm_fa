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
__device__ __forceinline__ void thread_aws_reduce_twodim_(Tensor<Engine0, Layout0> const &acc_s, Tensor<Engine1, Layout1> &row_sum, Operator &op, int page_block_size, int n_block) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 2, "Only support 2D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(row_sum) == size<0>(acc_s));
    // CUTE_STATIC_ASSERT_V(size<1>(row_sum) == size<1>(acc_s));
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_s); mi++) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_s); ni++) {
            int page_idx = (n_block * page_block_size + ni) / page_block_size;
            row_sum(mi, page_idx) = op(row_sum(mi, page_idx), acc_s(mi, ni));
        }
    }
}


template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_aws_allreduce_twodim_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        #pragma unroll
        for (int j = 0; j < size<1>(src); j++){
            dst(i, j) = Allreduce<4>::run(src(i, j), op);
        }
    }
}


template<int KMRows, int MaxPages>
struct AttentionSumV2 {
    using TensorT = decltype(make_tensor<float>(Shape<Int<KMRows>, Int<MaxPages>>{}));
    
    __forceinline__ __device__ AttentionSumV2() {};
    // 这里从后往前遍历的块
    template<typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void update_sum_aw(Tensor0 &acc_s, Tensor1 &aws, int n_block, int page_block_size, int kBlockN){
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        Tensor aws_rowcol = make_tensor(aws.data(), flash::convert_layout_acc_rowcol(aws.layout()));

        static_assert(decltype(size<0>(scores))::value == KMRows);
        // static_assert(decltype(size<1>(scores))::value == MaxPages);

        int page_idx;

        TensorT row_sum_aw;
        clear(row_sum_aw);
        AbsSumOp<float> abs_sum_op;
        flash::template thread_aws_reduce_twodim_(scores, row_sum_aw, abs_sum_op, page_block_size, n_block);
        flash::template quad_aws_allreduce_twodim_(row_sum_aw, row_sum_aw, abs_sum_op);
        #pragma unroll
        for (int mi = 0; mi < size<0>(row_sum_aw); mi++) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(row_sum_aw); ni++) {
                page_idx = (n_block * kBlockN + ni) / page_block_size;
                aws_rowcol(mi, page_idx) += row_sum_aw(mi, ni);
            }
        }

    };
};

}  // namespace flash