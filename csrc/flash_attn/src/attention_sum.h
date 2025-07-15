#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "philox.cuh"
#include "utils.h"

namespace flash {

using namespace cute;


template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_aws_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_aws_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum_aw(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    AbsSumOp<float> abs_sum_op;   // 绝对值求和，其中包含inf的值会被置0
    thread_aws_reduce_<zero_init>(tensor, sum, abs_sum_op);
}

template<int KMRows>
struct AttentionSum {

    using TensorT = decltype(make_tensor<float>(Shape<Int<KMRows>>{}));
    TensorT row_sum_aw;
    
    __forceinline__ __device__ AttentionSum() {};
    // 这里从后往前遍历的块
    template<typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void update_sum_aw(Tensor0 &acc_s, Tensor1 &aws, int n_block, int page_block_size, int kBlockN){
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        
        static_assert(decltype(size<0>(scores))::value == KMRows);

        flash::template reduce_sum_aw</*zero_init=*/true>(scores, row_sum_aw);
        AbsSumOp<float> abs_sum_op;
        flash::template quad_aws_allreduce_(row_sum_aw, row_sum_aw, abs_sum_op);
        
        Tensor aws_rowcol = make_tensor(aws.data(), flash::convert_layout_acc_rowcol(aws.layout()));

        #pragma unroll
        for (int mi = 0; mi < size(row_sum_aw); ++mi) {
            int page_idx = n_block * kBlockN / page_block_size;
            aws_rowcol(mi, page_idx) += row_sum_aw(mi);
        }
    };
};

}  // namespace flash