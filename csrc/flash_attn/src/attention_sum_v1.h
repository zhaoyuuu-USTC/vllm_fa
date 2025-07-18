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
__device__ __forceinline__ void thread_aws_reduce_(Tensor<Engine0, Layout0> const &acc_s, Tensor<Engine1, Layout1> &row_sum, Operator &op) {
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
__device__ __forceinline__ void quad_aws_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}


template<int KMRows>
struct AttentionSumV1 {

    using TensorT = decltype(make_tensor<float>(Shape<Int<KMRows>>{}));
    
    __forceinline__ __device__ AttentionSumV1() {};
    // 这里从后往前遍历的块
    template<typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void update_sum_aw(Tensor0 &acc_s, Tensor1 &aws, int n_block, int page_block_size, int kBlockN){

        // printf("acc_s size: %d x %d x %d\n", int(size<0>(acc_s)), int(size<1>(acc_s)), int(size<2>(acc_s)));    4 x 1 x 32
        // printf("aws size: %d x %d x %d\n", int(size<0>(aws)), int(size<1>(aws)), int(size<2>(aws)));  4 x 1 x 4

        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        Tensor aws_rowcol = make_tensor(aws.data(), flash::convert_layout_acc_rowcol(aws.layout()));

        static_assert(decltype(size<0>(scores))::value == KMRows);
        // static_assert(decltype(size<1>(scores))::value == MaxPages);
        // 这个是page_size更大的场景，每个block只对应
        int page_idx = n_block * kBlockN / page_block_size;
        TensorT row_sum_aw;
        clear(row_sum_aw);

        AbsSumOp<float> abs_sum_op;
        // 输出scores和row_sum_aw的size   这里肯定没写对，应该是存的横坐标为 MaxPageSize
        // printf("scores size: %d x %d\n", int(size<0>(scores)), int(size<1>(scores)));  2 x 64
        // printf("row_sum_aw size: %d\n", int(size<0>(row_sum_aw)));    2
        
        flash::template thread_aws_reduce_(scores, row_sum_aw, abs_sum_op);
        
        // printf("row_sum_aw_before: %f, %f\n", row_sum_aw[0], row_sum_aw[1]);
        flash::template quad_aws_allreduce_(row_sum_aw, row_sum_aw, abs_sum_op);
        // printf("row_sum_aw_after: %f, %f\n", row_sum_aw[0], row_sum_aw[1]);
        

        // 输出row_sum_aw
        // printf("row_sum_aw: %d, %d\n", n_block, page_idx);
        // #pragma unroll
        // for (int mi = 0; mi < size<0>(row_sum_aw); mi++) {
        //     printf("%f ", static_cast<float>(row_sum_aw(mi)));
        // }
        // printf("\n");
        
        #pragma unroll
        for (int mi = 0; mi < size<0>(row_sum_aw); mi++) {
            aws_rowcol(mi, page_idx) += row_sum_aw(mi);
        }

    };
};

}  // namespace flash