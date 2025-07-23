#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "philox.cuh"
#include "utils.h"

#include <cstdio>
#include <fstream>

namespace flash {

using namespace cute;


template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_aws_reduce_PLB_v2_(Tensor<Engine0, Layout0> const &acc_s, Tensor<Engine1, Layout1> &row_sum, Operator &op) {
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
__device__ __forceinline__ void quad_aws_allreduce_PLB_v2_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}


template<int KMRows, int MaxPages>
struct AttentionSumPLB_V2 {
    using TensorT = decltype(make_tensor<float>(Shape<Int<KMRows>>{}));
    using TensorAws = decltype(make_tensor<float>(Shape<Int<KMRows>, Int<MaxPages>>{}));
    
    TensorAws aws;

    __forceinline__ __device__ AttentionSumPLB_V2() {
        clear(aws);  // 初始化aws张量为0
    };
    // 这里从后往前遍历的块
    template<typename Tensor0>
    __forceinline__ __device__ void update_sum_aw(Tensor0 &acc_s, int n_block, int page_block_size, int kBlockN){
        TensorT row_sum_aw;  // 使用与aws相同的数据类型
        
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == KMRows);
        // printf("ASPLB_V2  KMRows: %d, scores.shape: %d x %d\n", KMRows, int(size<0>(scores)), int(size<1>(scores))); 2 x 64
        // 这个是page_size更大的场景，每个block只对应一个page, Page_size larger than kBlockN

        int page_nums = kBlockN / page_block_size;
        // printf("page_nums: %d, kBlockN: %d, page_block_size: %d\n", page_nums, kBlockN, page_block_size);
        int start_page_index = n_block * kBlockN / page_block_size;

        AbsSumOp<float> abs_sum_op;

        #pragma unroll
        for (int i = 0; i < page_nums; i++) {
            clear(row_sum_aw);  // 确保每次循环开始时清零

            int page_index = start_page_index + i;
            // printf("page_index: %d\n", page_index);
            // 256 / 128  == 2
            int start_index = i * page_block_size;
            int end_index = (i+1) * page_block_size;
            // printf(" start_page_index: %d, page_index: %d, start_index: %d, end_index: %d\n", start_page_index, page_index, start_index, end_index);
            // 直接使用原始scores张量，通过索引访问对应的页面数据
            // 避免创建依赖于运行时值的张量形状
            #pragma unroll
            for (int mi = 0; mi < KMRows; mi++) {
                #pragma unroll
                for (int ni = start_index; ni < end_index; ni++) {
                    row_sum_aw(mi) = abs_sum_op(row_sum_aw(mi), scores(mi, ni));
                }
            }
            flash::template quad_aws_allreduce_PLB_v2_(row_sum_aw, row_sum_aw, abs_sum_op);
            #pragma unroll
            for (int mi = 0; mi < size<0>(row_sum_aw); mi++) {
                aws(mi, page_index) += row_sum_aw(mi);
            }
        }
    };

    template<bool Split=false>
    __forceinline__ __device__ TensorAws get_attention_weights_sum()  {
        return aws;
    }

};

}  // namespace flash