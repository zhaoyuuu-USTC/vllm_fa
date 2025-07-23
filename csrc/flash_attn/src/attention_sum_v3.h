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
__device__ __forceinline__ void quad_aws_allreduce_PLB_V3_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<int KMRows, int MaxPages>
struct AttentionSum_V3 {
    using TensorT = decltype(make_tensor<float>(Shape<Int<KMRows>>{}));
    using TensorAws = decltype(make_tensor<float>(Shape<Int<KMRows>, Int<MaxPages>>{}));
    
    TensorAws aws;

    __forceinline__ __device__ AttentionSum_V3() {
        clear(aws);  // 初始化aws张量为0
    };

    template<typename Tensor0>
    __forceinline__ __device__ void update_sum_aw(Tensor0 &acc_s, int n_block, int page_block_size, int kBlockN){
        TensorT row_sum_aw;
        clear(row_sum_aw);  // 每次调用时清零

        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == KMRows);

        AbsSumOp<float> abs_sum_op;
        // printf("kBlockN: %d, page_block_size: %d\n", kBlockN, page_block_size);
        if (kBlockN > page_block_size){
            assert(kBlockN % page_block_size == 0);
            int page_nums = kBlockN / page_block_size;

            int start_page_index = n_block * kBlockN / page_block_size;
            
            #pragma unroll
            for (int i = 0; i < page_nums; i++) {
                clear(row_sum_aw);  // 确保每次循环开始时清零

                int page_index = start_page_index + i;
                // printf("page_index: %d\n", page_index);
                // 256 / 128  == 2
                int start_index = i * page_block_size;
                int end_index = (i+1) * page_block_size;

                #pragma unroll
                for (int mi = 0; mi < KMRows; mi++) {
                    #pragma unroll
                    for (int ni = start_index; ni < end_index; ni++) {
                        row_sum_aw(mi) = abs_sum_op(row_sum_aw(mi), scores(mi, ni));
                    }
                }
                flash::template quad_aws_allreduce_PLB_V3_(row_sum_aw, row_sum_aw, abs_sum_op);
                #pragma unroll
                for (int mi = 0; mi < size<0>(row_sum_aw); mi++) {
                    aws(mi, page_index) += row_sum_aw(mi);
                }
            }
        }
        else if (kBlockN <= page_block_size){
            assert(page_block_size % kBlockN == 0);
            int page_index = n_block * kBlockN / page_block_size;
            
            #pragma unroll
            for (int mi = 0; mi < size<0>(scores); mi++){
                #pragma unroll
                for (int ni = 0; ni < size<1>(scores); ni++){
                    row_sum_aw(mi) = abs_sum_op(row_sum_aw(mi), scores(mi, ni));
                }
            }
            flash::template quad_aws_allreduce_PLB_V3_(row_sum_aw, row_sum_aw, abs_sum_op);
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