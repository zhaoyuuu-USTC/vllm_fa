#
# This file is copied verbatim from vLLM:
# https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_flash_attn.py
#

from typing import List, Optional, Tuple

import pytest
import torch
import time

import flash_attn_wrapper  # noqa: F401

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
# HEAD_SIZES = [128, 256]
# BLOCK_SIZES = [16, 32]
# DTYPES = [torch.float16, torch.bfloat16]
# # one value large enough to test overflow in index calculation.
# # one value small enough to test the schema op check
# NUM_BLOCKS = [32768, 2048]

# NUM_HEADS = [(32, 2), (16, 2), (8, 2), (32, 4), (16, 4), (8, 4)]
NUM_HEADS = [(32, 2)]
HEAD_SIZES = [96]
BLOCK_SIZES = [128]
KV_LENS = [[2048]]
DTYPES = [torch.float16]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [2048]


def ref_paged_attn(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        query_lens: List[int],
        kv_lens: List[int],
        block_tables: torch.Tensor,
        scale: float,
        sliding_window: Optional[int] = None,
        soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    attn_raws: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = torch.triu(empty_mask,
                                             diagonal=kv_len -
                                                      (query_len + sliding_window) +
                                                      1).bool().logical_not()
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn_raw = attn.clone()
        attn_raws.append(attn_raw)
        # print(f"attn_raw.shape: {attn_raw.shape}")
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len
    return torch.cat(outputs, dim=0), attn_raws

# 两组，第一组三个序列，KV长度分别为1328, 18, 463，第二组四个序列，KV长度分别为1, 54, 293, 70
# @pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("kv_lens", KV_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
# @pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@torch.inference_mode()
def test_flash_attn_with_paged_kv(
        kv_lens: List[int],
        num_heads: Tuple[int, int],
        head_size: int,
        dtype: torch.dtype,
        block_size: int,
        soft_cap: Optional[float],
        num_blocks: int,
) -> None:
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_size,
                            dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size

    block_tables = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)
    block_tables_copy = torch.empty((num_seqs, max_num_blocks_per_seq), dtype=torch.int32, device=block_tables.device)

    block_tables_copy[:, :max_num_blocks_per_seq] = block_tables
    
    # t_2 = time.perf_counter()
    # for i in range(1):
    #     result_raw = torch.ops.vllm.flash_attn_with_kvcache(
    #         decode_query=query.unsqueeze(1),
    #         key_cache=key_cache,
    #         value_cache=value_cache,
    #         softmax_scale=scale,
    #         causal=True,
    #         block_table=block_tables,
    #         cache_seqlens=kv_lens_tensor,
    #         softcap=soft_cap if soft_cap is not None else 0,
    #     )
    # t_1 = time.perf_counter()
    # print(f"time_flash_attn_with_kvcache_raw: {(t_1 - t_2)}")

    t0 = time.perf_counter()
    t0 = time.perf_counter()
    for i in range(1):
        result = torch.ops.vllm.flash_attn_with_kvcache_aws(
            decode_query=query.unsqueeze(1),
            key_cache=key_cache,
            value_cache=value_cache,
            softmax_scale=scale,
            causal=True,
            block_table=block_tables,
            cache_seqlens=kv_lens_tensor,
            softcap=soft_cap if soft_cap is not None else 0,
        )
    t1 = time.perf_counter()

    block_aws_list = result

    if num_blocks <= 2048:
        test_utils = ["test_faketensor", "test_schema"]
    else:
        test_utils = ["test_faketensor"]
    t2 = time.perf_counter()
    for i in range(1):
        ref_output, attn_list = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=[1] * num_seqs,
            kv_lens=kv_lens,
            block_tables=block_tables_copy,
            scale=scale,
            soft_cap=soft_cap,
        )
    t3 = time.perf_counter()
    print(f"time_ref_paged_attn: {(t3 - t2)}")
    for i in range(len(attn_list)):
        # print(f"attn[{i}].shape: {attn[i].shape}")   # (32, 1, 1500) (32, 1, 1300)
        attn = attn_list[i].abs().squeeze(1).mean(dim=0) # (32, 1500) (32, 1300)

        block_aws = block_aws_list[i].abs().squeeze(0).mean(dim=0) 
        # print(f"attn.shape: {attn.shape}")            # len
        # print(f"block_aws.shape: {block_aws.shape}")  # (32)
 
        total_kv_len = attn.shape[0]
        # print(f"total_kv_len: {total_kv_len}")
        num_blocks = (total_kv_len + block_size - 1) // block_size
        pad_len = 32 * block_size - total_kv_len
        if pad_len > 0:
            attn_padded = torch.cat([attn, attn.new_zeros(pad_len)], dim=0)
        else:
            attn_padded = attn
        attn_blocks = attn_padded.view(32, block_size)

        attn_blocks_sum_mean = attn_blocks.abs().mean(dim=1)

        # 分别对 attn_blocks_sum_mean 和 block_aws_sum_mean 进行归一化
        attn_blocks_sum_mean_norm = (attn_blocks_sum_mean - attn_blocks_sum_mean.min()) / (attn_blocks_sum_mean.max() - attn_blocks_sum_mean.min() + 1e-8)
        block_aws_sum_mean_norm = (block_aws - block_aws.min()) / (block_aws.max() - block_aws.min() + 1e-8)

        # print(f"attn_blocks_sum_mean_norm: {attn_blocks_sum_mean_norm}")
        # print(f"block_aws_sum_mean_norm: {block_aws_sum_mean_norm}")

        print(f"attn_blocks_sum_mean_norm - block_aws_sum_mean_norm: {attn_blocks_sum_mean_norm - block_aws_sum_mean_norm}")
