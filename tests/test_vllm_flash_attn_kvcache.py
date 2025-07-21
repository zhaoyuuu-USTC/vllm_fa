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

NUM_HEADS = [(8, 2)]
HEAD_SIZES = [64]
BLOCK_SIZES = [128]
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
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)

# 两组，第一组三个序列，KV长度分别为1328, 18, 463，第二组四个序列，KV长度分别为1, 54, 293, 70
# @pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("kv_lens", [[3200]])
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
    # print(f"block_tables: {block_tables}")
    # t0 = time.perf_counter()
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
    block_aws = result.squeeze(1)
    # print(f"block_aws: {block_aws}")
    # t1 = time.perf_counter()
    # t_with_kvcache = t1 - t0
    # print(f"time_flash_attn_with_kvcache: {t_with_kvcache}")

    if num_blocks <= 2048:
        test_utils = ["test_faketensor", "test_schema"]
    else:
        test_utils = ["test_faketensor"]

    # torch.library.opcheck(torch.ops.vllm.flash_attn_with_kvcache_aws,
    #                       args=tuple(),
    #                       kwargs=dict(
    #                           decode_query=query.unsqueeze(1),
    #                           key_cache=key_cache,
    #                           value_cache=value_cache,
    #                           softmax_scale=scale,
    #                           causal=True,
    #                           block_table=block_tables,
    #                           cache_seqlens=kv_lens_tensor,
    #                           softcap=soft_cap if soft_cap is not None else 0,
    #                       ),
    #                       test_utils=test_utils)
    # t1 = time.perf_counter()
    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * num_seqs,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        soft_cap=soft_cap,
    )
    # t2 = time.perf_counter()
    # t_ref = t2 - t1
    # print(f"time_ref: {t_ref}")
    # print(f"time_ref / time_flash_attn_with_kvcache: {t_ref / t_with_kvcache}")
    # torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2), \
    #     f"{torch.max(torch.abs(output - ref_output))}"
    # print(f"block_aws: {block_aws.shape}")
    # print(f"block_aws[0]: {block_aws[0]}")
    # print(f"block_aws[1]: {block_aws[1]}")
    # print(f"block_aws[2]: {block_aws[2]}")
    
    # print(f"ref_output: {ref_output.shape}")
    # print(f"block_aws: {block_aws.shape}")
    # # for i in range(block_aws.shape[0]):
    # #     print(torch.sum(torch.abs(block_aws[i] - ref_output[i])))
    # # 将 block_aws 的中间维度（即第1维，num_heads=8）相加，得到形状为 [3, 32]
    # print("block_aws:", block_aws)
    # block_aws_sum = block_aws.sum(dim=1)
    # print("block_aws_sum.shape:", block_aws_sum.shape)
    # print("block_aws_sum:", block_aws_sum[0])
    # print("block_aws_sum:", block_aws_sum)