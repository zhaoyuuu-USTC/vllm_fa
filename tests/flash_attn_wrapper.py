from typing import Optional, List, Tuple
import torch

from vllm_flash_attn.flash_attn_interface import flash_attn_with_kvcache as _flash_attn_with_kvcache
from vllm_flash_attn.flash_attn_interface import flash_attn_with_kvcache_aws as _flash_attn_with_kvcache_aws
from vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func

@torch.library.custom_op("vllm::flash_attn_varlen_func", mutates_args=[])
def flash_attn_varlen_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Optional[List[int]] = None,
        softcap: float = 0.0,
        alibi_slopes: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # custom op does not support tuple input
    real_window_size: Tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])
    return _flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=real_window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        block_table=block_table,
    )


@flash_attn_varlen_func.register_fake  # type: ignore
def _(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Optional[List[int]] = None,
        softcap: float = 0.0,
        alibi_slopes: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op("vllm::flash_attn_with_kvcache", mutates_args=[])
def flash_attn_with_kvcache(
        decode_query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_seqlens: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        alibi_slopes: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        return_softmax_lse: bool = False,
) -> torch.Tensor:
    result = _flash_attn_with_kvcache(
        decode_query,
        key_cache,
        value_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
        alibi_slopes=alibi_slopes,
        softcap=softcap,
        return_softmax_lse=return_softmax_lse,
    )
    # 如果 return_softmax_lse=True，result 是 (out, softmax_lse)，但自定义操作符只能返回单个tensor
    # 所以我们只返回 attention output
    return result[0] if return_softmax_lse else result

@torch.library.custom_op("vllm::flash_attn_with_kvcache_aws", mutates_args=[])
def flash_attn_with_kvcache_aws(
        decode_query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_seqlens: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        alibi_slopes: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        return_softmax_lse: bool = False,
) -> torch.Tensor:
    result = _flash_attn_with_kvcache_aws(
        decode_query,
        key_cache,
        value_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
        alibi_slopes=alibi_slopes,
        softcap=softcap,
        return_softmax_lse=return_softmax_lse,
    )
    # 如果 return_softmax_lse=True，result 是 (out, softmax_lse)，但自定义操作符只能返回单个tensor
    # 所以我们只返回 attention output
    # print(f"result: {result.shape}")
    return result

@flash_attn_with_kvcache.register_fake  # type: ignore
def _(
        decode_query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_seqlens: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        alibi_slopes: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        return_softmax_lse: bool = False,
) -> torch.Tensor:
    return torch.empty_like(decode_query)

@flash_attn_with_kvcache_aws.register_fake  # type: ignore
def _(
        decode_query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_seqlens: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        alibi_slopes: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        return_softmax_lse: bool = False,
) -> torch.Tensor:
    # 创建与decode_query相同dtype的output
    output = torch.empty_like(decode_query)
    block_aws_shape = output.shape[:-1] + (32,)
    block_aws = torch.empty(block_aws_shape, dtype=output.dtype, device=output.device)
    return  block_aws