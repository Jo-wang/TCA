import math
import numpy as np
from typing import Callable, Tuple
import warnings
import torch
from torch import Tensor
from typing import Optional, Tuple
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import softmax, dropout, linear, _mha_shape_check, _in_projection_packed, pad, _in_projection
import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    DType = int

from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)


def _scaled_dot_product_attention_ToME(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    size: torch.Tensor = None
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.

    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.

    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.

        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    if attn_mask is not None:
        attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
    else:
        attn = torch.bmm(q, k.transpose(-2, -1))

    # Apply proportional attention
    if size is not None:
        attn = attn + size.log()[:, None, None, :, 0]

    attn = softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def multi_head_attention_forward_ToME(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    size: torch.Tensor = None
) -> Tuple[Tensor, Optional[Tensor]]:
    
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward_ToME,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = _scaled_dot_product_attention_ToME(q, k, v, attn_mask, dropout_p, size=size)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None


class MultiheadAttention_ToME(nn.MultiheadAttention):
    r"""
    inherit and modify the attention = attention + log(s)
    """

    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True, size: torch.Tensor = None) -> Tuple[Tensor, Optional[Tensor]]:
        
        is_batched = query.dim() == 3
        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        elif query.is_nested and key_padding_mask is not None:
            why_not_fast_path = "key_padding_mask is not supported with NestedTensor input"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    key_padding_mask if key_padding_mask is not None else attn_mask,
                    need_weights,
                    average_attn_weights)
        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward_ToME(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights, size=size)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward_ToME(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights, size=size)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
  

    def merge_masks(self, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                    query: Tensor) -> Tuple[Optional[Tensor], Optional[int]]:
        r"""
        Determine mask type and combine masks if necessary. If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2

            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type

def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        x = x.permute(1, 0, 2) # shape -> B, N, D
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def kth_bipartite_soft_matching(
    metric: torch.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // z.

    Input size is [batch, tokens, channels].
    z indicates the stride for the first set.
    z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(
    metric: torch.Tensor, r: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].

    This will reduce the number of tokens by r.
    """
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source


#################### ATS helper ######################
from timm.models.layers import DropPath
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=True,
        qk_scale=None
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.project = nn.Linear(dim, dim)

        self.n_segment = 8

    @staticmethod
    def softmax_with_policy(attn, policy, eps=1e-6):
        B, N, _ = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(
            1, 1, N, N
        )
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, policy, sampler):

        N, B, C = x.shape
        # x = x.transpose(1, 0)

        qkv= self.qkv(x) #1,197,3,12,64
        qkv_ = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads,int(qkv.shape[2]/(3*self.num_heads)))

        # Permute to match target shape (1, 197, 3, 12, 64)
        # Explanation: 
        # - Move the batch size to the first dimension
        # - Sequence length comes second
        # - 3 projections (Q, K, V) remain third
        # - Heads and head dimensions follow
        qkv_ = qkv_.permute(1, 0, 2, 3, 4)
        qkv_ = qkv_.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        ) # 3,1,12,197,64

        q, k, v = qkv_[0], qkv_[1], qkv_[2] # 1,12,197,64

        attn = (q @ k.transpose(-2, -1)) * self.scale # 1,12,197,197

        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) #1,197,12,64->1,197,768
        x = x.transpose(1,0)
        x = self.project(x)

        return x
    
class AdaptiveTokenSampler(Attention):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=True,
        qk_scale=None,
        drop_tokens=False
    ):
        super(AdaptiveTokenSampler, self).__init__(
            dim,
            num_heads,
            qkv_bias,
            qk_scale
        )

        self.out_zero_mask = nn.Parameter(torch.zeros(1, dim, dtype=torch.float16), requires_grad=False) #1,768
        self.drop_tokens = drop_tokens

    @staticmethod
    def get_unique_indices(indices: Tensor, max_value: int) -> Tensor:
        """
        :param indices: indices of the tokens to be sampled
        :param max_value: maximum number of the tokens to be sampled
        :return: unique indices of the tokens to be sampled
        """
        sorted_indices = torch.sort(indices, dim=1)[0]

        shift_left = F.pad(sorted_indices[:, 1:], (0, 1), value=1.0)
        unique_indices = torch.where(
            (shift_left - sorted_indices) == 0,
            max_value * torch.ones_like(indices),
            sorted_indices,
        )
        unique_indices = torch.sort(unique_indices, dim=1)[0]
        return unique_indices

    @staticmethod
    def create_ys(normalized_cdf: Tensor, n_tokens: int) -> Tensor:
        """
        Sample uniformly from y-axis.
        """

        B = normalized_cdf.shape[0]
        # epsilon = (1 / (n_tokens - 1)) / 2
        ys = (
            torch.linspace(
                start=0,
                end=1.0,
                steps=n_tokens - 1,
                device=normalized_cdf.device,
            )
            .unsqueeze(0)
            .repeat(B, 1)
        )
        ys_start = (
            torch.min(normalized_cdf + (normalized_cdf == 0).float() * 1e8, dim=1)[0]
            .unsqueeze(-1)
            .expand_as(ys)
        )
        steps = (
            torch.range(0, n_tokens - 2, device=normalized_cdf.device)
            .unsqueeze(0)
            .expand_as(ys_start)
        )
        ys = ys_start + (((ys * (n_tokens - 2)) - ys_start * steps) / (n_tokens - 2))

        return ys

    @staticmethod
    def score_assignment_step(attn: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Token Score Assignment Step.
        :param attn: attention matrix
        :param v: values
        :return: sorted significance scores and their corresponding indices
        """

        B, H, _, _ = attn.shape
        C = v.shape[3] * H
        v_norm = torch.linalg.norm(
            v.transpose(1, 2).reshape(B, attn.shape[2], C), ord=2, dim=2
        )  # value norm of size [B x T]
        significance_score = attn[:, :, 0].sum(
            dim=1
        )  # attention weights of CLS token of size [B x T]
        significance_score = significance_score * v_norm  # [B x T]
        significance_score = significance_score[:, 1:]  # [B x T-1]

        significance_score = significance_score / significance_score.sum(
            dim=1, keepdim=True
        )  # [B x T-1]
        sorted_scores, sorted_indices = torch.sort(
            significance_score, descending=False, dim=1
        )

        return sorted_scores, sorted_indices

    def inverse_transform_sampling(
        self,
        sorted_scores: Tensor,
        sorted_indices: Tensor,
        attn: Tensor,
        n_tokens: int,
        raw_x: Tensor,
        n_ref_tokens: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample tokens based on their significance scores.
        """
        raw_x = raw_x.transpose(1, 0)
        B, N, C = raw_x.shape

        cdf = torch.cumsum(sorted_scores, dim=1)  # [B x T-1]

        normalized_cdf = (  # normalized cdf
            cdf - cdf.min(dim=1)[0].unsqueeze(dim=1)
        ) / ((cdf.max(dim=1)[0] - cdf.min(dim=1)[0]) / 1.0).unsqueeze(dim=1)

        ys = self.create_ys(normalized_cdf, n_ref_tokens).unsqueeze(
            dim=2
        )  # sampled values from y-axis of size [B, n-1, 1]
        normalized_cdf = normalized_cdf.unsqueeze(dim=1)  # [B, 1, N - 1]

        # expanded_ys = torch.Tensor.expand(ys, (B, n_tokens - 1, N - 1))
        expanded_ys = torch.Tensor.expand(ys, (B, ys.shape[1], ys.shape[1])) # 1, n-1, n-1
        diff_tokens = ys.shape[1] - (N - 1)
        tokens_to_pick_ind = torch.min(
            torch.abs(expanded_ys - F.pad(normalized_cdf, (diff_tokens, 0))),
            dim=2,
        )[
            1
        ]  # [B x n-1]

        # Offsetting token indices
        tokens_to_pick_ind = tokens_to_pick_ind - diff_tokens

        # Sort attention matrix and add CLS weights.
        attn_sorted = torch.gather(
            attn[:, :, 1:],
            2,
            sorted_indices.unsqueeze(1)
            .unsqueeze(-1)
            .expand(B, self.num_heads, N - 1, N),
        )  # [B x h x T-1 x T]

        attn_tmp = F.pad(attn_sorted, (0, 0, 0, 1), value=0.0)  # [B x h x T x T]

        # # Sort tokens and add CLS token.
        raw_x_tmp = torch.gather(
            raw_x[:, 1:], 1, sorted_indices.unsqueeze(-1).expand(B, N - 1, C)
        )
        raw_x_tmp = F.pad(raw_x_tmp, (0, 0, 0, 1), value=0.0)  # [B x n x C]

        unique_indices = self.get_unique_indices(
            indices=tokens_to_pick_ind, max_value=N - 1
        )[:, : N - 1]

        # Prune the attention matrix and input tokens.
        attn_tmp = torch.gather(
            attn_tmp,
            2,
            unique_indices.unsqueeze(1)
            .unsqueeze(3)
            .expand(B, self.num_heads, n_tokens - 1, N),
        ) # B, H, n-1, n
        raw_x_tmp = torch.gather(
            raw_x_tmp, 1, unique_indices.unsqueeze(2).expand(B, n_tokens - 1, C)
        ) # B, n-1, C

        attn_tmp = torch.cat([attn[:, :, 0:1], attn_tmp], dim=2)
        raw_x_tmp = torch.cat([raw_x[:, 0:1], raw_x_tmp], dim=1)

        policy = (unique_indices != (N - 1)).unsqueeze(-1).float()
        policy = F.pad(policy, (0, 0, 1, 0), value=1.0)
        selected_x = raw_x_tmp
        attn = attn_tmp

        sampler = torch.nonzero(policy)

        return selected_x, attn, policy, sampler
    
    def forward(
        self,
        x: Tensor,
        policy: Tensor,
        sampler: Tensor,
        n_tokens: float,
        raw_x: Tensor,
        n_ref_tokens: int,
    ):
        N, B, C = x.shape
        # x = x.transpose(1, 0)

        if isinstance(N, Tensor):
            N = N.cpu().item()

        if n_tokens > N:  # Number of tokens to be sampled can't be larger than N.
            n_tokens = N
        if n_tokens <= 1.0:  # When n_tokens is a ratio.
            n_tokens = n_tokens * N
        if n_tokens < 8:  # Number of tokens to be sampled can't be less than 8.
            n_tokens = 8

        # n_tokens = round(n_tokens)
        if N < n_tokens:
            n_tokens = N

        qkv = self.qkv(x)
        qkv_ = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads,int(qkv.shape[2]/(3*self.num_heads)))

        # Permute to match target shape (1, 197, 3, 12, 64)
        # Explanation: 
        # - Move the batch size to the first dimension
        # - Sequence length comes second
        # - 3 projections (Q, K, V) remain third
        # - Heads and head dimensions follow
        qkv_ = qkv_.permute(1, 0, 2, 3, 4)
        qkv_ = qkv_.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        ) # 3,1,12,197,64
        # qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
        #     2, 0, 3, 1, 4
        # )
        
        # TODO: need to init a policy matrics for this calculation
        qkv_ = qkv_ * policy.unsqueeze(0).unsqueeze(2)  # Get rid of previously removed tokens.
        q, k, v = (
            qkv_[0],
            qkv_[1],
            qkv_[2],
        )

        attn_no_softmax = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax_with_policy(attn_no_softmax, policy)  # [B x H x T x T]

        # --------------------------
        # Token Score Assignment
        # --------------------------

        sorted_scores, sorted_indices = self.score_assignment_step(attn, v)

        # --------------------------
        # Inverse Transform Sampling
        # --------------------------

        selected_x, attn, policy, sampler = self.inverse_transform_sampling(
            sorted_scores, sorted_indices, attn, n_tokens, raw_x, n_ref_tokens
        )

        x = (attn @ v).transpose(1, 2).reshape(B, attn.shape[2], C) # 1, 197, 768

        # Pruning
        if self.drop_tokens:
            out_mask_size = policy.sum(1).max().int() # 192

            sampler_out = sampler[:, 0] * out_mask_size + sampler[:, 1]
            sampler = sampler[:, 0] * n_tokens + sampler[:, 1]
            sampler_input = sampler.unsqueeze(-1).expand(-1, C) # 192, 768
            sampler_output = sampler_out.unsqueeze(-1).expand(-1, C) # 192, 768
            flatten_x = x.reshape(-1, C) #197,768
            flatten_selected_x = selected_x.reshape(-1, C) #197,768

            x_prunned = torch.gather(flatten_x, 0, sampler_input) #192,768
            selected_x_prunned = torch.gather(flatten_selected_x, 0, sampler_input) #192,768

            out_zero_mask = self.out_zero_mask.expand(B * out_mask_size, -1) #192,768

            x = out_zero_mask.scatter(
                0, sampler_output, x_prunned, reduce="add"
            ).reshape((B, out_mask_size, C)) # 1,192,768
            selected_x = out_zero_mask.scatter(
                0, sampler_output, selected_x_prunned, reduce="add"
            ).reshape((B, out_mask_size, C))# 1,192,768

            policy = (
                out_zero_mask[:, 0]
                .scatter(0, sampler_out, 1, reduce="add")
                .reshape(B, out_mask_size, 1)
            )

        x = x.transpose(1, 0)
        x = self.project(x)
        x = x.transpose(1, 0)
        x = x * policy
        # selected_x = selected_x.transpose(1, 0)
        return x, selected_x, policy, sampler
    
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        insert_control_point=False,
    ):
        super().__init__()
        self.insert_control_point = insert_control_point
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, policy: Tensor = None, sampler: Tensor = None) -> Tensor:
        x = x + self.attn(x=self.norm1(x), policy=policy, sampler=sampler)
        
        if policy is not None:
            x = x * policy
        out = self.mlp(x=self.norm2(x), policy=policy, sampler=sampler)
        if policy is not None:
            x = x * policy
        return x
    
    
################### Ours #################################
def token_clustering(token, num_center):
    from sklearn.cluster import kmeans_plusplus
   
    token_ = token.permute(1, 0, 2).squeeze() 
    center, _ = kmeans_plusplus(token_.cpu().numpy(), n_clusters=num_center, random_state=1)
    center = torch.tensor(center).to(token_.device)
    distances = torch.cdist(token_.float(), center.float())  
    cluster_assignment = torch.argmin(distances, dim=1)     
    weighted_tokens = torch.zeros(num_center, token_.shape[1], device=token_.device)
    
 
    for i in range(num_center):
        cluster_tokens = token_[cluster_assignment == i]  

        if len(cluster_tokens) == 0:
            continue
        
        cluster_center = center[i]  

      
        token_distances = torch.norm(cluster_tokens - cluster_center, dim=1) 

      
        epsilon = 1e-8
        token_distances[token_distances == 0] = epsilon
        weights = 1 / token_distances
        weights[token_distances == epsilon] = 1.0 
        weights = weights / weights.sum()  

     
        weighted_tokens[i] = torch.sum(cluster_tokens * weights.unsqueeze(-1), dim=0)  

def k_center_greedy(token, num_cluster):
    """
    X: numpy array of shape (n, d), where n is the number of tokens, and d is the dimension of embeddings.
    k: the number of clusters (or centers) to select.

    Returns:
    centers: List of indices of selected centers.
    """
    n = token.shape[0]
   
    center = [torch.randint(n, (1,)).item()]
   
    min_distances = torch.norm(token - token[center[0]], dim=1)

  
    for _ in range(num_cluster - 1):
       
        next_center = torch.argmax(min_distances).item()
        center.append(next_center)

       
        new_distances = torch.norm(token - token[next_center], dim=1)
        min_distances = torch.minimum(min_distances, new_distances)
        
    center_tokens = torch.stack([token[i] for i in center])
    return center_tokens

def coreset_averaging(token, num_centers):
    batch_size = token.shape[1]  
    num_tokens = token.shape[0]  
    feature_dim = token.shape[2]  
    weighted_tokens = torch.zeros(num_centers, batch_size, feature_dim, device=token.device)
    for b in range(batch_size):
        
        token_batch = token[:, b, :].squeeze(1)  

      
        centers = k_center_greedy(token_batch, num_cluster=num_centers)

       
        distances = torch.cdist(token_batch.float(), centers.float())

        cluster_assignment = torch.argmin(distances, dim=1)

      
        for i in range(num_centers):
           
            cluster_tokens = token_batch[cluster_assignment == i]

            if cluster_tokens.shape[0] == 0:  
                continue
         
            cluster_center = centers[i]

            token_distances = torch.norm(cluster_tokens - cluster_center, dim=1)

           
            epsilon = 1e-8
            token_distances[token_distances == 0] = epsilon  
            weights = 1 / token_distances
            weights[token_distances == epsilon] = 1.0  
            weights = weights / weights.sum()  

            weighted_tokens[i, b] = torch.sum(cluster_tokens * weights.unsqueeze(-1), dim=0)

    return weighted_tokens  