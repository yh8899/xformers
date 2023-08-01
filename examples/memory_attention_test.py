import xformers.ops
from xformers.ops import fmha
import torch
import math
from einops import rearrange, repeat
import torch.nn.functional as F
from typing import Optional
import numpy as np
import sys

sys.path.append("/root/picasso/yangh/project/flash-attention/tests/")

from test_flash_attn import convert_flash_attn_S_to_softmax

seed = 1024 # or any of your favorite number 

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ref_attention_bmhk(q, k, v, attn_bias, scale=None) -> torch.Tensor:
    assert q.ndim == 4

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape(
            [t.shape[0] * t.shape[2], t.shape[1], t.shape[3]]
        )

    if isinstance(attn_bias, xformers.ops.AttentionBias):
        attn_bias = attn_bias.materialize(
            (q.shape[0], q.shape[2], q.shape[1], k.shape[1]),
            device=q.device,
            dtype=torch.float32,
        ).reshape([q.shape[0] * q.shape[2], q.shape[1], k.shape[1]])
    out = ref_attention(T(q), T(k), T(v), attn_bias, scale=scale)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))

def ref_attention(q, k, v, attn_bias=None, drop_mask=None, p=0.0, scale=None):
    if q.ndim == 4:
        assert p == 0.0
        return ref_attention_bmhk(q, k, v, attn_bias=attn_bias)
    # q = q.float()
    # k = k.float()
    # v = v.float()

    scale = scale if scale is not None else (1 / q.shape[-1] ** 0.5)
    q = q * scale

    attn = q @ k.transpose(-2, -1)
    if attn_bias is not None:
        if isinstance(attn_bias, xformers.ops.AttentionBias):
            # Always create in B,H,Mq,Mk format
            attn_bias_tensor = attn_bias.materialize(
                (q.shape[0], 1, q.shape[1], k.shape[1]),
                device=q.device,
                dtype=torch.float32,
            )
        else:
            attn_bias_tensor = attn_bias
        if attn_bias_tensor.ndim == 4:
            assert q.shape[0] == attn_bias_tensor.shape[0] * attn_bias_tensor.shape[1]
            attn_bias_tensor = attn_bias_tensor.reshape(
                [-1, *attn_bias_tensor.shape[2:]]
            )
        attn = attn + attn_bias_tensor.float()
    attn = attn.softmax(-1)
    if drop_mask is not None:
        attn = attn * (drop_mask / (1 - p))
    return attn @ v

def _get_drop_mask(op, batch_size, q_len, kv_len, p, device):
    if op == fmha.cutlass.FwOp:
        mask = torch.empty((batch_size, 1, q_len, kv_len), device=device)
        rand_uniform = torch.ops.xformers._cutlass_rand_uniform(p, mask)
        mask = (rand_uniform > p).to(torch.float32)
        mask = mask.reshape(batch_size, q_len, kv_len)
    else:
        mask = torch.empty((batch_size, q_len, kv_len), device=device)
        mask = torch.ops.xformers._temp_dropout(mask, p)

    return mask

def assert_allclose(
    out: Optional[torch.Tensor],
    ref: Optional[torch.Tensor],
    msg: str = "failed",
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> None:
    assert out is not None, f"{msg}: output Tensor is None"
    assert ref is not None, f"{msg}: reference Tensor is None"
    assert out.shape == ref.shape, f"Shape: {out.shape} (expected: {ref.shape})"
    if out.numel() == 0:
        return
    flatten_diff = ((out - ref).abs() - atol - ref.abs() * rtol).flatten()
    max_pos = flatten_diff.argmax()
    max_location = np.unravel_index(int(max_pos), out.shape)
    max_diff = flatten_diff[max_pos]
    num_different = torch.count_nonzero(flatten_diff > 0)
    percentage = num_different / flatten_diff.numel()
    del flatten_diff
    assert torch.allclose(out, ref, rtol=rtol, atol=atol), (
        f"{msg}: "
        f"out={out.flatten()[max_pos]} and ref={ref.flatten()[max_pos]} (diff={max_diff} > 0)"
        f" at {max_location} of shape {tuple(out.shape)} / atol={atol}, rtol={rtol}"
        f"/ total failing elements: {num_different}, percentage={percentage}"
    )

p = 0.2
batch_size = 1
q_len = 1024
k_len = 64
kv_len = 1024
device = "cuda"

query = torch.randn((batch_size, q_len, k_len)).to(device="cuda", dtype=torch.float16)
key = torch.randn((batch_size, kv_len, k_len)).to(device="cuda", dtype=torch.float16)
value = torch.randn((batch_size, kv_len, k_len)).to(device="cuda", dtype=torch.float16)

op = fmha.flash.FwOp

torch.manual_seed(seed)
out = xformers.ops.memory_efficient_attention(query, key, value, p=p, op=(op, None))

torch.manual_seed(seed)
out2 = xformers.ops.memory_efficient_attention(query, key, value, p=p, op=(op, None))

assert_allclose(out, out2, "dropout reproducibility")

torch.manual_seed(seed)
mask = _get_drop_mask(op, batch_size, q_len, kv_len, p, device)
att_ref = ref_attention(query, key, value, p=p)
assert_allclose(out, att_ref, atol=1e-1, rtol=1e-1), f"{(out - att_ref).abs().max()}"

# grad = torch.randn_like(att)
# att.backward(grad, retain_graph=True)

# print(grad)