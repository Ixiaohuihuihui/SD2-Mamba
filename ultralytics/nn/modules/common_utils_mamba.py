import torch
import torch.nn.functional as F
import math
from functools import partial
from typing import Callable, Any

import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import DropPath

from typing import Optional

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
try:
    import selective_scan_cuda_core
    import selective_scan_cuda_oflex
    import selective_scan_cuda_ndstate
    import selective_scan_cuda_nrow
    import selective_scan_cuda
except:
    pass

try:
    "sscore acts the same as mamba_ssm"
    import selective_scan_cuda_core
except Exception as e:
    print(e, flush=True)
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    import selective_scan_cuda
    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


class LayerNorm2d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# Cross Scan
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs, None, None


# cross selective scan ===============================
class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        ctx.delta_softplus = delta_softplus
        ctx.backnrows = backnrows
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def cross_selective_scan_with_dsm(
    x: torch.Tensor,
    x_proj_weight: torch.Tensor,
    x_proj_bias: Optional[torch.Tensor],
    dt_projs_weight: torch.Tensor,
    dt_projs_bias: torch.Tensor,
    A_logs: torch.Tensor,
    Ds: torch.Tensor,
    out_norm: Optional[torch.nn.Module] = None,
    out_norm_shape="v0",
    delta_softplus=True,
    force_fp32=False,
    nrows=1,
    backnrows=1,
    ssoflex=True,
    SelectiveScan=SelectiveScanCore,
    # DSM specific parameters
    semantic_density_map: Optional[torch.Tensor] = None,
    dsm: Optional[DensityAwareModulation] = None,
):
    """
    Cross selective scan with DSM support
    Only modulates delta and B parameters before selective scan
    """
    if semantic_density_map is None or dsm is None:
        return cross_selective_scan(
            x, x_proj_weight, x_proj_bias, dt_projs_weight, dt_projs_bias,
            A_logs, Ds, out_norm, out_norm_shape, delta_softplus, force_fp32,
            nrows, backnrows, ssoflex, SelectiveScan
        )
    
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    xs = CrossScan.apply(x)
    
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    
    # HiPPO matrix
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    dts_for_dsm = dts.view(B, K, D, L).permute(0, 1, 2, 3)  # [B, K, D, L]
    Bs_for_dsm = Bs.view(B, K, N, L).permute(0, 1, 2, 3)    # [B, K, N, L]
    
    dts_modulated, Bs_modulated = dsm(semantic_density_map, dts_for_dsm, Bs_for_dsm)
    
    dts = dts_modulated.permute(0, 1, 2, 3).contiguous().view(B, -1, L)
    Bs = Bs_modulated.permute(0, 1, 2, 3).contiguous()  # 保持[B, K, N, L]格式

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)

    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm is not None:
        if out_norm_shape in ["v1"]:  # (B, C, H, W)
            y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
        else:  # (B, L, C)
            y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
            y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if True else y)


class DensityAwareModulation(nn.Module):
    """
    Density-aware Sequence Modulation (DSM) module
    """
    def __init__(self, modulation_scale=0.3):
        super().__init__()
        self.modulation_scale = modulation_scale
        
        self.local_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.delta_mapper = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.B_mapper = nn.Conv1d(1, 1, kernel_size=1, bias=False)
    
    def forward(self, semantic_density_map, delta, B):

        batch_size, C, H, W = semantic_density_map.shape
        L = H * W
        
        local_signal = self.local_conv(semantic_density_map)  # [B, 1, H, W]
        global_signal = self.global_pool(semantic_density_map)  # [B, 1, 1, 1]
        global_signal = global_signal.expand(batch_size, 1, H, W)  # [B, 1, H, W]
        
        delta_modulation = local_signal + global_signal  # [B, 1, H, W]
        B_modulation = local_signal + global_signal  # [B, 1, H, W]
        
        delta_modulation = delta_modulation.view(batch_size, 1, L)  # [B, 1, L]
        B_modulation = B_modulation.view(batch_size, 1, L)  # [B, 1, L]
        
        delta_modulation = self.delta_mapper(delta_modulation)  # [B, 1, L]
        B_modulation = self.B_mapper(B_modulation)  # [B, 1, L]
        
        K_delta, D = delta.shape[1], delta.shape[2]
        K_B, N = B.shape[1], B.shape[2]
        
        delta_modulation = delta_modulation.unsqueeze(1).expand(batch_size, K_delta, D, L)  # [B, K, D, L]
        alpha = 1 + self.modulation_scale * torch.tanh(delta_modulation)  # α = 1 + λ ⊙ tanh(f_Δ(tilde{Δ}))
        
        B_modulation = B_modulation.unsqueeze(1).expand(batch_size, K_B, N, L)  # [B, K, N, L]
        beta = 1 + self.modulation_scale * torch.tanh(B_modulation)   # β = 1 + λ ⊙ tanh(f_B(tilde{B}))
        
        # Δ_φ = Δ ⊙ α, B_φ = B ⊙ β
        delta_modulated = delta * alpha
        B_modulated = B * beta
        
        return delta_modulated, B_modulated


