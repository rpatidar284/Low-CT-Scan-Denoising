from __future__ import annotations
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

_CHUNK = 64

def selective_scan_fast(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    dtype_in = u.dtype
    u = u.float(); delta = delta.float()
    A = A.float(); B = B.float(); C = C.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, D_dim, L = u.shape
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    ys = []
    x = torch.zeros(batch, D_dim, A.shape[1], device=u.device, dtype=u.dtype)
    for start in range(0, L, _CHUNK):
        end = min(start + _CHUNK, L)
        chunk = end - start
        u_c = u[:, :, start:end]
        delta_c = delta[:, :, start:end]
        deltaA_c = torch.exp(torch.einsum("bdl,dn->bdln", delta_c, A))
        if not is_variable_B:
            dBu_c = torch.einsum("bdl,dn,bdl->bdln", delta_c, B, u_c)
        elif B.dim() == 3:
            dBu_c = torch.einsum("bdl,bnl,bdl->bdln", delta_c, B[:, :, start:end], u_c)
        else:
            B_exp = repeat(B[:, :, :, start:end], "b g n l -> b (g h) n l", h=D_dim // B.shape[1])
            dBu_c = torch.einsum("bdl,bdnl,bdl->bdln", delta_c, B_exp, u_c)
        y_chunk = []
        for i in range(chunk):
            x = deltaA_c[:, :, i, :] * x + dBu_c[:, :, i, :]
            if not is_variable_C:
                y_i = torch.einsum("bdn,dn->bd", x, C)
            elif C.dim() == 3:
                y_i = torch.einsum("bdn,bn->bd", x, C[:, :, start + i])
            else:
                C_exp = repeat(C[:, :, :, start + i], "b g n -> b (g h) n", h=D_dim // C.shape[1])
                y_i = torch.einsum("bdn,bdn->bd", x, C_exp)
            y_chunk.append(y_i)
        ys.append(torch.stack(y_chunk, dim=2))
    y = torch.cat(ys, dim=2)
    if D is not None:
        y = y + u * rearrange(D.float(), "d -> d 1")
    if z is not None:
        y = y * F.silu(z.float())
    y = y.to(dtype_in)
    return (y, x) if return_last_state else y
