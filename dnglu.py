"""dual-nonlinear-glu: a gated layer whose *value* branch is nonlinear too.

SwiGLU computes

    h <- h + W_d ( silu(W_g h)  *  (W_u h) )
                   ^^^^^^^^^^^     ^^^^^^^
                   gate: silu      value: passed through unchanged

This replaces the identity on the value branch with a second nonlinearity:

    h <- h + W_d ( f_{a,b}(W_g h)  *  g_{A,B}(W_u h) )

with

    f_{a,b}(x) = a x                    (x >= 0)      saturates below to -b
                 b tanh(a x / b)        (x <  0)

    g_{A,B}(y) = y / A                  (y >= 0)      logarithmic below
                 (B/A) asinh(y / B)     (y <  0)

Both are C^2 but not C^3; a, b, A, B are learned per channel.

The product has an exact potential: with F' = f and G' = g,

    Phi(u, v) = F(u) G(v)    and    d^2 Phi / du dv = f(u) g(v) = the output

and for this pair F and G are elementary (log-cosh and an asinh integral),
which is not true for silu.

0BSD.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["f_ab", "g_AB", "F_ab", "G_AB", "DualGLU", "DualGLUNet"]

_pos = lambda r: F.softplus(r) + 1e-4
_inv = lambda v: math.log(math.expm1(max(v - 1e-4, 1e-6)))


def f_ab(x, a, b):
    """Gate branch. Linear above zero, saturating to -b below. f'(0) = a.

    Use torch.where: a ReLU rewrite gives the same values but a wrong
    gradient at exactly x = 0, because relu'(0) = 0 in PyTorch.
    """
    return torch.where(x >= 0, a * x, b * torch.tanh(a * x / b))


def g_AB(y, A, B):
    """Value branch. Linear above zero, logarithmic below. g'(0) = 1/A."""
    return torch.where(y >= 0, y / A, (B / A) * torch.asinh(y / B))


def F_ab(x, a, b):
    """Antiderivative of f, with F(0) = 0. Elementary."""
    return torch.where(x >= 0,
                       a * x * x / 2,
                       (b * b / a) * torch.log(torch.cosh((a * x / b).clamp(-30, 30))))


def G_AB(y, A, B):
    """Antiderivative of g, with G(0) = 0. Elementary."""
    return torch.where(y >= 0,
                       y * y / (2 * A),
                       (B / A) * (y * torch.asinh(y / B) - torch.sqrt(B * B + y * y) + B))


class DualGLU(nn.Module):
    """One residual block.

    d_model: residual width
    d_hidden: gate/value width. d_model // 3 matches a plain MLP block's
              parameter count (three matrices instead of one).
    depth:   total depth; W_down is initialised at 1/sqrt(depth), which
             matters a great deal and is not optional past depth ~16.
    """

    def __init__(self, d_model, d_hidden=None, depth=1, b0=0.1):
        super().__init__()
        m = d_hidden or max(1, d_model // 3)
        ia, ib = _inv(1.0), _inv(b0)
        self.Wg = nn.Parameter(torch.randn(m, d_model) * d_model ** -0.5)
        self.Wu = nn.Parameter(torch.randn(m, d_model) * d_model ** -0.5)
        self.Wd = nn.Parameter(torch.randn(d_model, m) * (m ** -0.5) * depth ** -0.5)
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.a = nn.Parameter(torch.full((m,), ia))
        self.b = nn.Parameter(torch.full((m,), ib))
        self.A = nn.Parameter(torch.full((m,), ia))
        self.B = nn.Parameter(torch.full((m,), ib))

    def branches(self, h):
        return h @ self.Wg.t(), h @ self.Wu.t()

    def potential(self, h):
        """Phi = F(u) G(v). Its mixed second derivative is the block's output.

        Training-time only; nothing here is needed at inference.
        """
        u, v = self.branches(h)
        return F_ab(u, _pos(self.a), _pos(self.b)) * G_AB(v, _pos(self.A), _pos(self.B))

    def forward(self, h):
        u, v = self.branches(h)
        z = f_ab(u, _pos(self.a), _pos(self.b)) * g_AB(v, _pos(self.A), _pos(self.B))
        return h + z @ self.Wd.t() + self.bias


class DualGLUNet(nn.Module):
    def __init__(self, d_in, d_out, d_model=64, depth=4, d_hidden=None, b0=0.1):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList(
            [DualGLU(d_model, d_hidden, depth=depth, b0=b0) for _ in range(depth)])
        self.out = nn.Linear(d_model, d_out)

    def forward(self, x):
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)
