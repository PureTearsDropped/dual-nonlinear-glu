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

__all__ = ["f_ab", "g_AB", "F_ab", "G_AB", "f_fast", "g_fast", "f_sym", "g_sym",
           "asinh_pw", "g_pw", "DualGLU", "DualGLUNet"]

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


# ---------------------------------------------------------------- cheap forms
#
# Two different things live here; do not confuse them.
#
# (1) f_fast / g_fast are *approximations of the same functions*. Train with
#     f_ab / g_AB, then switch a trained model to these for inference.
#     f_fast is f_ab with tanh -> clamp, which collapses both branches into a
#     single max. g_fast is g_AB with asinh -> u/sqrt(1+u^2/3) on the negative
#     side; it keeps the positive side linear, as g_AB does. The g_fast swap
#     is a trade, not a free lunch: measured, it costs 4.4% loss. asinh_pw /
#     g_pw further down cost nothing measurable.
#
# (2) f_sym / g_sym are a *different design*, not an approximation. g_sym
#     applies asinh on both sides, so the positive half is compressed too.
#     They are branch-free and measured better than the exact pair when trained
#     with from the start (RESULTS 2). Do not swap them into a model trained
#     with g_AB -- the positive half would be compressed in a way it never saw.
#
# On GPU these kernels are bandwidth-bound, not ALU-bound: `torch.asinh` costs
# about as much as an add. A `where` reads three tensors and a condition where
# `asinh` reads one, and measures 1.2-2.4x an `asinh` depending on size (RTX
# 5090, f32). Branch-free is what makes these cheap -- by removing tensor
# traffic, not arithmetic.

def f_fast(x, a, b):
    """f_ab with tanh -> clamp. Exactly max(a*x, -b); one op, no branch."""
    return torch.maximum(a * x, -b)


def g_fast(y, A, B):
    """g_AB with asinh -> u/sqrt(1+u^2/3). Keeps the positive side linear.

    Not free: swapped into a trained depth-16 network it costs 4.4% loss
    (0.033041 -> 0.034510, up to 5.5% on a single seed). Use g_pw instead if
    the point is to avoid a transcendental -- it is the same function to
    within 1e-7 and reproduces the loss exactly.
    """
    return torch.where(y >= 0, y / A, y / (A * torch.sqrt(1 + y * y / (3 * B * B))))


f_sym = f_fast                       # the same max; f has no positive-side curve


def g_sym(y, A, B):
    """asinh on both sides. A different function from g_AB, not an
    approximation of it: the positive half is compressed as well."""
    return (B / A) * torch.asinh(y / B)


# --------------------------------------------------- piecewise (hardware only)
#
# asinh evaluated by polynomials alone, for targets with no transcendental
# unit. This is NOT a GPU optimisation: measured on an RTX 5090 (float32, 2M
# elements) the piecewise form costs 30x a `torch.asinh`, because these kernels
# are bandwidth-bound and splitting into segments only adds traffic. It is here
# so a trained model can be evaluated the way hardware would evaluate it.
#
# The segment boundaries come from the measured argument distribution of a
# trained network, not from the shape of asinh. After training, B shrinks to
# about 0.05-0.14, so the value branch sees |y/B| with median ~9.6 and 78% of
# all evaluations at u >= 4 -- where asinh is essentially a logarithm. An
# earlier design spent its degrees on [0,1], which carries 5.6% of the traffic.
# See APPROX.md.
#
# Coefficients: Remez, quantised to a 2^-24 grid.

_LN2 = math.log(2.0)

# asinh(u)/u on u^2 in [0,1], degree 4, relative error 8.1e-6.
_P_ODD = (0.9999919533729553, -0.16622233390808105, 0.07097268104553223,
          -0.03091973066329956, 0.007558107376098633)
# ln(m)/(2z) on z^2 in [0,1/9], degree 3, relative error 1.8e-7.
# The coefficients sit near 1, 1/3, 1/5, 1/7 because this is the atanh series.
_P_LOG = (0.9999998211860657, 0.3333800435066223, 0.19794732332229614,
          0.1711302399635315)


def _horner(c, t):
    y = torch.full_like(t, c[-1])
    for k in range(len(c) - 2, -1, -1):
        y = y * t + c[k]
    return y


def _ln_pw(s):
    """ln s for s > 0, from the exponent and the mantissa.

    s = 2^e m with m in [1,2), so ln s = e ln2 + ln m, and
    ln m = 2 z P(z^2) with z = (m-1)/(m+1) in [0,1/3].
    """
    m, e = torch.frexp(s)                  # frexp gives m in [0.5,1)
    m = m * 2
    z = (m - 1) / (m + 1)
    return (e - 1).to(s.dtype) * _LN2 + 2 * z * _horner(_P_LOG, z * z)


def asinh_pw(v):
    """asinh by polynomials only.

        u < 1          u P(u^2)
        1 <= u < 2^12  ln(u + sqrt(1 + u^2))    an identity, no truncation
        u >= 2^12      ln2 + ln u               asinh(u) - ln(2u) ~ 1/(4u^2)

    The upper split exists to keep u*u from overflowing in float32, not for
    accuracy; writing ln2 + ln u rather than ln(2u) avoids forming 2u for the
    same reason. With the degree-4 odd segment shipped here, the maximum
    absolute error is 7.1e-6 on the distribution a trained network produces and
    8.9e-6 across the whole float32 range; the degree-5 coefficients in
    APPROX.md bring the first to 9.9e-7, which does not change any loss we
    measured. No NaN or inf anywhere in float32; -0.0 keeps its sign.
    """
    u = v.abs()
    uc = u.clamp(max=1.0)
    small = uc * _horner(_P_ODD, uc * uc)
    um = u.clamp(max=4096.0)
    mid = _ln_pw(um + torch.sqrt(1 + um * um))
    big = _LN2 + _ln_pw(u.clamp(min=1.0))
    mag = torch.where(u < 1, small, torch.where(u < 4096.0, mid, big))
    return torch.copysign(mag, v)     # sign() would turn -0.0 into +0.0


def g_pw(y, A, B):
    """g_sym with asinh replaced by asinh_pw. On a trained depth-16 network
    this reproduces the loss of g_sym to seven figures (relative deviation
    1.1e-7 per seed), with either the degree-4 or the degree-5 odd segment."""
    return (B / A) * asinh_pw(y / B)


class DualGLU(nn.Module):
    """One residual block.

    d_model: residual width
    d_hidden: gate/value width. d_model // 3 matches a plain MLP block's
              parameter count (three matrices instead of one).
    depth:   total depth; W_down is initialised at 1/sqrt(depth), which
             matters a great deal and is not optional past depth ~16.
    """

    def __init__(self, d_model, d_hidden=None, depth=1, b0=0.1, mode="sym"):
        """mode: "sym"   branch-free, asinh both sides -- best measured, train with it
                 "exact" the C^2 pair, positive side linear
                 "fast"  approximations of "exact", for inference on a model
                         trained with mode="exact" (see set_mode)
                 "pw"    "sym" with asinh by polynomials only -- what hardware
                         without a transcendental unit would compute. Slower
                         than "sym" on a GPU; see asinh_pw.
        """
        super().__init__()
        self.mode = mode
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

    def set_mode(self, mode):
        """Switch the pair used at run time.

        "exact" -> "fast" is a safe inference-time swap: same functions, cheaper
        evaluation. Any swap involving "sym" changes the function itself.
        """
        self.mode = mode
        return self

    def _pair(self):
        return {"exact": (f_ab, g_AB), "fast": (f_fast, g_fast),
                "sym": (f_sym, g_sym), "pw": (f_sym, g_pw)}[self.mode]

    def potential(self, h):
        """Phi = F(u) G(v). Its mixed second derivative is the block's output.

        Defined for mode="exact" only; F_ab and G_AB are the antiderivatives of
        that pair. Training-time only -- nothing here runs at inference.
        """
        u, v = self.branches(h)
        return F_ab(u, _pos(self.a), _pos(self.b)) * G_AB(v, _pos(self.A), _pos(self.B))

    def forward(self, h):
        f, g = self._pair()
        u, v = self.branches(h)
        z = f(u, _pos(self.a), _pos(self.b)) * g(v, _pos(self.A), _pos(self.B))
        return h + z @ self.Wd.t() + self.bias


class DualGLUNet(nn.Module):
    def __init__(self, d_in, d_out, d_model=64, depth=4, d_hidden=None, b0=0.1,
                 mode="sym"):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList(
            [DualGLU(d_model, d_hidden, depth=depth, b0=b0, mode=mode)
             for _ in range(depth)])
        self.out = nn.Linear(d_model, d_out)

    def set_mode(self, mode):
        for blk in self.blocks: blk.set_mode(mode)
        return self

    def forward(self, x):
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)
