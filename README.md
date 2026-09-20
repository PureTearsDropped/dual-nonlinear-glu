# dual-nonlinear-glu

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22851805.svg)](https://doi.org/10.5281/zenodo.22851805)
[![License: 0BSD](https://img.shields.io/badge/License-0BSD-blue.svg)](LICENSE)

Revisiting a design choice the field made in 2017 and has kept since: **the value
branch of a gated layer is linear**.

$$\text{SwiGLU:}\quad h \leftarrow h + W_d\Big(\mathrm{silu}(W_g h)\ \odot\ \underbrace{(W_u h)}_{\text{linear}}\Big)$$

Making it nonlinear too is not a new idea — it is the **Gated Tanh Unit**, and
Dauphin et al. (2017) measured it, argued against it on gradient grounds, and the
GLU → GEGLU → SwiGLU line descends from that rejection. See [Prior work](#prior-work).

On the synthetic benchmark here, with a residual and a $1/\sqrt{L}$ branch
initialisation, **that rejection does not reproduce**.

$$h \leftarrow h + W_d\Big(f_{a,b}(W_g h)\ \odot\ g_{A,B}(W_u h)\Big)$$

$$f_{a,b}(x)=\begin{cases}a x & x\ge0\\ b\tanh(ax/b) & x<0\end{cases}
\qquad
g_{A,B}(y)=\begin{cases}y/A & y\ge0\\ (B/A)\,\mathrm{asinh}(y/B) & y<0\end{cases}$$

Both of these are **linear above zero**, and measuring what that linear half
costs turned out to matter more than the choice of pair. Removing it from the
gate — a plain $b\tanh(ax/b)$ on both sides, which is the Gated Tanh Unit's own
gate — is worth 1.5–5.9× on top, on both teachers. The tables below carry every
combination; the legend for the letters is directly under them.

## The pieces

Every model below is the same block with two slots filled differently:

```
h <- h + W_d ( gate(W_g h)  *  value(W_u h) )

gate                                      value
  T   b tanh(a x / b)   bounded both        G   (B/A) asinh(y / B)   asinh both sides
  t   tanh(x)           T, nothing learned  g   y/A        (y>=0)    linear above
  C   clamp(a x, -b, b) T, no transcendental    (B/A)asinh(y/B) (y<0)
  F   max(a x, -b)      linear above        s   silu(y)
  f   a x       (x>=0)  linear above        i   y                    what SwiGLU uses
      b tanh(a x/b) (x<0)
  s   silu(x)           what SwiGLU uses
  i   x
```

`a, b, A, B` are learned per channel. **`F`, `f` and `g` are linear above zero;
`T` and `G` are not** — that distinction turns out to be the whole story.
Models are named gate-then-value: `fg` is the pair specified below, `si` is
SwiGLU, `ii` is bilinear.

## Results

8 seeds, all models at identical parameter counts. Normalised MSE, lower is better.

**Teacher: a product of two $\tanh$ projections, depth 32**

| gate | value | loss | |
|---|---|---|---|
|`T`|`G`|**0.00162 ± 0.00011**|9.7× below silu⊙silu|
|`t`|`G`|0.00277 ± 0.00012|gate fixed — nothing learned in it|
|`C`|`G`|0.00326 ± 0.00011|gate is a clamp — no transcendental|
|`T`|`g`|0.00483 ± 0.00020||
|`T`|`i`|0.00511 ± 0.00010|bounded gate alone, value left linear|
|`F`|`G`|0.00886 ± 0.00019||
|`f`|`G`|0.00954 ± 0.00019||
|`s`|`s`|0.01566 ± 0.00086|−0.2 σ vs `fg`|
|`f`|`g`|0.01575 ± 0.00064|the pair specified below|
|`g`|`g`|0.01697 ± 0.00056||
|`f`|`f`|0.01717 ± 0.00102||
|`F`|`g`|0.01848 ± 0.00048||
|`t`|`i`|0.01907 ± 0.00113||
|`s`|`g`|0.01969 ± 0.00084||
|`f`|`i`|0.04925 ± 0.00072||
|`s`|`i` — **SwiGLU**|0.07117 ± 0.00031||
|`i`|`i` — **bilinear**|0.08786 ± 0.00101||

**Teacher: an iterated map, no product at all, depth 16**

| gate | value | loss | |
|---|---|---|---|
|`T`|`g`|**0.01676 ± 0.00073**|3.1× below `fg`|
|`T`|`G`|0.01747 ± 0.00074|1.9 σ from the row above — a tie|
|`C`|`G`|0.01902 ± 0.00046|gate is a clamp — no transcendental, costs 9%|
|`t`|`G`|0.02117 ± 0.00057|gate fixed — nothing learned in it|
|`F`|`G`|0.02506 ± 0.00055||
|`f`|`G`|0.02607 ± 0.00074||
|`T`|`i`|0.02643 ± 0.00086|0.9 σ from `fG` — fixing either branch alone lands here|
|`F`|`g`|0.05071 ± 0.00103||
|`f`|`g`|0.05236 ± 0.00117|the pair specified below|
|`s`|`i` — **SwiGLU**|0.06952 ± 0.00112||
|`s`|`s`|0.07132 ± 0.00145||
|`t`|`i`|0.08321 ± 0.00108|**worse than SwiGLU** — see below|
|`i`|`i` — **bilinear**|0.09263 ± 0.00034||

## What the two tables together say

**The two teachers disagree about which ingredient matters.**

| | $\tanh$-product teacher | iterated-map teacher |
|---|---|---|
|making the value branch nonlinear at all|**4.5×**|**nothing** (silu⊙silu ≈ SwiGLU)|
|using $f$ and $g$ specifically|**nothing** (ties silu⊙silu)|**33%**|
|removing the **value** branch's linear half (`g`→`G`)|**1.65–2.98×**|0.96–2.02×|
|removing the **gate's** linear half (`f`→`T`)|**3.26–5.89×**|**1.49–3.12×**|

Neither of the first two survives both tasks. **The last row does**, at the
largest size, with no cell that fails: across two teachers that disagree about
everything else, what costs is the gate being linear above zero. The row above
it survives almost as well but has one cell (gate `T`, iterated map) where it
buys nothing.

This reverses an earlier reading of the same data. Comparing `f` against `F`
looked like "the gate is worth 4%" — but `f` and `F` differ only *below* zero,
so that comparison never touched the linear half. `T` does, and the gate turns
out to be the larger of the two effects.

**The two changes are redundant, not additive.** On the iterated map, fixing the
gate alone (`Ti`, 0.02643) and fixing the value alone (`fG`, 0.02607) land 0.9 σ
apart; doing both (`TG`, 0.01747) is a further 1.5×, not 3 × 1.5.

Bilinear (no nonlinearity anywhere) is last in both — the product alone is not
enough here, unlike in Dauphin et al., where bilinear beat a linear network by 40
perplexity points and lost to GLU by 20.

The 2017 gradient argument against dual nonlinearity — that both branches
contribute a downscaling factor and the product vanishes with depth — does not
reproduce at depth 32 here. **Why it does not is unresolved.** A natural guess
was that $f$ and $g$ have derivative exactly $a$ and $1/A$ on their whole
positive half, so nothing downscales there, unlike $\tanh'\cdot\sigma'$ in GTU.
The measurements above kill that guess: the best forms are `T` and `G`, which
have *no* linear half and therefore do downscale on both sides.

**Boundedness or the learned scale?** `T` learns a slope and a saturation
level; `silu` learns nothing. `t` — a plain `tanh(x)` — is the control. Since
`b tanh(a x/b) = b tanh((a/b) x)` and a per-channel gain is absorbable into
$W_d$, the only shape `T` can learn is the knee `a/b`; a trained `TG` puts it at
about 1.78 where `t` is pinned at 1.

The answer depends on the other branch. With value `G`, the fixed gate is
already most of the way: `tG` beats every linear-above gate, so boundedness
alone buys 2.5× and the learned knee adds 1.2×. With value `i`, the fixed gate
is a **disaster** — `ti` is worse than SwiGLU and nearly at bilinear.

The operating point explains it: a fixed `tanh` sits in its own linear region
(median argument 0.378), so when the gate is the sole nonlinearity the block
degenerates towards bilinear. The learned gates drive themselves into saturation
(median argument 3.2). The learned scale's job is placing the operating point,
and that only decides the outcome when nothing else is nonlinear. RESULTS 4.

**The saturated gate is not a sign function.** Swapping a trained `T` for
`b sign(x)` costs 3.70×; swapping it for `clamp(a x, -b, b)` costs 1.19×. The
linear ramp through zero matters; the smooth knee above it is cheap. Trained
with the clamp from the start, `CG` costs 9% against `TG` on the iterated map —
**so the gate needs no transcendental at all.**

**One thing these tables do not establish.** The margin over SwiGLU is far
larger on the $\tanh$-product teacher (13.9× for `Ti`) than on the iterated map
(2.6×); SwiGLU beats bounded gates on real language, so a 14× the other way on a
synthetic teacher says more about the teacher than about SwiGLU.

## Prior work

| | form | finding |
|---|---|---|
|**Gated Tanh Unit** — Gated PixelCNN (2016), named in Dauphin et al. (2017)|$\tanh(Wx)\odot\sigma(Vx)$|**both branches nonlinear**|
|**GLU** — [Dauphin et al. 2017](https://arxiv.org/abs/1612.08083)|$(Wx)\odot\sigma(Vx)$|**better than GTU**; the value branch is kept linear deliberately|
|bilinear — Mnih & Hinton (2007), measured in Dauphin et al.|$(Wx)\odot(Vx)$|beats linear by 40 ppl, loses to GLU by 20|
|**GEGLU / SwiGLU** — [Shazeer 2020](https://arxiv.org/abs/2002.05202)|$\mathrm{GELU}(xW)\odot xV$, $\mathrm{Swish}(xW)\odot xV$|value branch still linear; no both-nonlinear variant tested|

Dauphin et al.'s argument, verbatim:

> The gradient of the LSTM-style gating of which we dub gated tanh unit (GTU) is
> $\nabla[\tanh(X)\otimes\sigma(X)] = \tanh'(X)\nabla X\otimes\sigma(X)+\sigma'(X)\nabla X\otimes\tanh(X)$.
> Notice that it gradually vanishes as we stack layers because of the downscaling
> factors $\tanh'(X)$ and $\sigma'(X)$. In contrast, the gradient of the gated
> linear unit ... has a path $\nabla X\otimes\sigma(X)$ without downscaling for
> the activated gating units ... a multiplicative skip connection which helps
> gradients flow through the layers.

**Dual-nonlinear gating is not new. What is reported here is that the 2017
conclusion does not hold on this benchmark, and that the two teachers disagree
about why.**

## The potential

With $F'=f$ and $G'=g$, the block's output is a mixed second derivative:

$$\Phi(u,v)=F(u)G(v),\qquad \frac{\partial^2\Phi}{\partial u\,\partial v}=f(u)g(v)$$

verified to $1.6\times10^{-9}$. For this pair $F$ and $G$ are elementary
($\ln\cosh$, and an $\mathrm{asinh}$ integral); for silu the corresponding
antiderivative needs a dilogarithm. So $\Phi$ is computable at training time for
$f\odot g$ and not for SwiGLU, and never appears in the forward pass.

This is a *local* potential per gate. The network is **not** an energy-based model
and no stability guarantee follows.

## Limits

- one synthetic task family (teacher–student regression, width 64, input 32);
  **no language, no vision, no real data**
- two teachers give **opposite** accounts of which ingredient matters
- the 2017 gradient argument is contradicted here but not explained
- $\Phi$ has been verified as an identity and logged; **never used as a
  regulariser or objective**
- $a,b,A,B$ barely move under the `f`/`F` gates — $a$ ends near 0.82 from an
  initial 1.0 — but under the bounded `T` gate $a$ moves to about 0.27, so
  "the initialisation does the work" is a statement about the unbounded gates,
  not a general one
- `T`'s advantage is mostly the operating point, not boundedness as such: a
  fixed $\tanh$ with a nonlinear value branch keeps 2.5× of the 3.0×, but with a
  linear value branch it falls below SwiGLU
- the margin over SwiGLU is 13.9× on the $\tanh$-product teacher and 2.6× on the
  iterated map; SwiGLU beats bounded gates on real language, so the larger
  number is most likely a property of that teacher
- curvature-aware ternary rounding was tried and **failed** (RESULTS 7)
- no comparison against LayerNorm'd blocks, which is how real transformers avoid
  the instability that forced the $1/\sqrt{L}$ init here

## Files

| | |
|---|---|
|[`dnglu.py`](dnglu.py)|reference implementation|
|[`SPEC.md`](SPEC.md)|construction, initialisation, the potential, diagnostics|
|[`RESULTS.md`](RESULTS.md)|all measurements, what was ruled out, what failed|
|[`bench.py`](bench.py)|benchmark with every baseline above|
|[`APPROX.md`](APPROX.md)|asinh by polynomials only, for hardware without a transcendental unit|

```python
from dnglu import DualGLUNet
model = DualGLUNet(d_in=32, d_out=8, d_model=64, depth=32)
```

```bash
python bench.py --diagnose                     # is the task even usable?
python bench.py --depth 32 --seeds 8 --model fg,ss,swiglu,bilinear
python bench.py --depth 16 --seeds 8 --model fg,ss,swiglu,bilinear --teacher deep
```

## Two things that mattered more than the activation

**Initialise $W_d$ at $1/\sqrt{L}$.** Worth 30–41% to plain MLPs at depth 64;
SwiGLU reaches NaN by depth 16 without it. This dominated every activation choice
measured.

**Check that your benchmark needs depth.** Measure a linear baseline, a
one-nonlinearity baseline, and an ablation with the interior zeroed, before
comparing anything. An earlier version of this work spent a day on a task that a
network with *all weights zero* solved best. `bench.py --diagnose` runs the check;
SPEC.md section 6 has the rest.

## Related

[relu-tanh-asinh](https://github.com/PureTearsDropped/relu-tanh-asinh)
([10.5281/zenodo.22847921](https://doi.org/10.5281/zenodo.22847921)) applies the
same $f$ and $g$ **in series** — $f(C\,g(h))$ — rather than as a product. That
construction loses to SwiGLU. It also carries the third-derivative analysis of the
pair and the residual-initialisation result in more detail.

## Citing

```bibtex
@software{dual_nonlinear_glu,
  author  = {PureTearsDropped},
  title   = {dual-nonlinear-glu: revisiting the linear value branch of gated layers},
  year    = {2026},
  doi     = {10.5281/zenodo.22851805},
  url     = {https://github.com/PureTearsDropped/dual-nonlinear-glu}
}
```

**Note:** the archived v0.1.0 predates the prior-work section and the
silu⊙silu and bilinear baselines. Its framing overstates the novelty. `main` is
corrected.

## Licence

0BSD.
