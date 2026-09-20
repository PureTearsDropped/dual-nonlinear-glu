# dual-nonlinear-glu

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22851806.svg)](https://doi.org/10.5281/zenodo.22851806)
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

## Results

8 seeds, all models at identical parameter counts. Normalised MSE, lower is better.

**Teacher: a product of two $\tanh$ projections, depth 32**

| gate | value | loss | |
|---|---|---|---|
|silu|silu|**0.01566 ± 0.00086**|−0.2 σ vs $f\odot g$|
|$f$|$g$|**0.01575 ± 0.00064**|—|
|$g$|$g$|0.01697 ± 0.00056|+7.7% (4.0 σ)|
|$f$|$f$|0.01717 ± 0.00102|+9.0% (3.3 σ)|
|silu|$g$|0.01969 ± 0.00084|+25% (10.5 σ)|
|$f$|identity|0.04925 ± 0.00072|+213%|
|silu|identity — **SwiGLU**|0.07117 ± 0.00031|+352%|
|identity|identity — **bilinear**|0.08786 ± 0.00101|+458%|

**Teacher: an iterated map, no product at all, depth 16**

| gate | value | loss | |
|---|---|---|---|
|$f$|$g$|**0.05236 ± 0.00117**|—|
|silu|identity — **SwiGLU**|0.06952 ± 0.00112|+33% (30 σ)|
|silu|silu|0.07132 ± 0.00145|+36% (29 σ)|
|identity|identity — **bilinear**|0.09263 ± 0.00034|+77%|

## What the two tables together say

**The two teachers disagree about which ingredient matters.**

| | $\tanh$-product teacher | iterated-map teacher |
|---|---|---|
|making the value branch nonlinear at all|**4.5×**|**nothing** (silu⊙silu ≈ SwiGLU)|
|using $f$ and $g$ specifically|**nothing** (ties silu⊙silu)|**33%**|

So neither "dual nonlinearity is the point" nor "this pair is the point" survives
both tasks. What survives both is narrower: **$f\odot g$ is first in both**, and
bilinear (no nonlinearity anywhere) is last in both — the product alone is not
enough here, unlike in Dauphin et al., where bilinear beat a linear network by 40
perplexity points and lost to GLU by 20.

The 2017 gradient argument against dual nonlinearity — that both branches
contribute a downscaling factor and the product vanishes with depth — does not
reproduce at depth 32 here. **Why it does not is unresolved.** A natural guess is
that $f$ and $g$ have derivative exactly $a$ and $1/A$ on their entire positive
half, so nothing downscales there, unlike $\tanh'\cdot\sigma'$ in GTU. But
silu⊙silu has no such property and ties $f\odot g$ on one of the two teachers, so
that explanation is not supported either.

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
- $a,b,A,B$ barely move during training — the initialisation does the work, and
  it is not understood why the learning does not use those degrees of freedom
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
  doi     = {10.5281/zenodo.22851806},
  url     = {https://github.com/PureTearsDropped/dual-nonlinear-glu}
}
```

**Note:** the archived v0.1.0 predates the prior-work section and the
silu⊙silu and bilinear baselines. Its framing overstates the novelty. `main` is
corrected.

## Licence

0BSD.
