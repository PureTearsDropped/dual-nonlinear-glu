# dual-nonlinear-glu

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22851806.svg)](https://doi.org/10.5281/zenodo.22851806)
[![License: 0BSD](https://img.shields.io/badge/License-0BSD-blue.svg)](LICENSE)

SwiGLU passes its **value** branch through unchanged:

$$h \leftarrow h + W_d\Big(\underbrace{\mathrm{silu}(W_g h)}_{\text{gate}}\ \odot\ \underbrace{(W_u h)}_{\text{value: identity}}\Big)$$

Putting a second nonlinearity there is worth **3.6–4.5×** on the benchmark here;
*which* nonlinearity is worth a further 8–9%.

$$h \leftarrow h + W_d\Big(f_{a,b}(W_g h)\ \odot\ g_{A,B}(W_u h)\Big)$$

$$f_{a,b}(x)=\begin{cases}a x & x\ge0\\ b\tanh(ax/b) & x<0\end{cases}
\qquad
g_{A,B}(y)=\begin{cases}y/A & y\ge0\\ (B/A)\,\mathrm{asinh}(y/B) & y<0\end{cases}$$

$f$ saturates below zero, $g$ grows logarithmically below zero; both are linear
above and $C^2$ but not $C^3$; $a,b,A,B$ are learned per channel.

## Result

Depth 32, 8 seeds, matched parameters (~136k), teacher–student regression.
Normalised MSE, lower is better:

| gate | value | loss |
|---|---|---|
|$f$|$g$|**0.01575 ± 0.00064**|
|$g$|$g$|0.01697 ± 0.00056 (+7.7%)|
|$f$|$f$|0.01717 ± 0.00102 (+9.0%)|
|silu|$g$|0.01969 ± 0.00084 (+25%)|
|$f$|identity|0.04925 ± 0.00072 (+213%)|
|silu|identity — **SwiGLU**|0.07117 ± 0.00031 (+352%)|

All six have exactly 136,392 parameters.

The ordering holds on a teacher with no product structure at all, where
$f\odot g$ beats $f\odot f$ by 8.1 σ and $g\odot g$ by 16 σ.

## Read this before quoting the headline number

**The 4.5× margin over SwiGLU is mostly a structural match with the teacher.**
On a product teacher, a gated student wins partly by resembling it. Removing the
product:

| teacher | $f\odot g$ vs SwiGLU |
|---|---|
|relu product|44×|
|$\tanh$ product|4.5×|
|**iterated map (no product)**|**1.33×**|

**33% is the honest number**, and it is one synthetic task family — no language,
no vision, no real data. Two competing explanations were tested and eliminated
(activation growth with depth; the extra per-channel parameters $a,b,A,B$), which
is what makes the remaining effect worth recording, not the size of the headline.

## The potential

With $F'=f$ and $G'=g$, the block's output is a mixed second derivative:

$$\Phi(u,v)=F(u)G(v),\qquad \frac{\partial^2\Phi}{\partial u\,\partial v}=f(u)g(v)$$

verified to $1.6\times10^{-9}$. For this pair $F$ and $G$ are elementary
($\ln\cosh$ and an $\mathrm{asinh}$ integral); for SwiGLU the corresponding $F$
needs a dilogarithm. So the output can be read as the **cross-curvature of a
potential in two features** rather than as "gate times value", and the potential
is computable at training time while never appearing in the forward pass.

This is a *local* potential per gate. The network is **not** an energy-based
model and no stability guarantee follows from it.

## Files

| | |
|---|---|
|[`dnglu.py`](dnglu.py)|reference implementation, ~120 lines|
|[`SPEC.md`](SPEC.md)|construction, initialisation, the potential, diagnostics|
|[`RESULTS.md`](RESULTS.md)|measurements, including what was ruled out and what is not shown|
|[`bench.py`](bench.py)|benchmark with all baselines and ablations|

```python
from dnglu import DualGLUNet
model = DualGLUNet(d_in=32, d_out=8, d_model=64, depth=32)
```

```bash
python bench.py --diagnose                      # is the task even usable?
python bench.py --depth 32 --seeds 8 --model fg,swiglu
python bench.py --depth 16 --seeds 8 --model fg,ff,gg,swiglu --teacher deep
```

## Two things that matter more than the activation

**Initialise $W_d$ at $1/\sqrt{L}$.** Worth 30–41% to plain MLPs at depth 64, and
SwiGLU reaches NaN by depth 16 without it (there is no normalisation in these
blocks). This dominated every activation choice measured.

**Check that your benchmark needs depth.** Measure a linear baseline, a
one-nonlinearity baseline, and an ablation with the interior zeroed, before
comparing anything. An earlier version of this work spent a day on a task that a
network with *all weights zero* solved best. `bench.py --diagnose` runs the
check; SPEC.md section 6 lists the rest.

## Related

[relu-tanh-asinh](https://github.com/PureTearsDropped/relu-tanh-asinh)
([10.5281/zenodo.22847921](https://doi.org/10.5281/zenodo.22847921)) applies the
same $f$ and $g$ **in series** — $f(C\,g(h))$ — rather than as a product. That
construction loses to SwiGLU; this one does not. The series version also carries
the third-derivative analysis of the pair and the residual-branch initialisation
result in more detail.

## Citing

```bibtex
@software{dual_nonlinear_glu,
  author  = {PureTearsDropped},
  title   = {dual-nonlinear-glu: making the value branch of a gated layer nonlinear},
  year    = {2026},
  doi     = {10.5281/zenodo.22851806},
  url     = {https://github.com/PureTearsDropped/dual-nonlinear-glu}
}
```

The DOI above always resolves to the latest version.

## Licence

0BSD.
