# Results

Normalised MSE (loss ÷ target variance) on teacher–student regression. Lower is
better. Every table gives the seed count; differences are reported in standard
errors, not just means.

Common setup: residual width 64, gate/value width 64/3, 5000 steps, Adam 3e-3
with cosine decay, batch 512, fresh data each step (no overfitting, no ceiling),
$W_d$ initialised at $L^{-1/2}$, per-seed gradient clipping.

## 1. Teachers

A gated student on a product teacher can win by *resembling the teacher*, so
three teachers are used. Nonlinearity alone is not enough — a teacher must also
need depth:

| teacher | $1-R^2$ | linear student | one nonlinearity | deep student | **depth value** |
|---|---|---|---|---|---|
|random 4-layer GELU MLP, $N(0,1)$ inputs|0.2227|0.00148|0.00127|0.00177|**0.72**|
|same, $N(0,3)$ inputs|0.4301|0.03814|0.02906|0.03456|**0.84**|
|random Fourier, $\sin$|0.9983|0.99249|0.99252|0.99251|**1.00**|
|**$\tanh$ product**|**0.9979**|**0.91782**|**0.30824**|**0.14574**|**2.12**|
|**relu product**|—|—|—|—|usable|
|**iterated map** (not a product)|—|—|—|—|usable|

depth value = one-nonlinearity ÷ deep. **Near 1 means the task is useless** — a
same-width shallow student already matches. The `sin` teacher is the other
failure: highly nonlinear, and no student learns anything.

$\tanh$ product: $y=W_o\big(\tanh(W_Ax)\odot\tanh(W_Bx)\big)$, $x\sim N(0,1)^{32}$,
$W_A,W_B\in\mathbb{R}^{128\times32}$, $y\in\mathbb{R}^8$.
Iterated map: $z\leftarrow\tanh(2z)+\tfrac12 q$ three times, then a linear read-out.

## 2. The main result: the value branch

Depth 32, 8 seeds, $\tanh$-product teacher, matched parameters (~136k):

| gate | value | loss | std |
|---|---|---|---|
|$f$|$g$|**0.01587**|0.00059|
|$f$|$f$|0.01694|0.00062|
|$g$|$g$|0.01737|0.00088|
|silu|$g$|0.01944|0.00089|
|$f$|**identity**|0.04829|0.00066|
|silu|**identity** (SwiGLU)|0.07095|0.00086|

**Making the value branch nonlinear is worth 3.7–4.5×. Which nonlinearity is
worth 6–9%.** Both are real; the sizes differ by an order of magnitude.

| change | effect |
|---|---|
|value: identity → nonlinear|**+267% to +347%**|
|among nonlinear pairs: choose $f\odot g$|**+6.7% (3.5 σ) to +9.4% (4.0 σ)**|

Contributions separate cleanly. Replacing the value identity with $g$ takes
0.07095 → 0.01944 (3.7×); replacing the gate silu with $f$ takes 0.07095 →
0.04829 (1.5×). **The value branch carries about twice the effect of the gate.**

## 3. The ranking survives a change of teacher

The $\tanh$-product teacher favours $\tanh$-shaped students — $f$'s negative side
*is* $b\tanh(ax/b)$. On a teacher with no product at all:

| | $\tanh$ product, depth 32 | iterated map, depth 16 |
|---|---|---|
|$f\odot g$|**0.01587**|**0.05189**|
|$f\odot f$|0.01694 (+6.7%)|0.05504 (+6.1%, 8.1 σ)|
|$g\odot g$|0.01737 (+9.4%)|0.05861 (+12.9%, 16 σ)|
|SwiGLU|0.07095 (+347%)|0.06894 (+33%)|

**The ordering and the relative gaps between the nonlinear pairs are nearly
identical.** The margin over SwiGLU, however, is strongly teacher-dependent:

| teacher | $f\odot g$ vs SwiGLU |
|---|---|
|relu product (depth 16)|**44×**|
|$\tanh$ product (depth 32)|**4.5×**|
|**iterated map, not a product (depth 16)**|**1.33×**|

**Most of the headline margin is the match between a gated student and a product
teacher.** What survives the match being removed is a 33% gap, not a 4.5× one.

## 4. Two explanations tested and eliminated

**Activation growth with depth** — no. Per-layer statistics at depths 4/16/32:

| depth | model | $h$ std, first → last layer | $\mathbb{E}\lvert fg\rvert$ | $\mathbb{E}\lvert\Phi\rvert$ |
|---|---|---|---|---|
|4|$f\odot g$|0.417 → 0.430|0.108 → 0.173|0.009 → 0.023|
|4|SwiGLU|0.443 → 0.455|0.114 → 0.165|—|
|16|$f\odot g$|0.425 → 0.417|0.076 → 0.083|0.008|
|16|SwiGLU|0.452 → 0.460|0.081 → 0.071|—|
|32|$f\odot g$|0.420 → 0.398|0.056 → 0.057|0.005|
|32|SwiGLU|0.464 → 0.467|0.075 → 0.070|—|

Both stay in 0.40–0.47 at every depth; $\Phi$ stays within 0.005–0.023. With the
$L^{-1/2}$ initialisation, neither swells. **The gap is not an activation-scale
effect.**

**Parameter asymmetry** — no. $f\odot g$ carries $4m$ per-channel parameters
($a,b,A,B$) that SwiGLU does not. Giving SwiGLU per-channel input and output
scales changes nothing:

| depth 32, 8 seeds | loss |
|---|---|
|SwiGLU|0.07095 ± 0.00086|
|SwiGLU + per-channel scales|0.07204 ± 0.00084|
|$f\odot g$|**0.01587 ± 0.00059**|

## 5. Depth

$\tanh$-product teacher, 8 seeds, $L^{-1/2}$ initialisation throughout:

| depth | $f\odot g$ | SwiGLU | MLP gelu |
|---|---|---|---|
|4|0.12666 ± 0.00124|0.13823 ± 0.00024|0.14067 ± 0.00109|
|32|**0.01587 ± 0.00059**|0.07095 ± 0.00086|0.11926 ± 0.00128|

**9% at depth 4 becomes 4.5× at depth 32.** Without the $L^{-1/2}$ init, SwiGLU
does not reach depth 16 at all (NaN), and plain MLPs lose 30–41% (SPEC 5).

## 6. What this does not show

- one task family (synthetic regression, width 64, input dim 32); **no language,
  no vision, no real data**
- the margin over SwiGLU is 1.33× once the student/teacher structural match is
  removed — the 4.5× headline is mostly that match
- $\Phi$ has been verified as an identity and logged as a diagnostic; **it has not
  been used as a regulariser or a training objective**
- $a,b,A,B$ barely move during training; the initialisation is doing the work and
  it is not understood why the learning does not use these degrees of freedom
- no comparison against LayerNorm'd blocks, which is how real transformers avoid
  the instability that forced the $L^{-1/2}$ init here

## 7. Reproducing

```bash
python bench.py --diagnose                       # baselines: is the task usable?
python bench.py --depth 32 --seeds 8 --model fg,swiglu,gelu
python bench.py --depth 32 --seeds 8 --model fg,ff,gg,sg,fi --teacher tanh
python bench.py --depth 16 --seeds 8 --model fg,ff,gg,swiglu --teacher deep
```

Seeds are stacked into a leading dimension, so 8 seeds cost little more than 1
(≈35× faster than looping). Gradient clipping is per-seed; a global norm would
couple the models.
