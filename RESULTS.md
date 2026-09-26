# Results

Normalised MSE (loss ÷ target variance) on teacher–student regression. Lower is
better. Every table gives the seed count; differences are reported in standard
errors, not just means.

Common setup: residual width 64, gate/value width 64/3, 5000 steps, Adam 3e-3
with cosine decay, batch 512, fresh data each step (no overfitting, no ceiling),
$W_d$ initialised at $L^{-1/2}$, per-seed gradient clipping.

## Notation

Every model here is a gated residual block,

```
h <- h + W_d ( gate(W_g h)  *  value(W_u h) )
```

and they differ only in which two functions fill those slots. `a, b, A, B` are
learned per channel and kept positive by softplus.

| gate | | |
|---|---|---|
|`T`|`b tanh(a x / b)`|tanh on both sides — bounded above **and** below|
|`F`|`max(a x, -b)`|linear above, clamped to `-b` below|
|`f`|`a x` for `x>=0`, `b tanh(a x / b)` for `x<0`|linear above, tanh below — the gate specified in SPEC.md|
|`t`|`tanh(x)`|the same shape with nothing learned — control for `T`|
|`C`|`clamp(a x, -b, b)`|piecewise-linear `T`; no transcendental|
|`s`|`silu(x)`|what SwiGLU uses|
|`i`|`x`|identity|

| value | | |
|---|---|---|
|`G`|`(B/A) asinh(y / B)`|asinh on both sides|
|`g`|`y/A` for `y>=0`, `(B/A) asinh(y/B)` for `y<0`|linear above, asinh below — the value specified in SPEC.md|
|`s`|`silu(y)`| |
|`i`|`y`|identity — what SwiGLU uses|

Three of these have a **linear positive half**: `F`, `f` and `g`. `T` and `G` do
not. Section 4 is about what that half costs.

Models are named gate-then-value, so `fg` is the pair SPEC.md specifies, `si` is
SwiGLU and `ii` is bilinear.

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

## 2. The value branch, and what it is worth

Depth 32, 8 seeds, $\tanh$-product teacher, identical parameter counts (136,392):

| gate | value | loss | vs $f\odot g$ |
|---|---|---|---|
|silu|silu|**0.01566 ± 0.00086**|−0.6% (−0.2 σ)|
|$f$|$g$|**0.01575 ± 0.00064**|—|
|$g$|$g$|0.01697 ± 0.00056|+7.7% (4.0 σ)|
|$f$|$f$|0.01717 ± 0.00102|+9.0% (3.3 σ)|
|silu|$g$|0.01969 ± 0.00084|+25.0% (10.5 σ)|
|$f$|identity|0.04925 ± 0.00072|+213% (98 σ)|
|silu|identity — SwiGLU|0.07117 ± 0.00031|+352% (220 σ)|
|identity|identity — bilinear|0.08786 ± 0.00101|+458% (171 σ)|

On this teacher the effect is **making the value branch nonlinear**, worth
3.6–4.5×. Among the two rows above, *which* nonlinearity is worth nothing —
silu⊙silu and $f\odot g$ are within 0.2 σ of each other.

That conclusion came from a table in which **every gate is linear above zero**.
Section 4 adds gates that are not, and the picture changes: `TG` reaches
**0.00162 ± 0.00011** on this same teacher, 9.7× below both rows above. Which
nonlinearity does matter, and the pair that shows it is not $f\odot g$. Contributions separate cleanly: swapping the value
identity for $g$ takes 0.07117 → 0.01969 (3.6×), swapping the gate silu for $f$
takes 0.07117 → 0.04925 (1.4×). **The value branch carries about twice the gate's
effect.**

Bilinear — no nonlinearity on either branch — is **last**, behind SwiGLU. In
Dauphin et al. (2017) bilinear beat a purely linear network by >40 perplexity and
lost to GLU by 20; here the product alone is not enough at all.

## 3. A second teacher reverses which ingredient matters

The $\tanh$-product teacher favours $\tanh$-shaped students — $f$'s negative side
*is* $b\tanh(ax/b)$ — and a gated student on a product teacher can win by
resembling it. On an iterated map with no product at all, depth 16, 8 seeds:

| gate | value | loss | vs $f\odot g$ |
|---|---|---|---|
|$f$|$g$|**0.05236 ± 0.00117**|—|
|silu|identity — SwiGLU|0.06952 ± 0.00112|+32.8% (30 σ)|
|silu|silu|0.07132 ± 0.00145|+36.2% (29 σ)|
|identity|identity — bilinear|0.09263 ± 0.00034|+76.9% (94 σ)|

**silu⊙silu no longer beats SwiGLU.** Making the value branch nonlinear buys
nothing here; only $f\odot g$ does.

$f\odot g$ is not the best configuration on this teacher, though — section 4
changes one ingredient at a time and reaches 0.01676, another 3.1× below the
row above.

| | $\tanh$ product | iterated map |
|---|---|---|
|value branch nonlinear at all|**4.5×**|**nothing**|
|$f$ and $g$ specifically|**nothing** (ties silu⊙silu)|**33%**|
|removing the value branch's linear half (`g`→`G`)|**1.65–2.98×**|0.96–2.02×|
|removing the **gate's** linear half (`f`→`T`)|**3.26–5.89×**|**1.49–3.12×**|

**The two teachers give opposite accounts about the first two rows.** Neither
"dual nonlinearity is the point" nor "this pair is the point" survives both.

The last row survives both, at the largest size and with no cell that fails:
**the gate's linear positive half is what costs.** The row above it survives
almost as well but has one cell (gate `T`, iterated map) where it buys nothing.
Both rows were added after sections 2 and 3 were written, and both required
gates that sections 2 and 3 did not contain. Section 4 has the measurements;
`TG` and `Tg`, not $f\odot g$, are first, and bilinear is last on both.

Earlier measurements at a smaller scope, kept for the record:

| teacher | $f\odot g$ vs SwiGLU |
|---|---|
|relu product (depth 16)|44×|
|$\tanh$ product (depth 32)|4.5×|
|iterated map (depth 16)|1.33×|

## 4. The linear positive half, on each branch

`F`, `f` and `g` are all linear above zero. `T` and `G` are not. Removing that
half is a single change that can be made on either branch, and the two branches
give different answers.

All 8 seeds, parameter-matched.

**tanh-product teacher, depth 32**

| gate | value | loss | |
|---|---|---|---|
|`T`|`G`|**0.00162 ± 0.00011**|—|
|`t`|`G`|0.00277 ± 0.00012|+71% (20 σ) — gate fixed, nothing learned in it|
|`C`|`G`|0.00326 ± 0.00011|+101% (30 σ) — gate is a clamp, no transcendental|
|`T`|`g`|0.00483 ± 0.00020|+198% (40 σ)|
|`T`|`i`|0.00511 ± 0.00010|+215%|
|`F`|`G`|0.00886 ± 0.00019|+447% (93 σ)|
|`f`|`G`|0.00954 ± 0.00019|+489%|
|`s`|`s`|0.01566 ± 0.00086|+867%|
|`f`|`g`|0.01575 ± 0.00064|+872%|
|`F`|`g`|0.01848 ± 0.00048|+1041%|
|`t`|`i`|0.01907 ± 0.00113|+1077%|
|`s`|`i` — SwiGLU|0.07117 ± 0.00031|+4294%|

**iterated-map teacher, depth 16**

| gate | value | loss | |
|---|---|---|---|
|`T`|`g`|**0.01676 ± 0.00073**|—|
|`T`|`G`|0.01747 ± 0.00074|+4.2% (1.9 σ — a tie)|
|`C`|`G`|0.01902 ± 0.00046|+13% (5.0 σ) — gate is a clamp, no transcendental|
|`t`|`G`|0.02117 ± 0.00057|+26% — gate fixed, nothing learned in it|
|`F`|`G`|0.02506 ± 0.00055|+50% (23 σ)|
|`f`|`G`|0.02607 ± 0.00074|+56%|
|`T`|`i`|0.02643 ± 0.00086|+58%|
|`F`|`g`|0.05071 ± 0.00103|+203%|
|`f`|`g`|0.05236 ± 0.00117|+212% (73 σ)|
|`s`|`i` — SwiGLU|0.06952 ± 0.00112|+315%|
|`s`|`s`|0.07132 ± 0.00145|+325%|
|`t`|`i`|0.08321 ± 0.00108|+396% — **worse than SwiGLU**|
|`i`|`i` — bilinear|0.09263 ± 0.00034|+453%|

### What each removal is worth

Gate, `f` → `T`, holding the value branch fixed:

| | value `g` | value `G` |
|---|---|---|
|tanh product|**3.26×**|**5.89×**|
|iterated map|**3.12×**|**1.49×**|

Value, `g` → `G`, holding the gate fixed:

| | gate `f` | gate `F` | gate `T` |
|---|---|---|---|
|tanh product|**1.65×**|**2.09×**|**2.98×**|
|iterated map|**2.01×**|**2.02×**|0.96× — no effect|

**The gate's linear half costs more, and costs it everywhere.** All four gate
cells are gains, on both teachers, 1.5–5.9×. The value branch is 1.0–3.0× and
has one cell where it does nothing at all.

**Section 2 and section 3 were measured inside a family where every gate was
linear above zero.** `f` and `F` differ only below zero, so comparing them
measured the tanh shoulder, not the linear half — which is why section 4 first
reported the gate as worth 4%. With `T` in the comparison the ordering reverses:
the gate is the larger effect.

### The two changes are redundant, not additive

On the iterated map:

| | loss |
|---|---|
|`Ti` — gate fixed only|0.02643 ± 0.00086|
|`fG` — value fixed only|0.02607 ± 0.00074|
|`FG`|0.02506 ± 0.00055|
|`TG` — both|0.01747 ± 0.00074|

`Ti` and `fG` are 0.9 σ apart: fixing either branch alone lands in the same
place. Doing both is a further 1.5×, not the 3×1.5 that independent effects
would give.

### The control: boundedness, or the learned scale?

`T` learns a slope and a saturation level; `silu` learns nothing. `t` is the
control — `tanh(x)` with nothing learned. Note that `b tanh(a x / b)` equals
`b tanh((a/b) x)`, and a per-channel gain on the gate is absorbable into `W_d`'s
columns, so the only shape `T` can learn is the knee position `a/b`. A trained
`TG` reaches `a/b ≈ 1.78`; `t` pins it at 1.

What the learned knee is worth depends entirely on the other branch:

| | fixed `t` | learned `T` | learned is worth |
|---|---|---|---|
|value `G`, iterated map|0.02117|0.01747|1.21×|
|value `G`, tanh product|0.00277|0.00162|1.71×|
|value `i`, iterated map|0.08321|0.02643|**3.15×**|
|value `i`, tanh product|0.01907|0.00511|**3.73×**|

With a nonlinear value branch the fixed gate is already most of the way there —
`tG` at 0.02117 still beats every linear-above gate, so **boundedness alone
buys 2.5× and the learned knee adds 1.2×**. With a linear value branch the
fixed gate is a disaster: `ti` at 0.08321 is **worse than SwiGLU** and nearly at
bilinear.

The operating point explains it. Magnitude of the argument reaching the `tanh`,
after 2000 steps on the iterated map (raw MSE here, not normalised):

| | loss | median \|arg\| | 99th pct | median \|out\| / saturation |
|---|---|---|---|---|
|`ti`|0.04903|0.378|1.56|0.361|
|`tG`|0.02302|0.904|3.68|0.718|
|`Ti`|0.02987|3.726|18.48|0.999|
|`TG`|0.01931|3.173|17.26|0.996|
|`si`|0.04468|0.490|2.17|—|

**A fixed `tanh` sits in its own linear region** (median argument 0.378), so with
a linear value branch the whole block degenerates towards bilinear — which is
where `ti` lands. The learned gates drive themselves deep into saturation
instead. So the answer is neither "boundedness" nor "learnability" on its own:
the learned scale is what places the operating point, and the operating point
only decides the outcome when the gate is the sole nonlinearity.

### The saturated gate is not a sign function

Most of a trained `T` is saturated, which suggests it has become `b·sign(x)`.
It has not. Taking a trained `TG` and swapping only the gate (same weights,
raw MSE, ratios are what matter):

| gate at evaluation | loss | vs trained |
|---|---|---|
|`b tanh(a x / b)` — as trained|0.00855|1.000×|
|`clamp(a x, -b, b)`|0.01016|1.189×|
|`b sign(x)`|0.03164|**3.699×**|

The linear ramp through zero is doing real work; only the smooth knee above it
is cheap. Trained *with* the clamp from the start, `CG` reaches 0.01902 on the
iterated map (1.09× off `TG`) and 0.00326 on the tanh product (2.01× off). **The
gate needs no transcendental at all** — two comparisons and a multiply — at a
cost of 9% on the teacher that is not made of tanh.

### Two things this does not establish

**The gap against SwiGLU is largest where the teacher is a tanh product**
(13.9× for `Ti`, against 2.6× on the iterated map). SwiGLU wins on real language
against exactly this kind of bounded gate. A 14× in the other direction on a
synthetic teacher is more likely a statement about the teacher than about
SwiGLU. Section 8.

## 5. Two explanations tested and eliminated

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

## 6. Depth

$\tanh$-product teacher, 8 seeds, $L^{-1/2}$ initialisation throughout:

| depth | $f\odot g$ | SwiGLU | MLP gelu |
|---|---|---|---|
|4|0.12666 ± 0.00124|0.13823 ± 0.00024|0.14067 ± 0.00109|
|32|**0.01587 ± 0.00059**|0.07095 ± 0.00086|0.11926 ± 0.00128|

**9% at depth 4 becomes 4.5× at depth 32.** Without the $L^{-1/2}$ init, SwiGLU
does not reach depth 16 at all (NaN), and plain MLPs lose 30–41% (SPEC 5).

## 7. Curvature-aware ternary rounding: tried, failed

Rank weights by a second-order estimate of the cost of rounding them, keep the
top $k$ per row, so the sparsity is identical across criteria. Depth 16, 8 seeds,
$\tanh$-product teacher, three matrices ($W_g,W_u,W_d$) ternarised:

| criterion | kept | loss |
|---|---|---|
|real weights|1.00|0.03256 ± 0.00271|
|**magnitude $\lvert\mu\rvert$**|0.50|**0.09711 ± 0.00163**|
|$\sqrt{v}\,\lvert\mu\rvert$|0.50|0.18702 ± 0.00345|
|$v\,(2\lvert\mu\rvert-s)$|0.50|0.20750 ± 0.00445|
|**magnitude**|0.25|**0.13039 ± 0.00082**|
|$v\,(2\lvert\mu\rvert-s)$|0.25|0.47496 ± 0.01923|

$v$ is Adam's second moment, used as a diagonal curvature estimate.

Two things went wrong. First, the criterion: for *pruning to zero* the cost is
$\tfrac12 H\mu^2$, giving $\sqrt{H}\lvert\mu\rvert$; but ternarisation sends a kept
weight to $\pm s$, not to its own value, so the correct quantity is

$$\tfrac12 H\big[(s-\lvert\mu\rvert)^2-\mu^2\big]=\tfrac12 H s\,(s-2\lvert\mu\rvert)
\quad\Longrightarrow\quad\text{keep the largest } H(2\lvert\mu\rvert-s)$$

Second, **fixing the derivation made it slightly worse** (0.18702 → 0.20750).
Note also that $2\lvert\mu\rvert-s$ without the $H$ factor gives *exactly* the
magnitude ranking — it is a monotone function of $\lvert\mu\rvert$ — so only the
$H$ weighting can change anything, and it changes it for the worse.

Likely reasons, untested: Adam's $v=\mathbb{E}[g^2]$ carries no correlation
between weights, where GPTQ uses the input covariance $H=2XX^\top$ and compensates
the remaining weights after each rounding; and $v$ is large for weights that are
still *moving*, not for weights that are *important*. **Magnitude is the best
criterion tried.**

## 8. What this does not show

- sections 1–9 are one task family (synthetic regression, width 64, input dim
  32) with **no normalisation anywhere**. A character-level language model with
  LayerNorm was measured afterwards and **reverses most of what follows** —
  see [`LM.md`](LM.md). The short version: with a LayerNorm in place, SwiGLU
  wins, the value branch's `asinh` buys nothing, the bounded gate loses, and the
  per-channel learned scale is worth zero. Every mechanism that works below is a
  way of setting a per-channel operating point, and normalisation sets it
  already. That file is not yet integrated into these sections
- **two teachers disagree about which ingredient matters** (section 3)
- **dual-nonlinear gating is not new**: it is the Gated Tanh Unit, measured and
  rejected by Dauphin et al. (2017) on the grounds that both branches contribute
  a gradient downscaling factor. That rejection does not reproduce here, and
  **why it does not is unresolved** — the obvious explanation (that $f$ and $g$
  have derivative exactly $a$ and $1/A$ on their whole positive half, so nothing
  downscales) fails, because silu⊙silu has no such property and ties $f\odot g$
  on one of the two teachers
- $\Phi$ has been verified as an identity and logged as a diagnostic; **it has not
  been used as a regulariser or a training objective**
- $a,b,A,B$ barely move during training; the initialisation is doing the work and
  it is not understood why the learning does not use these degrees of freedom
- no comparison against LayerNorm'd blocks, which is how real transformers avoid
  the instability that forced the $L^{-1/2}$ init here

## 9. Reproducing

```bash
python bench.py --diagnose                       # baselines: is the task usable?
python bench.py --depth 32 --seeds 8 --model fg,swiglu,gelu
python bench.py --depth 32 --seeds 8 --model fg,ss,ff,gg,sg,fi,swiglu,bilinear
python bench.py --depth 16 --seeds 8 --model fg,ss,swiglu,bilinear --teacher deep
```

Seeds are stacked into a leading dimension, so 8 seeds cost little more than 1
(≈35× faster than looping). Gradient clipping is per-seed; a global norm would
couple the models.
