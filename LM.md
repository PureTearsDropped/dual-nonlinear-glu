# A real task: enwik8 with LayerNorm

Everything else in this repository is synthetic teacher–student regression with
no normalisation. This file records what happened on a character-level language
model with LayerNorm, which is the setting the synthetic work was standing in
for. **It reverses most of the synthetic conclusions**, and the reversals have a
single consistent explanation.

Raw numbers only; the integration into RESULTS.md is not done yet.

## Setup

enwik8, byte level. First 90 MB train, next 5 MB validation. Pre-norm decoder,
`d_model` 256, 6 layers, 4 heads, context 256, `d_ff` 512, batch 32 — so 20 000
steps is 164 M tokens, about 1.8 epochs. AdamW, weight decay 0.1, betas
(0.9, 0.95), cosine to zero, 200 warmup steps, gradient clip 1.0. Loss is
validation bits per byte over 60 fresh batches. SwiGLU is 4 141 568 parameters;
every variant below is matched to that within 0.3% and the `Tk`/`Pow`/`sigmapi`
families match it exactly.

FFN variants, gate ⊙ value, elementwise:

| name | gate | value | learned activation params |
|---|---|---|---|
|`swiglu`|`silu(g)`|`u`|0|
|`Ti`|`b tanh(a g / b)`|`u`|2 per channel|
|`TG`|`b tanh(a g / b)`|`(B/A) asinh(u/B)`|4 per channel|
|`TkG K`|`tanh(k_j g)`, `k_j` fixed, K values over [1/4, 4]|`asinh(u)`|**0**|
|`Pow w`|`silu(g)`|`sign(u)|u|^w`, `w` fixed|**0**|
|`sigmapi K`|`silu(u_1)`|`u_2 ⋯ u_K`, `d_ff` scaled by 3/(K+1)|**0**|
|`fg`|`a g` (g≥0), `b tanh(ag/b)` (g<0)|`u/A` (u≥0), `(B/A)asinh(u/B)` (u<0)|4 per channel|

## 1. Learning rate, seed 0

Every variant optimises near 4e-3; the first comparisons were run at 1e-3, which
is off the optimum for all of them.

| | 5e-4 | 1e-3 | 2e-3 | 4e-3 | 6e-3 | 8e-3 |
|---|---|---|---|---|---|---|
|`swiglu`|1.6661|1.5998|1.5516|**1.5230**|—|1.6263|
|`Ti`|1.6648|1.6104|1.5596|**1.5355**|—|—|
|`TG`|—|1.6092|1.5638|**1.5367**|1.5511|—|

## 2. SwiGLU wins, and the value branch buys nothing

Three seeds at 1e-3. Seeds are paired: the same seed is the same initialisation
and the same data order, so the paired column is the meaningful one.

| | mean | sd | per seed |
|---|---|---|---|
|`swiglu`|1.5905|0.0086|1.5998 1.5888 1.5828|
|`Ti`|1.6011|0.0089|1.6104 1.6003 1.5926|
|`TG`|1.6034|0.0084|1.6092 1.6073 1.5938|

| paired difference | mean | per seed | t (df 2) |
|---|---|---|---|
|`TG` − `Ti`|+0.0024|−0.0012 +0.0070 +0.0013|+0.97|
|`TG` − `swiglu`|+0.0129|+0.0094 +0.0184 +0.0110|+4.66|
|`Ti` − `swiglu`|+0.0106|+0.0106 +0.0114 +0.0097|+21.9|

**`TG` − `Ti` changes sign across seeds.** The value branch's `asinh`, worth
1.65–2.98× on the synthetic teachers, is worth nothing here. And the bounded
gate, worth 1.5–5.9× there, loses to `silu` here on all three seeds.

At the tuned 4e-3 the ordering is the same: `swiglu` 1.5230, `Ti` 1.5355,
`TG` 1.5367. Cost: `swiglu` 13–14 ms/step, `TG` 20–21 ms/step, so at equal wall
clock the gap widens.

## 3. The power family: the identity is the optimum

`Pow(y,w) = Exp(w Log y)` restricted to the reals is `sign(y)|y|^w`. It contains
the identity at w = 1 and the log-like end as w → 0. Sweeping it on the value
branch with `silu` on the gate, lr 4e-3, seed 0:

| w | 0.5 | 1.0 | 1.25 | 1.5 |
|---|---|---|---|---|
| | 1.5555 | **1.5230** | 1.5268 | 1.5529 |

w = 1.0 reproduces `swiglu` to five digits (1.5230358 vs 1.5230412), which
verifies the implementation. The curve is a smooth, nearly symmetric valley with
its floor at the identity: compressing and expanding both cost about +0.03.
**SwiGLU's linear value branch is the bottom of this family, not an arbitrary
choice.** `asinh` is one point near the compressive end, so "asinh does not help"
is a special case of "compression does not help".

## 4. Per-channel learned scale buys nothing either

`TkG` fixes the knee per channel from a table of K values on [1/4, 4] and learns
nothing in the activation — exactly SwiGLU's parameter count. lr 4e-3, seed 0:

| K | 1 | 2 | 4 | 16 |
|---|---|---|---|---|
| | **1.5354** | 1.5523 | 1.5431 | 1.5488 |

K = 1 — one shape for every channel, zero learned activation parameters — ties
`Ti` (1.5355) and `TG` (1.5367), which learn 6 144 and 12 288 of them. **More
gate shapes is worse, not better**; K = 2 is worst because its two knees are the
endpoints with nothing in between.

(An earlier K sweep placed the knees at 2^(i−(K−1)/2), so K also widened the
*range*; K = 16 then spanned 0.0055 to 181 and scored 1.6795. That was a design
error, not a property of the idea.)

## 5. More factors loses to more width

`sigmapi K` multiplies K projections instead of 2, with `d_ff` scaled by
3/(K+1) to hold the parameter count. lr 4e-3, seed 0:

| | loss | `d_ff` | params |
|---|---|---|---|
|K = 2 (= SwiGLU)|1.5242|512|4 141 568|
|K = 3|1.5319|384|4 141 568|
|K = 4|1.5531|307|4 140 032|
|`swiglu`, `d_ff` 384|1.5334|384|3 551 744|

At a fixed width of 384 the third factor helps by 0.0015 for 17% more
parameters; at a fixed budget, putting those parameters back into width helps by
0.0077. **Degree 2 is where the value is.** All of these gaps are at or inside
one seed sigma (0.0086), so this is a "no signal" result, not a measured loss.

## 6. Without the FFN's LayerNorm, only the bounded gate survives

Removing the FFN's `LayerNorm` (keeping attention's) with the residual branch
scaled by 1/sqrt(2L), lr 4e-3, seed 0:

| | |
|---|---|
|`swiglu` (silu gate, unbounded above)|**NaN**|
|`fg` (f gate, unbounded above)|**NaN**|
|`TG` (T gate, bounded)|1.8287|

Removing both norms makes `fg` diverge as well. The bounded gate is the only one
that trains at all — which is the synthetic finding about the gate's linear
positive half, appearing here as trainability rather than as loss. But 1.8287 is
far worse than 1.5230 with the norm, so **the activation does not replace
normalisation; it only survives without it.**

The output range explains it: with a=b=A=B=1, `silu(x)·y` and `f(x)·g(y)` reach
±15 over |x|,|y| ≤ 4 and concentrate in one corner, while `T(x)·G(y)` reaches
±2 and uses all four quadrants.

## 7. What the learned parameters do

Median over all channels and layers, after training:

| | knee a/b | b | B |
|---|---|---|---|
|`Ti`, lr 1e-3|1.285|0.769|—|
|`TG`, lr 1e-3|1.279|0.767|0.798|
|`Ti`, lr 4e-3|1.466|0.663|—|
|`TG`, lr 4e-3|1.550|0.600|0.648|
|`TG`, no FFN norm|1.093|0.542|0.535|

They move, but not far from the initialisation at 1.0, and K = 1 with the knee
pinned at 1.0 matches them. On the synthetic task the same parameters moved much
further (a from 1.0 to 0.27) and mattered a great deal.

## 8. One multiplication per node is not a limitation

`(g·h)(u·h) = h^T[(g u^T + u g^T)/2] h` — verified to 4.3e-14 — so one node is
one **rank-2** quadratic form, and the layer is a sum of `d_ff` of them.

Fitting a random symmetric 16×16 `M` with `Σ_j c_j (g_j·h)(u_j·h)`:

| `d_ff` | 4 | 8 | 15 | 16 | 32 |
|---|---|---|---|---|---|
| MSE / var | 1.2e-01 | 3.6e-23 | 1.6e-31 | 1.9e-31 | 2.4e-31 |

`d/2` nodes are enough to span every quadratic form exactly. At `d_model` 256
that is 128 nodes against the 512 available — a factor of four of headroom. The
elementwise product is a cheap factorisation of a quadratic layer, and it is not
the binding constraint.

## The single explanation

Every synthetic effect that fails here was a way of setting a per-channel
operating point: compressing the value branch, bounding the gate, learning the
knee, diversifying the knee. **LayerNorm sets the operating point already**, so
none of them has work left to do. The synthetic benchmark had no normalisation —
which is why it needed the 1/sqrt(L) residual init, and why these mechanisms
looked like 2–6× effects there.

## What this still does not show

One model size (4.1 M), one dataset, 20 000 steps, and — except for section 2 —
**one seed**. Differences of 0.002 to 0.008 are at or inside the seed sigma of
0.0086 and should be read as "no signal", not as measured losses. The learning
rate was swept per variant for `swiglu`, `Ti` and `TG` only; the `Tk`, `Pow` and
`sigmapi` families were run at 4e-3 because that is where those three optimise.

0BSD.
