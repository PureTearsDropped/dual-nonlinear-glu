# Specification

`[verified]` = checked numerically in this repository. `[untested]` = designed, not measured.

## 1. The block

$$\boxed{\;h \leftarrow h + W_d\Big(f_{a,b}\big(W_g h\big)\ \odot\ g_{A,B}\big(W_u h\big)\Big) + \beta\;}$$

SwiGLU is the special case $f=\mathrm{silu}$, $g=\mathrm{id}$; the Gated Tanh Unit
of Dauphin et al. (2017) is $f=\tanh$, $g=\sigma$.
**Both branches nonlinear is not a new design** — it is GTU, which that paper
measured and rejected because $\tanh'$ and $\sigma'$ both downscale the gradient.
What is specified here is one instance of that family whose effect on two
synthetic teachers is reported in RESULTS; the two teachers disagree about
whether the dual nonlinearity or the particular pair is what matters.

$W_g,W_u\in\mathbb{R}^{m\times n}$, $W_d\in\mathbb{R}^{n\times m}$. Taking
$m=n/3$ matches the parameter count of a plain MLP block.

## 2. The two functions

$$f_{a,b}(x)=\begin{cases}a\,x & x\ge 0\\[4pt] b\,\tanh\!\big(\tfrac{a x}{b}\big) & x<0\end{cases}
\qquad
g_{A,B}(y)=\begin{cases}\dfrac{y}{A} & y\ge 0\\[6pt] \dfrac{B}{A}\,\mathrm{asinh}\!\big(\tfrac{y}{B}\big) & y<0\end{cases}$$

```python
f_ab = lambda x, a, b: torch.where(x >= 0, a*x,  b*torch.tanh(a*x/b))
g_AB = lambda y, A, B: torch.where(y >= 0, y/A, (B/A)*torch.asinh(y/B))
```

| | $f$ | $g$ |
|---|---|---|
|above zero|linear, slope $a$|linear, slope $1/A$|
|below zero|**saturates to $-b$**|**grows like $\ln(2\lvert y\rvert/B)$**|
|range|$(-b,\infty)$|$\mathbb{R}$|
|smoothness|$C^2$, not $C^3$|$C^2$, not $C^3$|
|at 0|$f'(0)=a$, $f''(0)=0$, $f'''(0^-)=-2a^3/b^2$|$g'(0)=1/A$, $g''(0)=0$, $g'''(0^-)=-1/(AB^2)$|

$a,b,A,B$ are learned per channel, kept positive by softplus.

**The pair above is not the best-measured form.** Both $f_{a,b}$ and $g_{A,B}$
are linear above zero, and on this benchmark that linear half is what costs.
Removing it from either branch gives

$$T_{a,b}(x)=b\,\tanh\!\big(\tfrac{a x}{b}\big),\qquad
  G_{A,B}(y)=\frac{B}{A}\,\mathrm{asinh}\!\big(\tfrac{y}{B}\big),$$

each applied on **both** sides. Both are branch-free. Removing it from the gate
is worth 1.5–5.9× and from the value branch 1.0–3.0×, on two teachers that
disagree about much else (RESULTS 4). The two are largely redundant: fixing
either branch alone lands in the same place.

These are different functions, not approximations of $f_{a,b}$ and $g_{A,B}$ —
the positive halves are compressed too — so a network trained with one pair must
not be evaluated with the other. `dnglu.py` defaults to $T_{a,b}\odot G_{A,B}$
(`mode="gtu"`); $\max(ax,-b)\odot G_{A,B}$ is `mode="sym"` and the pair
specified above is `mode="exact"`.

What this costs is the conjugacy: $g_{A,B}$ was built so that $g\circ f$ deviates
from the identity in a controlled way, and $G_{A,B}$ is not that. What it does
*not* cost is section 3 or the smoothness. The potential stays elementary,

$$\int_0^y G_{A,B} = \frac{B}{A}\Big(y\,\mathrm{asinh}\tfrac{y}{B} - \sqrt{B^2+y^2} + B\Big),\quad\text{[verified]}$$

and $G_{A,B}$ is analytic everywhere, where $g_{A,B}$ is only $C^2$. The
initialisation and the diagnostics below apply unchanged.

**Do not rewrite with ReLU.** `a*relu(x) - b*tanh(a*relu(-x)/b)` has identical
values but **the gradient is wrong at exactly $x=0$** — PyTorch's `relu'(0)=0`
makes it 0 where the correct value is $a$. `[verified]` The `where` form is also
1.36x faster (0.178 ms vs 0.242 ms for 2M elements).

**Never write** `atanh(tanh(u))` to build $g$: on the reals that is the identity,
so $g$ collapses to $y/A$ and $B$ disappears; in float32, $\tanh(10)$ rounds to
1.0 and `atanh(1.0) = inf`. The useful identity is

$$\mathrm{atanh}\!\left(\frac{u}{\sqrt{1+u^2}}\right)=\mathrm{asinh}(u)$$

## 3. The potential

With $F'=f$ and $G'=g$, set $\Phi(u,v)=F(u)\,G(v)$. Then

$$\boxed{\;\frac{\partial^2\Phi}{\partial u\,\partial v}=f(u)\,g(v)=\text{the block's output}\;}$$

`[verified]` to $1.6\times10^{-9}$ by finite differences.

**For this pair, $F$ and $G$ are elementary:**

$$F_{a,b}(x)=\begin{cases}\dfrac{a}{2}x^2 & x\ge0\\[6pt] \dfrac{b^2}{a}\ln\cosh\!\big(\tfrac{ax}{b}\big) & x<0\end{cases}
\qquad
G_{A,B}(y)=\begin{cases}\dfrac{y^2}{2A} & y\ge0\\[6pt] \dfrac{B}{A}\Big[y\,\mathrm{asinh}\tfrac{y}{B}-\sqrt{B^2+y^2}+B\Big] & y<0\end{cases}$$

both with $F(0)=G(0)=0$.

**SwiGLU has a potential too but it is not elementary:** $\int\mathrm{silu}$ needs
a dilogarithm, $\int x\sigma(x)dx = x\ln(1+e^x)+\mathrm{Li}_2(-e^x)$.
`[verified]` — an earlier elementary guess for it was wrong.

What this buys `[untested]`:

- a **diagnostic**: log $\mathbb{E}[\Phi]$, $\mathbb{E}\lvert fG\rvert$, $\mathbb{E}\lvert Fg\rvert$ per layer
- a **training-only regulariser**: $\mathcal{L}+\lambda\mathbb{E}[\Phi^2]$, or on the
  gradient $\lVert\nabla\Phi\rVert^2=[f(u)G(v)]^2+[F(u)g(v)]^2$
- an **interpretation**: the output is the cross-curvature of a potential in the
  two features, rather than "gate times value"
- **train with $\Phi$, infer without it** — $\Phi$ never appears in the forward pass

What it does **not** buy: this is a *local* potential per gate. The network is not
an energy-based model and nothing about stability follows from it.

### 1.1 On the origin

Near zero, $f(u)\approx au$ and $g(v)\approx v/A$, so the branch output is
$\approx(a/A)uv$ — **bilinear, not identity**. The identity path is the residual.
In the all-positive quadrant $f$ and $g$ are *exactly* linear, so the block is
exactly bilinear there; curvature appears only when a coordinate crosses zero.
SwiGLU by contrast is curved everywhere, since $\mathrm{silu}$ is.

Tying $A=a$ makes the series composition $g(f(x))$ exactly the identity on the
positive side. **It is measurably worse** — +5.7% at depth 4 and +6.8% at depth 32
in the series construction — because $a$ also sets the slope entering $\tanh$ on
the negative side, and constraining it for the sake of the positive side costs
more than the tidiness is worth.

## 4. Initialisation

| symbol | value | note |
|---|---|---|
|$W_g,W_u$|$\mathcal{N}(0, 1/n)$| |
|$W_d$|$\mathcal{N}(0, 1/m)\cdot L^{-1/2}$|**$L^{-1/2}$ is not optional** (section 5)|
|$a,A$|1.0| |
|$b,B$|0.1|section 4.1|
|$\beta$|0| |

### 4.1 Choosing `b` and `B` — measure, do not guess

`[verified]` **$b$ barely moves during training**; what matters is where it starts.

Measure the 99th percentile of $a\lvert x\rvert/b$ (for $x<0$), the argument
reaching `tanh`:

| 99th pct | state |
|---|---|
|$\ll1$|saturation never engages; $f$ is effectively linear. **Lower $b$**|
|$1$–$3$|good|
|$\gg3$|fully saturated. **Raise $b$**|

Same for $\lvert y\rvert/B$ entering `asinh`.

### 4.2 The composition scale

If the two functions are composed in series rather than multiplied
(a different construction, measured in
[relu-tanh-asinh](https://github.com/PureTearsDropped/relu-tanh-asinh)),

$$g_{a,b}\big(f_{a,b}(x)\big)-x=-\frac{a^2}{2b^2}x^3+O(x^5)$$

`[verified]` so the local nonlinearity there scales as $(a/b)^2$. **$a$ is
therefore not a free gain the incoming weights can absorb.** The same ratio is a
useful guide here.

## 5. The residual-branch initialisation

$$W_d \sim \mathcal{N}\!\left(0,\ \frac{1}{m}\right)\cdot\frac{1}{\sqrt{L}}$$

`[verified]` This is worth more than any activation choice measured here:

| depth 64, 8 seeds | without | with | change |
|---|---|---|---|
|MLP relu|0.27164|**0.16099**|−41%|
|MLP gelu|0.17293|**0.11773**|−32%|
|MLP silu|0.17002|**0.11861**|−30%|
|SwiGLU|**NaN**|**0.05794**|—|

**SwiGLU reaches NaN by depth 16 without it**, because
$\mathrm{silu}(W_gh)\odot(W_uh)$ is quadratic in $h$ and there is no
normalisation in these blocks. Real transformers avoid this with a LayerNorm
before the FFN; the $L^{-1/2}$ scaling alone is enough here.

## 6. Diagnostics before trusting any number

**In this order. Do not proceed past a failure.** Skipping 1–3 or 5 means reading
a linear model's score as if it were a nonlinear one's; that happened twice
during this work.

| # | check | pass |
|---|---|---|
|1|**linear baseline** (Linear→Linear)|the model must beat it|
|2|**one-nonlinearity baseline** (Linear→$f$→Linear)|a deep model must clearly beat it|
|3|**depth value** = (2)/(deep)|**≥ 1.5**; near 1 means the task does not need depth|
|4|99th percentile into `tanh` and into `asinh`|1–3 (4.1)|
|5|**ablation**: zero the interior blocks|the model must clearly beat it|
|6|per-layer activation std across depth|should not drift; if it does, check 5|
|7|seeds|**≥ 8**; report difference ÷ standard error, not just means|
|8|**more than one teacher**|a gated student on a product teacher is measuring the match, not the architecture (RESULTS 3)|
