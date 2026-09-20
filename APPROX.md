# Piecewise approximation — asinh for hardware

The value branch of this layer is `g_sym(y) = (B/A) asinh(y/B)`. This document
replaces the `asinh` with polynomials only, for targets that have no
transcendental unit. **It is not a GPU optimisation** — see section 6.
Coefficients come from Remez, quantised to a 2^-24 grid.

## 1. The segments come from the argument distribution, not from asinh

A depth-16, width-64, 4-seed $F\odot G$ network was trained for 3000 steps on the
iterated-map teacher, and the arguments $|y/B|$ actually reaching `asinh` were
recorded (4 seeds × 4096 points × all 16 layers):

| range | share |
|---|---|
| [0,1) | 5.6 % |
| [1,2) | 5.6 % |
| [2,4) | 11.0 % |
| [4,8) | 20.5 % |
| [8,∞) | 57.3 % |

Median 9.58, 99th percentile 40.4, maximum 135. After training $B$ shrinks to
0.046–0.142, which is what makes $y/B$ large. **78% of all evaluations land at
$u \ge 4$**, where `asinh` is essentially a logarithm.

An earlier design split $[0,1]/[1,2]/[2,4]$ and treated everything above as a
tail, spending its highest degree on $[0,1]$. That reads the traffic backwards:
$[0,1]$ carries 5.6% of it. **Segment boundaries follow the distribution the
network produces, not the shape of the function.**

## 2. The form

With $u = |y/B| \ge 0$; `asinh` is odd, so the sign is restored at the end.

```
u < 1          u · P_odd(u²)
1 <= u < 2^12  ln( u + sqrt(1 + u²) )      an identity — no truncation error
u >= 2^12      ln2 + ln(u)                 asinh(u) − ln(2u) ~ 1/(4u²) < 1.5e-8
```

The upper split exists to stop $u^2$ from overflowing in float32, not for
accuracy. Writing `ln2 + ln u` instead of `ln(2u)` avoids forming $2u$, for the
same reason: the result is then finite across the whole float32 range.

The logarithm splits into exponent and mantissa. With $s = 2^e m$, $m \in [1,2)$:

```
ln s = e·ln2 + ln m
ln m = 2 z P_log(z²),   z = (m−1)/(m+1) ∈ [0, 1/3]
```

In this form `P_log` needs degree 3 (relative 1.65e-07). Fitting $m \in [1,2)$
directly needs degree 7 for the same absolute accuracy (1.92e-07). The atanh
form costs one divide to build $z$, so counting multiply-adds it is 3+1+1
against 7 — not a large difference. Choose the direct form where division is
expensive, the atanh form where degree is.

## 3. Coefficients (2^-24 grid)

```
P_odd  (degree 5, relative 1.09e-06)
  [ 0.999998927116394, -0.1665818691253662,  0.07390326261520386,
   -0.03923112154006958, 0.01714622974395752, -0.0038627982139587402]

P_odd  (degree 4, relative 8.08e-06)    <- the one dnglu.py ships
  [ 0.9999919533729553, -0.16622233390808105, 0.07097268104553223,
   -0.03091973066329956, 0.007558107376098633]

P_log  (degree 3, relative 1.79e-07)
  [ 0.9999998211860657, 0.3333800435066223, 0.19794732332229614,
    0.1711302399635315]
```

`P_log`'s coefficients sit near 1, 1/3, 1/5, 1/7 because this is the atanh
series; that is a check, not a coincidence.

## 4. Measured in float32

| input | max absolute | max relative |
|---|---|---|
| uniform [0,1] | 1.02e-06 | 1.18e-06 |
| uniform [1,4] | 3.61e-07 | 2.38e-07 |
| uniform [4,140] | 4.73e-07 | 1.23e-07 |
| \|N(0, 9.6)\| — shaped like the measured distribution | 9.89e-07 | 1.17e-06 |
| [0, 1e-3] | 1.13e-09 | 1.15e-06 |
| [1e3, 1e6] | 1.66e-06 | 1.57e-07 |
| [1e18, 3e38] | 8.88e-06 | 1.05e-07 |

No NaN or inf anywhere in float32. $-0.0$ keeps its sign (`copysign`, not
`sign`, which would return $+0.0$).

## 5. Swapping it into a trained network

Depth 16, width 64, 8 seeds, iterated-map teacher, 4000 steps, then the same
weights evaluated on 20 × 1024 fresh points with `torch.asinh` replaced:

| value branch | loss (seed mean) | vs baseline | worst single-seed deviation |
|---|---|---|---|
| `torch.asinh` (baseline) | 0.016702 | 1.0000× | — |
| piecewise, odd deg 5 + log deg 3 | 0.016702 | 1.0000× | 1.13e-07 |
| piecewise, odd deg 4 + log deg 3 | 0.016702 | 1.0000× | 1.13e-07 |

Seven figures. Raising the odd segment from degree 4 to 5 changes nothing —
94% of evaluations take the logarithm path, so the accuracy of $[0,1]$ does not
reach the loss. **Degree 4 is enough.**

The same on a network trained with the `exact` pair ($f_{a,b} \times g_{A,B}$,
`asinh` on the negative side only), replacing only the `asinh` inside $g$:

| value branch | loss (seed mean) | vs baseline | worst single-seed deviation |
|---|---|---|---|
| `g_AB` exact (baseline) | 0.033041 | 1.0000× | — |
| piecewise, odd deg 5 + log deg 3 | 0.033041 | 1.0000× | 1.15e-07 |
| piecewise, odd deg 4 + log deg 3 | 0.033041 | 1.0000× | 1.13e-07 |
| `g_fast` = u/sqrt(1+u²/3) | 0.034510 | 1.0444× | 5.55e-02 |
| both-sides asinh (a different function) | 0.051260 | 1.5514× | 6.32e-01 |

The existing `g_fast` swap is **not free** — it costs 4.4% loss. The piecewise
form costs nothing measurable. The last row is not an approximation at all; it
confirms the standing warning that `g_sym` must not be dropped into a network
trained with `g_AB`.

## 6. Do not use this on a GPU

`asinh_pw` as shipped, measured on an RTX 5090 in float32:

| elements | `torch.asinh` | `asinh_pw` | ratio |
|---|---|---|---|
| 2,000,000 | 0.0303 ms | 0.9235 ms | 30.5× |
| 32,000,000 | 0.2051 ms | 14.584 ms | 71.1× |

`torch.asinh` costs about as much as an add. These kernels are bandwidth-bound,
not ALU-bound, so splitting into segments only adds traffic — and the ratio gets
*worse* with size, which an arithmetic-count explanation cannot produce. The
piecewise form is for hardware with no transcendental unit and no room for a
table; it is not an inference speedup.

The same measurements corrected two claims that had already been published here:

- "a `where` measures ~4x an `asinh`" — measured 1.2–2.4× (N = 65k … 32M).
- "with small batches the layer is launch-bound, so op count is what matters" —
  it is bandwidth-bound. `where(x>=0,p,q)` reads three tensors and a condition
  where `asinh(x)` reads one; the 2.1–2.4× at large N is exactly that traffic
  ratio.

Dropping a branch does help, but because it removes tensor traffic, not
arithmetic.

## 7. Is tanh needed at all?

The best-measured pair, $F \odot G$, uses $\max(ax, -b)$ on the gate and contains
no `tanh`. Only the `exact` $f$ needs one. For that case the arguments $|ax/b|$
have median 2.28 and 99th percentile 14.7, with 73% inside $[0,4)$ — the
opposite shape to `asinh`, with the traffic near the origin.

A four-segment tanh tape was built ($[0,1]$ odd deg 4, $[1,2]$ deg 4, $[2,4]$ deg
5, all within 1.4e-05 relative in float32), but truncating at $u > 4$ to a constant 1 leaves
6.7e-04, so that branch needs extending before it is usable. It is not included
here.

## 8. Method note

The first Remez implementation used for this diverged on the odd form
$u\,P(u^2)$ — 5.2e+04 at degree 4, 1.0e+09 at degree 5. The cause was $u = 0$
among the initial reference points: the odd form's error is identically zero
there, so the interpolation row becomes $[0,\dots,0,\pm 1]$ and demands error
$\pm h$ at a point where no coefficient can produce it. The system is
inconsistent, not ill-conditioned.

The fits here use a Remez implementation that takes the exponent set directly,
so the substitution $t = u^2$ removes the problem rather than working around it.

0BSD.
