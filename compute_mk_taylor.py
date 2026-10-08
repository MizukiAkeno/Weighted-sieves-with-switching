"""Rigorous Taylor-model recomputation of Appendix Table 1.

This is a new implementation; it does not modify or call the old mesh C code.
Both delay systems

    (A+)'(x) = A-(x-1)/(x-1),  (A-)'(x) = A+(x-1)/(x-1),
    c_r'(x)  = c_{r-1}(x-1)/(x-1)

are propagated on unit intervals as midpoint polynomials plus a uniform Arb
remainder.  Final one-dimensional integrals are evaluated from those models by
exact polynomial division by their linear factors, with the division tails
folded into the Arb enclosure.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from flint import arb, ctx


@dataclass
class Model:
    coeffs: list[arb]
    rem: arb


def ub(x: arb) -> arb:
    y = abs(x)
    return y.mid() + y.rad()


def errball(r: arb) -> arb:
    return (-r).union(r)


def ae(x: Fraction | int) -> arb:
    x = Fraction(x)
    return arb(x.numerator) / x.denominator


def peval(p: list[arb], x: arb) -> arb:
    z = arb(0)
    for a in reversed(p):
        z = z * x + a
    return z


def eval_model(m: Model, x: arb) -> arb:
    return peval(m.coeffs, x) + errball(m.rem)


def divide_linear(m: Model, a: arb, slope: int, degree: int,
                  radius: arb) -> Model:
    """Enclose P(x)/(a+slope*x) on |x|<=radius."""
    q: list[arb] = []
    prev = arb(0)
    for k in range(degree + 1):
        pk = m.coeffs[k] if k < len(m.coeffs) else arb(0)
        cur = (pk - slope * prev) / a
        q.append(cur)
        prev = cur
    denom_min = a - radius
    tail = ub(q[-1]) * radius ** (degree + 1) / denom_min
    return Model(q, tail + m.rem / denom_min)


def antiderivative(p: list[arb]) -> list[arb]:
    return [arb(0)] + [a / (k + 1) for k, a in enumerate(p)]


def integrate_from(m: Model, left: arb, length_bound: arb) -> Model:
    p = antiderivative(m.coeffs)
    p[0] = -peval(p, left)
    return Model(p, m.rem * length_bound)


def truncate(m: Model, degree: int, radius: arb) -> Model:
    p = list(m.coeffs)
    r = m.rem
    while len(p) - 1 > degree:
        k = len(p) - 1
        r += ub(p.pop()) * radius ** k
    return Model(p, r)


def translate(m: Model, shift: arb) -> Model:
    """Rewrite P(x) as P(z+shift), retaining the same uniform remainder."""
    out = [m.coeffs[-1]]
    for constant in reversed(m.coeffs[:-1]):
        old = out
        out = [shift * old[0] + constant]
        out.extend(old[j - 1] + shift * old[j] for j in range(1, len(old)))
        out.append(old[-1])
    return Model(out, m.rem)


class TaylorTables:
    def __init__(self, max_x: int, max_r: int, degree: int, precision: int):
        ctx.prec = precision
        self.max_x = max_x
        self.max_r = max_r
        self.degree = degree
        self.h = arb(1) / 2
        self.Aplus: list[Model] = []
        self.Aminus: list[Model] = []
        self.c: list[list[Model | None]] = [
            [None for _ in range(max_x)] for _ in range(max_r + 1)
        ]
        self._build_A()
        self._build_c()

    def _propagate(self, same: Model, delayed: Model, old_left: int) -> Model:
        endpoint = eval_model(same, self.h)
        divided = divide_linear(
            delayed, arb(old_left) + self.h, 1, self.degree, self.h
        )
        out = integrate_from(divided, -self.h, arb(1))
        out.coeffs[0] += endpoint
        return truncate(out, self.degree, self.h)

    def _build_A(self) -> None:
        zero = Model([arb(0)], arb(0))
        plus = Model([arb(2)], arb(0))
        minus = zero
        self.Aplus.append(plus)
        self.Aminus.append(minus)
        for old_left in range(1, self.max_x - 1):
            new_left = old_left + 1
            oldp, oldm = plus, minus
            minus = self._propagate(oldm, oldp, old_left)
            if new_left < 3:
                plus = Model([arb(2)], arb(0))
            else:
                plus = self._propagate(oldp, oldm, old_left)
            self.Aplus.append(plus)
            self.Aminus.append(minus)

    def _build_c(self) -> None:
        for n in range(self.max_x):
            self.c[1][n] = Model([arb(1)], arb(0))
        for r in range(2, self.max_r + 1):
            current = Model([arb(0)], arb(0))
            self.c[r][0] = current
            for old_left in range(1, self.max_x - 1):
                new_left = old_left + 1
                if new_left < r:
                    current = Model([arb(0)], arb(0))
                else:
                    delayed = self.c[r - 1][old_left]
                    assert delayed is not None
                    current = self._propagate(current, delayed, old_left)
                self.c[r][new_left] = current

    @staticmethod
    def _piece_index(x: Fraction, max_x: int) -> int:
        n = x.numerator // x.denominator
        if x.denominator == 1 and n == max_x:
            n -= 1
        return max(1, min(n, max_x - 1))

    def _model(self, family: str, n: int, r: int | None = None) -> Model:
        if family == "plus":
            return self.Aplus[n - 1]
        if family == "minus":
            return self.Aminus[n - 1]
        assert family == "c" and r is not None
        model = self.c[r][n]
        assert model is not None
        return model

    def value(self, family: str, x: Fraction, r: int | None = None) -> arb:
        n = self._piece_index(x, self.max_x)
        local = ae(x) - (arb(n) + self.h)
        return eval_model(self._model(family, n, r), local)


    def integrate_kernel(self, family: str, lower: Fraction, upper: Fraction,
                         total: Fraction, r: int | None = None) -> arb:
        """Enclose int f(y)/(y(total-y)) dy, split at integer knots."""
        if upper <= lower:
            return arb(0)
        cuts = [lower]
        for k in range(lower.numerator // lower.denominator + 1,
                       upper.numerator // upper.denominator + 1):
            q = Fraction(k)
            if lower < q < upper:
                cuts.append(q)
        cuts.append(upper)
        ans = arb(0)
        B = ae(total)
        for left, right in zip(cuts, cuts[1:]):
            mid = (left + right) / 2
            n = self._piece_index(mid, self.max_x)
            m = self._model(family, n, r)
            old_center = arb(n) + self.h
            subcenter = (ae(left) + ae(right)) / 2
            m = translate(m, subcenter - old_center)
            xl, xr = ae(left) - subcenter, ae(right) - subcenter
            local_radius = (ae(right) - ae(left)) / 2
            # 1/(y(B-y)) = (1/B)(1/y + 1/(B-y)).
            d1 = divide_linear(m, subcenter, 1, self.degree, local_radius)
            d2 = divide_linear(m, B - subcenter, -1, self.degree, local_radius)
            for d in (d1, d2):
                anti = antiderivative(d.coeffs)
                part = peval(anti, xr) - peval(anti, xl)
                part += errball(d.rem * (xr - xl))
                ans += part / B
        return ans


ROWS = [
    ("0", 2, "505/1000", "5118/1000", None),
    ("0", 3, "295/1000", "8758/1000", None),
    ("0", 4, "184/1000", "14159/1000", None),
    ("0", 5, "118/1000", "22178/1000", None),
    ("0", 6, "78/1000", "33976/1000", None),
    ("0", 7, "52/1000", "51402/1000", None),

    ("half", 2, "473/1000", "10878/1000", "3357/1000"),
    ("half", 3, "267/1000", "20213/1000", "5823/1000"),
    ("half", 4, "164/1000", "33405/1000", "9544/1000"),
    ("half", 5, "105/1000", "52491/1000", "15057/1000"),
    ("half", 6, "69/1000", "80105/1000", "23186/1000"),
    ("half", 7, "46/1000", "120297/1000", "35168/1000"),
]


def midpoint_radius(x: arb) -> tuple[str, str]:
    return str(x.mid()), str(x.rad())


def serialize_row(row: dict) -> dict:
    scalar_names = ("omega_minus", "S_G", "M0", "M1", "M2", "M")
    return {
        **{name: row[name] for name in ("weight", "k", "theta", "s", "t")},
        **{name: midpoint_radius(row[name]) for name in scalar_names},
        "terms": [[r, *midpoint_radius(value)] for r, value in row["terms"]],
    }


def compute_row(tables: TaylorTables, weight: str, k: int,
                theta_s: str, s_s: str, t_s: str | None) -> dict:
    theta, s = Fraction(theta_s), Fraction(s_s)
    beta = s * theta
    Aminus = tables.value("minus", beta)
    omega_minus = Aminus / ae(beta)
    sg = arb(0)
    terms = []
    if weight == "0":
        for r in range(k + 1, s.numerator // s.denominator + 1):
            tr = tables.value("c", s, r)
            sg += tr
            terms.append((r, tr))
        first_integral = arb(0)
    else:
        assert t_s is not None
        t = Fraction(t_s)
        rmax = t.numerator // t.denominator + 1
        lower = t - 1
        upper = t * (1 - Fraction(1, 1) / s)
        for r in range(k + 1, rmax + 1):
            tr0 = tables.value("c", t, r) if r <= t.numerator // t.denominator else arb(0)
            tr1 = ae(t) * tables.integrate_kernel("c", lower, upper, t, r - 1) / 2
            tr = tr0 + tr1
            sg += tr
            terms.append((r, tr))
        alpha = t * theta
        lower = beta * (1 - Fraction(1, 1) / alpha)
        upper = beta - 1
        first_integral = tables.integrate_kernel("plus", lower, upper, beta) / 2
    M0 = ae(s) * omega_minus
    M1 = ae(s) * first_integral
    M2 = arb(2) * sg / ae(theta)
    M = M0 - M1 - M2
    return {
        "weight": weight, "k": k, "theta": theta_s, "s": s_s, "t": t_s,
        "omega_minus": omega_minus, "S_G": sg, "M0": M0,
        "M1": M1, "M2": M2, "M": M, "terms": terms,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=60)
    ap.add_argument("--precision", type=int, default=256)
    ap.add_argument("--output", default="taylor_results.json")
    args = ap.parse_args()
    max_x = 124
    max_r = 122
    tables = TaylorTables(max_x, max_r, args.degree, args.precision)
    results = [compute_row(tables, *row) for row in ROWS]
    payload = {"degree": args.degree, "precision": args.precision,
               "method": "unit-interval midpoint Taylor models with Arb remainders",
               "rows": [serialize_row(row) for row in results]}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in results:
        print(row["weight"], row["k"], "S_G", row["S_G"], "M", row["M"])


if __name__ == "__main__":
    main()
