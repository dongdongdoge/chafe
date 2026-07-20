"""Independent exact-rational oracle for tiny hidden-slot instances.

This module intentionally does not call the coefficient recurrence.  It
enumerates every labeled background placement/payload realization, every
target placement/payload realization, and every dummy sequence.  Probabilities
are ``fractions.Fraction`` objects throughout.  It is exponential and is only
an oracle for tiny instances.
"""

from __future__ import annotations

import itertools
import math
import operator
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence


Histogram = tuple[int, ...]
RationalLaw = tuple[Fraction, ...]
RationalDistribution = dict[Histogram, Fraction]


def _require_integer_dimension(
    value: object,
    name: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(normalized)


@dataclass(frozen=True)
class RationalNeighborPair:
    name: str
    background_counts: tuple[int, ...]
    background_laws: tuple[RationalLaw, ...]
    target_law_p: RationalLaw
    target_law_q: RationalLaw | None = None
    family_scope: str = "pair_specific"

    @classmethod
    def homogeneous(
        cls,
        name: str,
        n: int,
        background_law: Sequence[Fraction],
        target_law_p: Sequence[Fraction],
        target_law_q: Sequence[Fraction] | None = None,
        family_scope: str = "pair_specific",
    ) -> "RationalNeighborPair":
        checked_n = _require_integer_dimension(n, "n")
        return cls(
            name=name,
            background_counts=(checked_n,),
            background_laws=(tuple(background_law),),
            target_law_p=tuple(target_law_p),
            target_law_q=(tuple(target_law_q) if target_law_q is not None else None),
            family_scope=family_scope,
        )


def _validate_law(law: Sequence[Fraction], K: int | None, name: str) -> RationalLaw:
    values = tuple(Fraction(value) for value in law)
    if K is not None and len(values) != K:
        raise ValueError(f"{name} has length {len(values)}, expected {K}")
    if not values or any(value < 0 for value in values):
        raise ValueError(f"{name} is empty or negative")
    if sum(values, Fraction(0)) != 1:
        raise ValueError(f"{name} does not have exact unit mass")
    return values


def _histogram(values: Sequence[int], K: int) -> Histogram:
    counts = [0] * K
    for value in values:
        counts[value] += 1
    return tuple(counts)


def cyclic_convolution_fraction(
    left: Sequence[Fraction], right: Sequence[Fraction]
) -> RationalLaw:
    if len(left) != len(right):
        raise ValueError("cyclic laws must use the same domain")
    K = len(left)
    output = [Fraction(0)] * K
    for i, p in enumerate(left):
        for j, q in enumerate(right):
            output[(i + j) % K] += Fraction(p) * Fraction(q)
    if sum(output, Fraction(0)) != 1:
        raise AssertionError("rational cyclic convolution lost mass")
    return tuple(output)


def finite_matched_dummy_law_fraction(
    decoder: str,
    n: int,
    N: int,
    background_law: Sequence[Fraction],
) -> RationalLaw:
    law = _validate_law(background_law, None, "background_law")
    n = _require_integer_dimension(n, "n", minimum=1)
    N = _require_integer_dimension(N, "N", minimum=1)
    if decoder == "drop":
        return law
    if decoder != "pool":
        raise ValueError("invalid decoder or occupancy parameters")
    K = len(law)
    identity = tuple(Fraction(1) if value == 0 else Fraction(0) for value in range(K))
    powers = [identity]
    for _ in range(n):
        powers.append(cyclic_convolution_fraction(powers[-1], law))
    empty = Fraction(N - 1, N) ** n
    nonempty = 1 - empty
    output = [Fraction(0)] * K
    for count in range(1, n + 1):
        count_probability = (
            Fraction(math.comb(n, count))
            * Fraction(1, N) ** count
            * Fraction(N - 1, N) ** (n - count)
        )
        weight = count_probability / nonempty
        for value, probability in enumerate(powers[count]):
            output[value] += weight * probability
    if sum(output, Fraction(0)) != 1:
        raise AssertionError("rational matched dummy law lost mass")
    return tuple(output)


def _decode(slots: Sequence[Sequence[int]], decoder: str, K: int) -> tuple[int, ...]:
    outputs: list[int] = []
    for payloads in slots:
        if decoder == "drop":
            if len(payloads) == 1:
                outputs.append(payloads[0])
        elif decoder == "pool":
            if payloads:
                outputs.append(sum(payloads) % K)
        else:
            raise ValueError("decoder must be 'drop' or 'pool'")
    return tuple(outputs)


def _dummy_histogram_distribution(
    count: int,
    nu: RationalLaw,
) -> RationalDistribution:
    count = _require_integer_dimension(count, "dummy_count")
    K = len(nu)
    output: defaultdict[Histogram, Fraction] = defaultdict(Fraction)
    for sequence in itertools.product(range(K), repeat=count):
        probability = Fraction(1)
        for value in sequence:
            probability *= nu[value]
        output[_histogram(sequence, K)] += probability
    result = dict(output)
    if sum(result.values(), Fraction(0)) != 1:
        raise AssertionError("dummy enumeration lost probability mass")
    return result


def _background_assignments(
    pair: RationalNeighborPair,
    N: int,
    K: int,
):
    option_lists = []
    for type_index, count in enumerate(pair.background_counts):
        law = pair.background_laws[type_index]
        options = [
            (slot, value, Fraction(1, N) * law[value])
            for slot in range(N)
            for value in range(K)
            if law[value]
        ]
        option_lists.extend([options] * count)
    for assignment in itertools.product(*option_lists):
        slots = [[] for _ in range(N)]
        probability = Fraction(1)
        for slot, value, item_probability in assignment:
            slots[slot].append(value)
            probability *= item_probability
        yield slots, probability


def _target_options(law: RationalLaw, N: int):
    for slot in range(N):
        for value, probability in enumerate(law):
            if probability:
                yield slot, value, Fraction(1, N) * probability


def exhaustive_output_distribution(
    pair: RationalNeighborPair,
    decoder: str,
    N: int,
    M: int,
    nu: Sequence[Fraction],
    side: str,
) -> RationalDistribution:
    N = _require_integer_dimension(N, "N", minimum=1)
    M = _require_integer_dimension(M, "M")
    if len(pair.background_counts) != len(pair.background_laws):
        raise ValueError("background counts/laws have different lengths")
    checked_counts = tuple(
        _require_integer_dimension(count, f"background_counts[{index}]")
        for index, count in enumerate(pair.background_counts)
    )
    K = len(nu)
    nu_law = _validate_law(nu, K, "nu")
    laws = tuple(
        _validate_law(law, K, f"background_law_{index}")
        for index, law in enumerate(pair.background_laws)
    )
    checked_pair = RationalNeighborPair(
        name=pair.name,
        background_counts=checked_counts,
        background_laws=laws,
        target_law_p=_validate_law(pair.target_law_p, K, "target_law_p"),
        target_law_q=(
            _validate_law(pair.target_law_q, K, "target_law_q")
            if pair.target_law_q is not None
            else None
        ),
        family_scope=pair.family_scope,
    )
    if side == "P":
        target_law = checked_pair.target_law_p
    elif side == "Q":
        target_law = checked_pair.target_law_q
    else:
        raise ValueError("side must be P or Q")

    dummy_cache = {
        count: _dummy_histogram_distribution(count, nu_law)
        for count in range(M + 1)
    }
    output: defaultdict[Histogram, Fraction] = defaultdict(Fraction)

    def add_decoded(real_values: tuple[int, ...], state_probability: Fraction) -> None:
        dummy_count = M - len(real_values)
        if dummy_count < 0:
            raise ValueError("decoded output exceeds M")
        real_histogram = _histogram(real_values, K)
        for dummy_histogram, dummy_probability in dummy_cache[dummy_count].items():
            final = tuple(a + b for a, b in zip(real_histogram, dummy_histogram))
            output[final] += state_probability * dummy_probability

    for background_slots, background_probability in _background_assignments(
        checked_pair, N, K
    ):
        if target_law is None:
            add_decoded(_decode(background_slots, decoder, K), background_probability)
            continue
        for target_slot, target_value, target_probability in _target_options(
            target_law, N
        ):
            slots = [list(payloads) for payloads in background_slots]
            slots[target_slot].append(target_value)
            add_decoded(
                _decode(slots, decoder, K),
                background_probability * target_probability,
            )

    result = dict(output)
    if sum(result.values(), Fraction(0)) != 1:
        raise AssertionError(f"{side} exhaustive distribution lost probability mass")
    if any(sum(histogram) != M for histogram in result):
        raise AssertionError("exhaustive output is not length M")
    return result


def exhaustive_neighbor_distributions(
    pair: RationalNeighborPair,
    decoder: str,
    N: int,
    M: int,
    nu: Sequence[Fraction],
) -> tuple[RationalDistribution, RationalDistribution]:
    P = exhaustive_output_distribution(pair, decoder, N, M, nu, "P")
    Q = exhaustive_output_distribution(pair, decoder, N, M, nu, "Q")
    return P, Q


def hockey_stick_alpha_fraction(
    P: Mapping[Histogram, Fraction],
    Q: Mapping[Histogram, Fraction],
    alpha: Fraction,
) -> Fraction:
    if alpha < 1:
        raise ValueError("alpha must be at least one")
    return sum(
        (
            probability - alpha * Q.get(outcome, Fraction(0))
            if probability > alpha * Q.get(outcome, Fraction(0))
            else Fraction(0)
        )
        for outcome, probability in P.items()
    )


def _one_sided_min_alpha_fraction(
    P: Mapping[Histogram, Fraction],
    Q: Mapping[Histogram, Fraction],
    delta: Fraction,
) -> Fraction | None:
    if hockey_stick_alpha_fraction(P, Q, Fraction(1)) <= delta:
        return Fraction(1)
    singular = sum(
        probability
        for outcome, probability in P.items()
        if Q.get(outcome, Fraction(0)) == 0
    )
    if singular > delta:
        return None
    terms = []
    for outcome, probability in P.items():
        q_probability = Q.get(outcome, Fraction(0))
        if q_probability > 0 and probability > q_probability:
            terms.append((probability / q_probability, probability, q_probability))
    terms.sort(key=lambda item: item[0])
    active_p = sum((item[1] for item in terms), Fraction(0))
    active_q = sum((item[2] for item in terms), Fraction(0))
    lower = Fraction(1)
    index = 0
    while index < len(terms):
        breakpoint = terms[index][0]
        if active_q:
            candidate = (singular + active_p - delta) / active_q
            if candidate <= breakpoint:
                return max(lower, candidate, Fraction(1))
        while index < len(terms) and terms[index][0] == breakpoint:
            active_p -= terms[index][1]
            active_q -= terms[index][2]
            index += 1
        lower = breakpoint
    return lower


def two_sided_min_alpha_fraction(
    P: Mapping[Histogram, Fraction],
    Q: Mapping[Histogram, Fraction],
    delta: Fraction,
) -> Fraction | None:
    if not Fraction(0) <= delta < Fraction(1):
        raise ValueError("delta must lie in [0,1)")
    forward = _one_sided_min_alpha_fraction(P, Q, delta)
    reverse = _one_sided_min_alpha_fraction(Q, P, delta)
    if forward is None or reverse is None:
        return None
    return max(forward, reverse)


def _log_positive_integer(value: int) -> float:
    if value <= 0:
        raise ValueError("log input must be positive")
    bits = value.bit_length()
    if bits <= 53:
        return math.log(value)
    shift = bits - 53
    leading = value >> shift
    return math.log(leading) + shift * math.log(2.0)


def log_fraction_safe(value: Fraction) -> float:
    fraction = Fraction(value)
    if fraction <= 0:
        raise ValueError("fraction log input must be positive")
    return _log_positive_integer(fraction.numerator) - _log_positive_integer(
        fraction.denominator
    )


def epsilon_from_alpha_fraction(alpha: Fraction | None) -> float:
    return log_fraction_safe(alpha) if alpha is not None else math.inf
