"""Finite hidden-slot privacy accountant over a cyclic payload domain.

The recurrence in this module is mathematically exact for the stated finite
model.  Its scalable implementation uses IEEE-754 arithmetic and therefore
reports separate numerical diagnostics; it must not be described as exact
rational evaluation.  The independent tiny-instance oracle lives in
``exhaustive_hidden_slot_oracle.py`` and uses ``fractions.Fraction``.

Model
-----
Background records are independent, with records iid within each declared
type.  Type ``t`` has ``n_t`` records and a payload law ``f_t`` on Z_K.  Every
record independently chooses one of N slots uniformly.  Under add/remove adjacency, Q contains only
the background records and P additionally contains one target whose payload
law is ``f_star``.  A pair may optionally supply a Q-side target law, which
gives replacement adjacency.

Drop emits the payload of a slot containing exactly one record.  Pool emits
the sum modulo K of every payload in a nonempty slot.  The decoder output is
padded with iid values from an arbitrary fixed public law nu to exactly M
values and uniformly permuted.  Because the final sequence law is
exchangeable, the count vector on Z_K is a sufficient statistic: all ordered
sequences with the same count vector have the same probability under P and Q,
and their likelihood ratio equals the count-vector likelihood ratio.

For L exchangeable remaining slots and a type-count vector r, the counts C_t
entering a distinguished slot are independent Binomial(r_t, 1/L).  Conditional
on C, the remaining records are again uniform over L-1 slots.  This gives the
finite coefficient/DP recurrence implemented below; no Poisson approximation
is used.  When M >= N, slotwise padding is exactly equivalent and reduces the
state size: every suppressed/empty slot contributes one nu dummy, followed by
M-N additional dummies.
"""

from __future__ import annotations

import itertools
import math
import operator
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


Histogram = tuple[int, ...]
FloatDistribution = dict[Histogram, float]
FloatLaw = tuple[float, ...]


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
class NeighborPair:
    """One finite adjacent pair, possibly with heterogeneous background types."""

    name: str
    background_counts: tuple[int, ...]
    background_laws: tuple[FloatLaw, ...]
    target_law_p: FloatLaw
    target_law_q: FloatLaw | None = None
    family_scope: str = "pair_specific"

    @classmethod
    def homogeneous(
        cls,
        name: str,
        n: int,
        background_law: Sequence[float],
        target_law_p: Sequence[float],
        target_law_q: Sequence[float] | None = None,
        family_scope: str = "pair_specific",
    ) -> "NeighborPair":
        checked_n = _require_integer_dimension(n, "n")
        return cls(
            name=name,
            background_counts=(checked_n,),
            background_laws=(tuple(float(x) for x in background_law),),
            target_law_p=tuple(float(x) for x in target_law_p),
            target_law_q=(
                tuple(float(x) for x in target_law_q)
                if target_law_q is not None
                else None
            ),
            family_scope=family_scope,
        )

    @property
    def adjacency(self) -> str:
        return "add_remove" if self.target_law_q is None else "replacement"

    @property
    def background_size(self) -> int:
        return sum(self.background_counts)


@dataclass
class ArithmeticDiagnostics:
    arithmetic_operations: int = 0
    normalization_operations: int = 0
    normalization_residual_l1_sum: float = 0.0
    max_normalization_mass_error: float = 0.0
    underflowed_positive_terms: int = 0
    final_p_mass_error: float = 0.0
    final_q_mass_error: float = 0.0

    def record_normalization(self, mass: float, support_size: int, label: str) -> None:
        error = abs(mass - 1.0)
        self.normalization_operations += 1
        self.normalization_residual_l1_sum += error
        self.max_normalization_mass_error = max(
            self.max_normalization_mass_error, error
        )
        self.arithmetic_operations += max(1, support_size)
        if label == "full_P":
            self.final_p_mass_error = error
        elif label == "full_Q":
            self.final_q_mass_error = error

    def l1_error_diagnostic(self) -> float:
        """Conservative L1 diagnostic under a simplified IEEE error model.

        Nonnegative stochastic convolutions are L1-nonexpansive.  We add every
        observed renormalization adjustment and a deliberately loose 32*u per
        counted arithmetic operation.  This is not an outward-rounded interval
        proof because exp/lgamma are evaluated by the platform math library.
        Fraction-oracle rows provide the formal zero-error certificate.
        """
        if self.underflowed_positive_terms:
            return 2.0
        unit_roundoff = sys.float_info.epsilon / 2.0
        bound = self.normalization_residual_l1_sum
        bound += 32.0 * unit_roundoff * self.arithmetic_operations
        return min(2.0, max(0.0, bound))


@dataclass
class PairAccountResult:
    pair_name: str
    adjacency: str
    P: FloatDistribution
    Q: FloatDistribution
    alpha_nominal: float
    epsilon_nominal: float
    alpha_float_guarded: float
    epsilon_float_guarded: float
    delta: float
    profile_at_nominal: float
    profile_at_float_guarded: float
    pure_alpha_common_support: float
    singular_delta: float
    float_l1_error_diagnostic_each_distribution: float
    float_hockey_stick_error_diagnostic: float
    diagnostics: ArithmeticDiagnostics


@dataclass
class FamilyAccountResult:
    decoder: str
    N: int
    M: int
    K: int
    delta: float
    family_scope: str
    pair_results: list[PairAccountResult]
    worst_pair_nominal: str
    worst_pair_float_guarded: str
    alpha_nominal: float
    epsilon_nominal: float
    alpha_float_guarded: float
    epsilon_float_guarded: float


def _validate_law(law: Sequence[float], K: int | None, name: str) -> FloatLaw:
    values = tuple(float(x) for x in law)
    if K is not None and len(values) != K:
        raise ValueError(f"{name} has length {len(values)}, expected {K}")
    if not values:
        raise ValueError(f"{name} is empty")
    if any(not math.isfinite(x) or x < 0.0 for x in values):
        raise ValueError(f"{name} has a negative or non-finite probability")
    mass = math.fsum(values)
    if abs(mass - 1.0) > 1e-10:
        raise ValueError(f"{name} has mass {mass}, expected one")
    if mass <= 0.0:
        raise ValueError(f"{name} has zero mass")
    return tuple(x / mass for x in values)


def cyclic_convolution(left: Sequence[float], right: Sequence[float]) -> FloatLaw:
    if len(left) != len(right):
        raise ValueError("cyclic laws must use the same domain")
    K = len(left)
    out = [0.0] * K
    for i, p in enumerate(left):
        for j, q in enumerate(right):
            out[(i + j) % K] += float(p) * float(q)
    mass = math.fsum(out)
    return tuple(x / mass for x in out)


def cyclic_powers(law: Sequence[float], maximum: int) -> list[FloatLaw]:
    maximum = _require_integer_dimension(maximum, "maximum")
    K = len(law)
    identity = tuple(1.0 if i == 0 else 0.0 for i in range(K))
    powers = [identity]
    base = tuple(float(x) for x in law)
    for _ in range(maximum):
        powers.append(cyclic_convolution(powers[-1], base))
    return powers


def finite_matched_dummy_law(
    decoder: str,
    n: int,
    N: int,
    background_law: Sequence[float],
) -> FloatLaw:
    """Finite iid-background reference law for a real decoded slot."""
    n = _require_integer_dimension(n, "n", minimum=1)
    N = _require_integer_dimension(N, "N", minimum=1)
    law = _validate_law(background_law, None, "background_law")
    if decoder == "drop":
        return law
    if decoder != "pool":
        raise ValueError("decoder must be 'drop' or 'pool'")
    powers = cyclic_powers(law, n)
    count_probabilities = _binomial_probabilities(n, 1.0 / N, None)
    nonempty = 1.0 - count_probabilities[0]
    if nonempty <= 0.0:
        raise ValueError("nonempty slot probability is zero")
    out = [0.0] * len(law)
    for count in range(1, n + 1):
        weight = count_probabilities[count] / nonempty
        for value, probability in enumerate(powers[count]):
            out[value] += weight * probability
    mass = math.fsum(out)
    return tuple(x / mass for x in out)


def enumerate_type_count_vectors(total: int, num_types: int) -> list[tuple[int, ...]]:
    total = _require_integer_dimension(total, "total")
    num_types = _require_integer_dimension(num_types, "num_types", minimum=1)
    if num_types == 1:
        return [(total,)]
    out = []
    for first in range(total + 1):
        for suffix in enumerate_type_count_vectors(total - first, num_types - 1):
            out.append((first,) + suffix)
    return out


def _validated_input_type_laws(
    input_type_laws: Sequence[Sequence[float]],
) -> tuple[FloatLaw, ...]:
    if not input_type_laws:
        raise ValueError("input_type_laws is empty")
    K = len(input_type_laws[0])
    return tuple(
        _validate_law(law, K, f"input_type_law_{idx}")
        for idx, law in enumerate(input_type_laws)
    )


def _fixed_background_add_remove_family(
    background_size: int,
    input_type_laws: Sequence[Sequence[float]],
    name_prefix: str,
    family_scope: str,
) -> list[NeighborPair]:
    background_size = _require_integer_dimension(
        background_size, "background_size"
    )
    laws = _validated_input_type_laws(input_type_laws)
    pairs = []
    for counts in enumerate_type_count_vectors(background_size, len(laws)):
        counts_text = "_".join(str(value) for value in counts)
        for target_type, target_law in enumerate(laws):
            pairs.append(
                NeighborPair(
                    name=f"{name_prefix}_counts_{counts_text}_target_{target_type}",
                    background_counts=counts,
                    background_laws=laws,
                    target_law_p=target_law,
                    family_scope=family_scope,
                )
            )
    return pairs


def fixed_background_add_remove_pair_family(
    background_size: int,
    input_type_laws: Sequence[Sequence[float]],
    name_prefix: str = "add_remove_fixed_background",
) -> list[NeighborPair]:
    """Exhaust one explicitly fixed add/remove background size."""
    return _fixed_background_add_remove_family(
        background_size,
        input_type_laws,
        name_prefix,
        "fixed_background_size_add_remove",
    )


def all_background_sizes_add_remove_pair_family(
    max_background_size: int,
    input_type_laws: Sequence[Sequence[float]],
    name_prefix: str = "add_remove_all_sizes",
) -> list[NeighborPair]:
    """Exhaust every add/remove background size from zero through the cap."""
    maximum = _require_integer_dimension(
        max_background_size, "max_background_size"
    )
    pairs: list[NeighborPair] = []
    for background_size in range(maximum + 1):
        pairs.extend(
            _fixed_background_add_remove_family(
                background_size,
                input_type_laws,
                f"{name_prefix}_n_{background_size}",
                "all_background_sizes_add_remove",
            )
        )
    return pairs


def fixed_size_replacement_pair_family(
    public_size: int,
    input_type_laws: Sequence[Sequence[float]],
    name_prefix: str = "replacement_fixed_size",
) -> list[NeighborPair]:
    """Exhaust fixed-size backgrounds and ordered distinct replacement types."""
    size = _require_integer_dimension(public_size, "public_size", minimum=1)
    laws = _validated_input_type_laws(input_type_laws)
    pairs: list[NeighborPair] = []
    for counts in enumerate_type_count_vectors(size - 1, len(laws)):
        counts_text = "_".join(str(value) for value in counts)
        for q_type, q_law in enumerate(laws):
            for p_type, p_law in enumerate(laws):
                if p_type == q_type:
                    continue
                pairs.append(
                    NeighborPair(
                        name=(
                            f"{name_prefix}_size_{size}_counts_{counts_text}_"
                            f"target_{q_type}_to_{p_type}"
                        ),
                        background_counts=counts,
                        background_laws=laws,
                        target_law_p=p_law,
                        target_law_q=q_law,
                        family_scope="fixed_public_size_replacement",
                    )
                )
    return pairs


def all_sizes_replacement_pair_family(
    max_public_size: int,
    input_type_laws: Sequence[Sequence[float]],
    name_prefix: str = "replacement_all_sizes",
) -> list[NeighborPair]:
    """Exhaust replacement families for every public size from one to the cap."""
    maximum = _require_integer_dimension(
        max_public_size, "max_public_size", minimum=1
    )
    pairs: list[NeighborPair] = []
    for size in range(1, maximum + 1):
        for pair in fixed_size_replacement_pair_family(
            size, input_type_laws, f"{name_prefix}_size_{size}"
        ):
            pairs.append(
                NeighborPair(
                    name=pair.name,
                    background_counts=pair.background_counts,
                    background_laws=pair.background_laws,
                    target_law_p=pair.target_law_p,
                    target_law_q=pair.target_law_q,
                    family_scope="all_public_sizes_replacement",
                )
            )
    return pairs


def _binomial_probabilities(
    n: int,
    p: float,
    diagnostics: ArithmeticDiagnostics | None,
) -> tuple[float, ...]:
    n = _require_integer_dimension(n, "n")
    if not 0.0 <= p <= 1.0:
        raise ValueError("invalid binomial parameters")
    if p == 0.0:
        return (1.0,) + (0.0,) * n
    if p == 1.0:
        return (0.0,) * n + (1.0,)
    logs = []
    for count in range(n + 1):
        value = (
            math.lgamma(n + 1)
            - math.lgamma(count + 1)
            - math.lgamma(n - count + 1)
            + count * math.log(p)
            + (n - count) * math.log1p(-p)
        )
        logs.append(value)
    maximum = max(logs)
    weights = []
    for value in logs:
        weight = math.exp(value - maximum)
        if weight == 0.0 and math.isfinite(value):
            if diagnostics is not None:
                diagnostics.underflowed_positive_terms += 1
        weights.append(weight)
    mass = math.fsum(weights)
    if diagnostics is not None:
        diagnostics.arithmetic_operations += 12 * (n + 1)
    return tuple(weight / mass for weight in weights)


def _add_histograms(left: Histogram, right: Histogram) -> Histogram:
    return tuple(a + b for a, b in zip(left, right))


def _unit_histogram(K: int, value: int) -> Histogram:
    return tuple(1 if index == value else 0 for index in range(K))


class _FiniteRecurrence:
    def __init__(
        self,
        pair: NeighborPair,
        decoder: str,
        N: int,
        M: int,
        nu: Sequence[float],
    ) -> None:
        if decoder not in {"drop", "pool"}:
            raise ValueError("decoder must be 'drop' or 'pool'")
        checked_N = _require_integer_dimension(N, "N", minimum=1)
        checked_M = _require_integer_dimension(M, "M")
        if len(pair.background_counts) != len(pair.background_laws):
            raise ValueError("background counts/laws have different lengths")
        checked_counts = tuple(
            _require_integer_dimension(count, f"background_counts[{index}]")
            for index, count in enumerate(pair.background_counts)
        )
        self.decoder = decoder
        self.N = checked_N
        self.M = checked_M
        self.counts = checked_counts
        self.K = len(nu)
        self.nu = _validate_law(nu, self.K, "nu")
        self.laws = tuple(
            _validate_law(law, self.K, f"background_law_{idx}")
            for idx, law in enumerate(pair.background_laws)
        )
        self.target_p = _validate_law(pair.target_law_p, self.K, "target_law_p")
        self.target_q = (
            _validate_law(pair.target_law_q, self.K, "target_law_q")
            if pair.target_law_q is not None
            else None
        )
        maximum_real = min(self.N, sum(self.counts) + 1)
        if self.M < maximum_real:
            raise ValueError(
                f"M={self.M} cannot hold the maximum {maximum_real} real outputs"
            )
        self.zero_hist = (0,) * self.K
        self.slotwise_padding = self.M >= self.N
        self.diagnostics = ArithmeticDiagnostics()
        self._binomial_cache: dict[tuple[int, int], tuple[float, ...]] = {}
        self.count_vectors = list(
            itertools.product(*(range(count + 1) for count in self.counts))
        )
        self.power_tables = [
            cyclic_powers(law, count)
            for law, count in zip(self.laws, self.counts)
        ]
        self._standard_emissions = {
            vector: self._background_emission(vector) for vector in self.count_vectors
        }
        self._dummy_histogram_cache: dict[int, FloatDistribution] = {
            0: {self.zero_hist: 1.0}
        }
        self.row_n_minus_1, self.row_n = self._build_standard_rows()

    def _normalize(
        self,
        distribution: Mapping[Histogram, float],
        label: str,
    ) -> FloatDistribution:
        cleaned: FloatDistribution = {}
        for outcome, probability in distribution.items():
            value = float(probability)
            if value < -1e-15 or not math.isfinite(value):
                raise ArithmeticError(f"{label} contains invalid probability {value}")
            if value > 0.0:
                cleaned[outcome] = max(0.0, value)
        mass = math.fsum(cleaned.values())
        if mass <= 0.0:
            raise ArithmeticError(f"{label} has zero mass")
        self.diagnostics.record_normalization(mass, len(cleaned), label)
        return {outcome: probability / mass for outcome, probability in cleaned.items()}

    def _background_aggregate_law(self, counts: tuple[int, ...]) -> FloatLaw:
        law = tuple(1.0 if value == 0 else 0.0 for value in range(self.K))
        for type_index, count in enumerate(counts):
            if count:
                law = cyclic_convolution(law, self.power_tables[type_index][count])
                self.diagnostics.arithmetic_operations += 3 * self.K * self.K
        return law

    def _background_emission(self, counts: tuple[int, ...]) -> FloatLaw | None:
        total = sum(counts)
        if self.decoder == "drop":
            if total != 1:
                return self.nu if self.slotwise_padding else None
            type_index = next(index for index, count in enumerate(counts) if count)
            return self.laws[type_index]
        if total == 0:
            return self.nu if self.slotwise_padding else None
        return self._background_aggregate_law(counts)

    def _target_emission(
        self,
        counts: tuple[int, ...],
        target_law: FloatLaw,
    ) -> FloatLaw | None:
        total = sum(counts)
        if self.decoder == "drop":
            if total == 0:
                return target_law
            return self.nu if self.slotwise_padding else None
        background = self._background_aggregate_law(counts)
        law = cyclic_convolution(target_law, background)
        self.diagnostics.arithmetic_operations += 3 * self.K * self.K
        return law

    def _merge_emission(
        self,
        base: Mapping[Histogram, float],
        emission: FloatLaw | None,
        weight: float,
        output: defaultdict[Histogram, float],
    ) -> None:
        if weight == 0.0:
            return
        if emission is None:
            for histogram, probability in base.items():
                output[histogram] += weight * probability
                self.diagnostics.arithmetic_operations += 2
            return
        for histogram, probability in base.items():
            common = weight * probability
            for value, emission_probability in enumerate(emission):
                if emission_probability == 0.0:
                    continue
                updated = list(histogram)
                updated[value] += 1
                output[tuple(updated)] += common * emission_probability
                self.diagnostics.arithmetic_operations += 3

    @staticmethod
    def _subvectors(vector: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
        return itertools.product(*(range(value + 1) for value in vector))

    def _count_vector_probability(
        self,
        remaining: tuple[int, ...],
        selected: tuple[int, ...],
        slots: int,
    ) -> float:
        probability = 1.0
        for total, count in zip(remaining, selected):
            cache_key = (total, slots)
            probabilities = self._binomial_cache.get(cache_key)
            if probabilities is None:
                probabilities = _binomial_probabilities(
                    total, 1.0 / slots, self.diagnostics
                )
                self._binomial_cache[cache_key] = probabilities
            probability *= probabilities[count]
            self.diagnostics.arithmetic_operations += 1
        return probability

    def _build_standard_rows(
        self,
    ) -> tuple[
        dict[tuple[int, ...], FloatDistribution],
        dict[tuple[int, ...], FloatDistribution],
    ]:
        zero_counts = (0,) * len(self.counts)
        previous: dict[tuple[int, ...], FloatDistribution] = {
            zero_counts: {self.zero_hist: 1.0}
        }
        row_n_minus_1 = previous if self.N == 1 else None
        for slots in range(1, self.N + 1):
            current: dict[tuple[int, ...], FloatDistribution] = {}
            for remaining in self.count_vectors:
                output: defaultdict[Histogram, float] = defaultdict(float)
                for selected_iter in self._subvectors(remaining):
                    selected = tuple(selected_iter)
                    rest = tuple(a - b for a, b in zip(remaining, selected))
                    base = previous.get(rest)
                    if not base:
                        continue
                    weight = self._count_vector_probability(
                        remaining, selected, slots
                    )
                    emission = self._standard_emissions[selected]
                    self._merge_emission(base, emission, weight, output)
                current[remaining] = self._normalize(
                    output, f"standard_slots_{slots}_counts_{remaining}"
                )
            previous = current
            if slots == self.N - 1:
                row_n_minus_1 = current
        if row_n_minus_1 is None:
            raise AssertionError("failed to retain N-1 standard-slot row")
        return row_n_minus_1, previous

    def _target_base_distribution(self, target_law: FloatLaw) -> FloatDistribution:
        output: defaultdict[Histogram, float] = defaultdict(float)
        for selected_iter in self._subvectors(self.counts):
            selected = tuple(selected_iter)
            rest = tuple(a - b for a, b in zip(self.counts, selected))
            base = self.row_n_minus_1.get(rest)
            if not base:
                continue
            weight = self._count_vector_probability(self.counts, selected, self.N)
            emission = self._target_emission(selected, target_law)
            self._merge_emission(base, emission, weight, output)
        return self._normalize(output, "target_base")

    def _dummy_histogram_distribution(self, count: int) -> FloatDistribution:
        count = _require_integer_dimension(count, "dummy_count")
        cached = self._dummy_histogram_cache.get(count)
        if cached is not None:
            return cached
        current: FloatDistribution = {self.zero_hist: 1.0}
        for _ in range(count):
            output: defaultdict[Histogram, float] = defaultdict(float)
            for histogram, probability in current.items():
                for value, dummy_probability in enumerate(self.nu):
                    if dummy_probability == 0.0:
                        continue
                    updated = list(histogram)
                    updated[value] += 1
                    output[tuple(updated)] += probability * dummy_probability
                    self.diagnostics.arithmetic_operations += 2
            current = dict(output)
        result = self._normalize(current, f"dummy_final_count_{count}")
        self._dummy_histogram_cache[count] = result
        return result

    def _finish_padding(
        self,
        base_distribution: Mapping[Histogram, float],
        label: str,
    ) -> FloatDistribution:
        output: defaultdict[Histogram, float] = defaultdict(float)
        for real_histogram, real_probability in base_distribution.items():
            current_length = sum(real_histogram)
            dummy_count = self.M - current_length
            if dummy_count < 0:
                raise ValueError("decoded output exceeds M")
            dummy_distribution = self._dummy_histogram_distribution(dummy_count)
            for dummy_histogram, dummy_probability in dummy_distribution.items():
                final = _add_histograms(real_histogram, dummy_histogram)
                output[final] += real_probability * dummy_probability
                self.diagnostics.arithmetic_operations += self.K + 2
        result = self._normalize(output, label)
        if any(sum(histogram) != self.M for histogram in result):
            raise AssertionError("final output histogram does not have length M")
        return result

    def output_distributions(self) -> tuple[FloatDistribution, FloatDistribution]:
        q_base = (
            self.row_n[self.counts]
            if self.target_q is None
            else self._target_base_distribution(self.target_q)
        )
        p_base = self._target_base_distribution(self.target_p)
        Q = self._finish_padding(q_base, "full_Q")
        P = self._finish_padding(p_base, "full_P")
        return P, Q


def _normalized_distribution(
    distribution: Mapping[Histogram, float],
) -> FloatDistribution:
    values = {
        outcome: float(probability)
        for outcome, probability in distribution.items()
        if float(probability) > 0.0
    }
    mass = math.fsum(values.values())
    if mass <= 0.0:
        raise ValueError("distribution has zero mass")
    return {outcome: probability / mass for outcome, probability in values.items()}


def hockey_stick_alpha(
    P: Mapping[Histogram, float],
    Q: Mapping[Histogram, float],
    alpha: float,
) -> float:
    if alpha < 1.0:
        raise ValueError("alpha must be at least one")
    p = _normalized_distribution(P)
    q = _normalized_distribution(Q)
    return math.fsum(
        max(probability - alpha * q.get(outcome, 0.0), 0.0)
        for outcome, probability in p.items()
    )


def two_sided_profile_alpha(
    P: Mapping[Histogram, float],
    Q: Mapping[Histogram, float],
    alpha: float,
) -> float:
    return max(hockey_stick_alpha(P, Q, alpha), hockey_stick_alpha(Q, P, alpha))


def _one_sided_min_alpha(
    P: Mapping[Histogram, float],
    Q: Mapping[Histogram, float],
    delta: float,
) -> float:
    p = _normalized_distribution(P)
    q = _normalized_distribution(Q)
    at_one = hockey_stick_alpha(p, q, 1.0)
    if at_one <= delta:
        return 1.0
    singular = math.fsum(
        probability for outcome, probability in p.items() if q.get(outcome, 0.0) == 0.0
    )
    if singular > delta:
        return math.inf
    terms: list[tuple[float, float, float]] = []
    for outcome, probability in p.items():
        q_probability = q.get(outcome, 0.0)
        if q_probability > 0.0:
            ratio = probability / q_probability
            if ratio > 1.0:
                terms.append((ratio, probability, q_probability))
    terms.sort(key=lambda item: item[0])
    active_p = math.fsum(item[1] for item in terms)
    active_q = math.fsum(item[2] for item in terms)
    index = 0
    lower = 1.0
    while index < len(terms):
        breakpoint = terms[index][0]
        if active_q > 0.0:
            candidate = (singular + active_p - delta) / active_q
            if candidate <= breakpoint:
                return max(lower, candidate, 1.0)
        while index < len(terms) and terms[index][0] == breakpoint:
            active_p -= terms[index][1]
            active_q -= terms[index][2]
            index += 1
        lower = breakpoint
    return lower


def two_sided_min_alpha(
    P: Mapping[Histogram, float],
    Q: Mapping[Histogram, float],
    delta: float,
) -> float:
    if not 0.0 <= delta < 1.0:
        raise ValueError("delta must lie in [0,1)")
    alpha = max(
        _one_sided_min_alpha(P, Q, delta),
        _one_sided_min_alpha(Q, P, delta),
    )
    return _repair_alpha_upward(P, Q, alpha, delta)


def _repair_alpha_upward(
    P: Mapping[Histogram, float],
    Q: Mapping[Histogram, float],
    alpha: float,
    delta: float,
) -> float:
    """Advance a rounded alpha until its evaluated profile is feasible.

    The active-prefix solution is algebraically exact for the supplied
    floating probabilities, but its final division can round downward.  A
    privacy accountant must return a feasible upper endpoint, so we repair by
    stepping toward +infinity.  The multiplicative fallback is only reached
    if more than 4096 adjacent floats would be required.
    """
    if not math.isfinite(alpha):
        return alpha
    repaired = max(1.0, alpha)
    for _ in range(4096):
        if two_sided_profile_alpha(P, Q, repaired) <= delta:
            return repaired
        repaired = math.nextafter(repaired, math.inf)
    increment = max(repaired * 1e-12, 1e-15)
    for _ in range(128):
        repaired = math.nextafter(repaired + increment, math.inf)
        if two_sided_profile_alpha(P, Q, repaired) <= delta:
            return repaired
        increment *= 2.0
    raise ArithmeticError("could not repair alpha to a feasible floating endpoint")


def _singular_delta(
    P: Mapping[Histogram, float], Q: Mapping[Histogram, float]
) -> float:
    p = _normalized_distribution(P)
    q = _normalized_distribution(Q)
    forward = math.fsum(value for key, value in p.items() if q.get(key, 0.0) == 0.0)
    reverse = math.fsum(value for key, value in q.items() if p.get(key, 0.0) == 0.0)
    return max(forward, reverse)


def _common_support_alpha_bound(
    P: Mapping[Histogram, float], Q: Mapping[Histogram, float]
) -> float:
    p = _normalized_distribution(P)
    q = _normalized_distribution(Q)
    bound = 1.0
    for outcome in set(p) | set(q):
        pp = p.get(outcome, 0.0)
        qq = q.get(outcome, 0.0)
        if pp > 0.0 and qq > 0.0:
            bound = max(bound, pp / qq, qq / pp)
    return bound


def account_pair(
    pair: NeighborPair,
    decoder: str,
    N: int,
    M: int,
    nu: Sequence[float],
    delta: float,
) -> PairAccountResult:
    recurrence = _FiniteRecurrence(pair, decoder, N, M, nu)
    P, Q = recurrence.output_distributions()
    alpha_nominal = two_sided_min_alpha(P, Q, delta)
    epsilon_nominal = math.log(alpha_nominal) if math.isfinite(alpha_nominal) else math.inf
    profile_nominal = (
        two_sided_profile_alpha(P, Q, alpha_nominal)
        if math.isfinite(alpha_nominal)
        else _singular_delta(P, Q)
    )
    ratio_bound = _common_support_alpha_bound(P, Q)
    l1_diagnostic = recurrence.diagnostics.l1_error_diagnostic()
    profile_error = min(1.0, (1.0 + ratio_bound) * l1_diagnostic)
    if profile_error >= delta or recurrence.diagnostics.underflowed_positive_terms:
        alpha_float_guarded = math.inf
        epsilon_float_guarded = math.inf
        profile_float_guarded = math.nan
    else:
        alpha_float_guarded = two_sided_min_alpha(P, Q, delta - profile_error)
        epsilon_float_guarded = (
            math.log(alpha_float_guarded)
            if math.isfinite(alpha_float_guarded)
            else math.inf
        )
        profile_float_guarded = (
            two_sided_profile_alpha(P, Q, alpha_float_guarded)
            if math.isfinite(alpha_float_guarded)
            else _singular_delta(P, Q)
        )
    return PairAccountResult(
        pair_name=pair.name,
        adjacency=pair.adjacency,
        P=P,
        Q=Q,
        alpha_nominal=alpha_nominal,
        epsilon_nominal=epsilon_nominal,
        alpha_float_guarded=alpha_float_guarded,
        epsilon_float_guarded=epsilon_float_guarded,
        delta=delta,
        profile_at_nominal=profile_nominal,
        profile_at_float_guarded=profile_float_guarded,
        pure_alpha_common_support=ratio_bound,
        singular_delta=_singular_delta(P, Q),
        float_l1_error_diagnostic_each_distribution=l1_diagnostic,
        float_hockey_stick_error_diagnostic=profile_error,
        diagnostics=recurrence.diagnostics,
    )


def account_pair_family(
    pairs: Sequence[NeighborPair],
    decoder: str,
    N: int,
    M: int,
    nu: Sequence[float],
    delta: float,
) -> FamilyAccountResult:
    if not pairs:
        raise ValueError("pair family is empty")
    N = _require_integer_dimension(N, "N", minimum=1)
    M = _require_integer_dimension(M, "M")
    scopes = {pair.family_scope for pair in pairs}
    family_scope = next(iter(scopes)) if len(scopes) == 1 else "explicit_mixed_scope"
    K = len(nu)
    results = [account_pair(pair, decoder, N, M, nu, delta) for pair in pairs]
    worst_nominal = max(results, key=lambda result: result.alpha_nominal)
    worst_float_guarded = max(results, key=lambda result: result.alpha_float_guarded)
    return FamilyAccountResult(
        decoder=decoder,
        N=N,
        M=M,
        K=K,
        delta=delta,
        family_scope=family_scope,
        pair_results=results,
        worst_pair_nominal=worst_nominal.pair_name,
        worst_pair_float_guarded=worst_float_guarded.pair_name,
        alpha_nominal=worst_nominal.alpha_nominal,
        epsilon_nominal=worst_nominal.epsilon_nominal,
        alpha_float_guarded=worst_float_guarded.alpha_float_guarded,
        epsilon_float_guarded=worst_float_guarded.epsilon_float_guarded,
    )
