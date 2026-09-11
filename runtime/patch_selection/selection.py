from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .time_constraints import TimeConstraint


@dataclass(frozen=True)
class TemporalSegment:
    start: int
    stop: int
    median_rank: float

    @property
    def units(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class SelectionResult:
    group_indices: torch.Tensor
    token_indices: torch.Tensor
    groups_per_unit: tuple[int, ...]
    segments: tuple[TemporalSegment, ...]
    mode: str
    requested_budget: int
    effective_budget: int


def _rank_for_energy(singular_values: torch.Tensor, energy: float) -> int:
    if singular_values.numel() == 0:
        return 0
    powers = singular_values.square()
    total = powers.sum()
    if not torch.isfinite(total) or total <= 0:
        return 1
    cumulative = torch.cumsum(powers, dim=0) / total
    return min(singular_values.numel(), int(torch.searchsorted(cumulative, energy).item()) + 1)


def _thin_svd_rows(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return singular values and Vh via the smaller row Gram matrix.

    Layer-8 has 144 groups and 1152 channels, so this is the exact thin SVD
    while diagonalizing 144x144 rather than invoking a 144x1152 full driver.
    """

    value = value.float()
    gram = value.matmul(value.T)
    eigenvalues, u = torch.linalg.eigh(gram)
    order = torch.arange(eigenvalues.numel() - 1, -1, -1, device=value.device)
    eigenvalues = eigenvalues.index_select(0, order).clamp_min(0)
    u = u.index_select(1, order)
    singular = eigenvalues.sqrt()
    vh = u.T.matmul(value) / singular.clamp_min(1e-8).unsqueeze(1)
    vh = torch.where(singular[:, None] > 1e-7, vh, torch.zeros_like(vh))
    return singular, vh


def _unit_factors(normalized: torch.Tensor, common_energy: float) -> tuple[list[torch.Tensor], list[int]]:
    factors: list[torch.Tensor] = []
    ranks: list[int] = []
    for unit in normalized:
        singular, vh = _thin_svd_rows(unit)
        rank = max(1, _rank_for_energy(singular, common_energy))
        factor = vh[:rank].T * singular[:rank].unsqueeze(0)
        norm = factor.norm()
        factors.append(factor / norm.clamp_min(1e-12))
        ranks.append(rank)
    return factors, ranks


def _common_basis(factors: Sequence[torch.Tensor], ranks: Sequence[int], common_energy: float) -> torch.Tensor:
    if not factors:
        raise ValueError("At least one unit factor is required")
    state = factors[0].float()
    basis = F.normalize(state, dim=0, eps=1e-8)
    for index, candidate in enumerate(factors[1:], start=1):
        joined = torch.cat((state, candidate.float()), dim=1)
        gram = joined.T.matmul(joined)
        eigenvalues, vectors = torch.linalg.eigh(gram)
        order = torch.arange(eigenvalues.numel() - 1, -1, -1, device=joined.device)
        singular = eigenvalues.index_select(0, order).clamp_min(0).sqrt()
        vectors = vectors.index_select(1, order)
        left = joined.matmul(vectors) / singular.clamp_min(1e-8).unsqueeze(0)
        cap = max(1, int(torch.tensor(list(ranks[: index + 1]), dtype=torch.float32).median().item()))
        keep = min(cap, _rank_for_energy(singular, common_energy))
        basis = left[:, :keep]
        state = basis * singular[:keep].unsqueeze(0)
    return basis


def segment_units(
    features: torch.Tensor,
    *,
    common_energy: float,
    min_common_energy: float,
    min_units: int,
    max_units: int,
) -> tuple[tuple[TemporalSegment, ...], torch.Tensor, tuple[int, ...]]:
    """Adaptive SVD segmentation with a candidate that cannot pollute the preceding segment."""

    if features.ndim != 3:
        raise ValueError("features must be [time, groups, dim]")
    unit_count = features.shape[0]
    if unit_count == 0:
        return (), features.float(), ()
    normalized = F.normalize(features.float(), dim=-1, eps=1e-6)
    factors, ranks = _unit_factors(normalized, common_energy)
    segments: list[TemporalSegment] = []
    start = 0
    accepted_factors = [factors[0]]
    accepted_ranks = [ranks[0]]
    basis = _common_basis(accepted_factors, accepted_ranks, common_energy)
    for index in range(1, unit_count):
        length = index - start
        candidate = factors[index]
        coverage = basis.T.matmul(candidate).square().sum() / candidate.square().sum().clamp_min(1e-12)
        should_split = length >= max_units or (length >= min_units and float(coverage) < min_common_energy)
        if should_split:
            median_rank = float(torch.tensor(accepted_ranks, dtype=torch.float32).median())
            segments.append(TemporalSegment(start, index, median_rank))
            start = index
            accepted_factors = [candidate]
            accepted_ranks = [ranks[index]]
        else:
            accepted_factors.append(candidate)
            accepted_ranks.append(ranks[index])
        basis = _common_basis(accepted_factors, accepted_ranks, common_energy)
    median_rank = float(torch.tensor(accepted_ranks, dtype=torch.float32).median())
    segments.append(TemporalSegment(start, unit_count, median_rank))
    return tuple(segments), normalized, tuple(ranks)


def _largest_remainder(total: int, weights: Sequence[float], minima: Sequence[int], capacities: Sequence[int]) -> list[int]:
    if not (len(weights) == len(minima) == len(capacities)):
        raise ValueError("allocation vectors have different lengths")
    if sum(minima) > total or sum(capacities) < total:
        raise ValueError("infeasible exact allocation")
    allocation = list(minima)
    remaining = total - sum(allocation)
    while remaining:
        active = [i for i in range(len(weights)) if allocation[i] < capacities[i]]
        if not active:
            raise RuntimeError("allocation exhausted all capacities")
        weight_sum = sum(max(0.0, weights[i]) for i in active)
        quotas = {
            i: remaining * (max(0.0, weights[i]) / weight_sum if weight_sum else 1.0 / len(active)) for i in active
        }
        floors = {i: min(capacities[i] - allocation[i], int(math.floor(quotas[i]))) for i in active}
        progressed = sum(floors.values())
        for i, value in floors.items():
            allocation[i] += value
        remaining -= progressed
        if not remaining:
            break
        order = sorted(active, key=lambda i: (quotas[i] - math.floor(quotas[i]), weights[i], -i), reverse=True)
        for i in order:
            if remaining == 0:
                break
            if allocation[i] < capacities[i]:
                allocation[i] += 1
                remaining -= 1
    return allocation


def _percentile(scores: torch.Tensor) -> torch.Tensor:
    flat = scores.flatten()
    if flat.numel() <= 1:
        return torch.ones_like(scores, dtype=torch.float32)
    order = torch.argsort(flat, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(flat.numel(), device=flat.device, dtype=torch.float32)
    return (ranks / (flat.numel() - 1)).reshape_as(scores)


def _basis_and_scores(segment_features: torch.Tensor, common_energy: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = segment_features.reshape(-1, segment_features.shape[-1])
    factors, ranks = _unit_factors(segment_features, common_energy)
    basis = _common_basis(factors, ranks, common_energy)
    coeff = flat.matmul(basis)
    common = coeff.square().sum(-1).reshape(segment_features.shape[:2])
    residual = (flat - coeff.matmul(basis.T)).square().sum(-1).reshape(segment_features.shape[:2])
    return basis, common, residual


def _delta_scores(features: torch.Tensor, basis: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    times, groups, _ = features.shape
    height, width = grid_hw
    if height * width != groups:
        raise ValueError(f"grid {grid_hw} does not contain {groups} groups")
    scores = torch.zeros((times, groups), device=features.device, dtype=torch.float32)
    if times < 2:
        return scores
    coeff = F.normalize(features.matmul(basis).float(), dim=-1, eps=1e-6)
    neighbor_mask = torch.zeros((groups, groups), device=features.device, dtype=torch.bool)
    for index in range(groups):
        y, x = divmod(index, width)
        for yy in range(max(0, y - 1), min(height, y + 2)):
            for xx in range(max(0, x - 1), min(width, x + 2)):
                neighbor_mask[index, yy * width + xx] = True
    for time in range(times - 1):
        left = coeff[time]
        right = coeff[time + 1]
        similarity = left.matmul(right.T).masked_fill(~neighbor_mask, -torch.inf)
        best_right = similarity.argmax(dim=1)
        best_left = similarity.argmax(dim=0)
        left_indices = torch.arange(groups, device=features.device)
        mutual = best_left[best_right] == left_indices
        mutual_left = left_indices[mutual]
        mutual_right = best_right[mutual]
        delta = (left[mutual_left] - right[mutual_right]).square().sum(-1)
        scores[time, mutual_left] = torch.maximum(scores[time, mutual_left], delta)
        scores[time + 1, mutual_right] = torch.maximum(scores[time + 1, mutual_right], delta)
    return scores


def _css_order(features: torch.Tensor, basis: torch.Tensor, limit: int) -> list[int]:
    """Time-stratified greedy CSS order over real projected group vectors."""

    times, groups, _ = features.shape
    projected = features.matmul(basis).float()
    remaining = [set(range(groups)) for _ in range(times)]
    selected_vectors: list[torch.Tensor] = []
    order: list[int] = []
    while len(order) < min(limit, times * groups):
        progressed = False
        for time in range(times):
            if not remaining[time] or len(order) >= limit:
                continue
            candidates = torch.tensor(sorted(remaining[time]), device=features.device)
            vectors = projected[time, candidates]
            if selected_vectors:
                q, _ = torch.linalg.qr(torch.stack(selected_vectors).T, mode="reduced")
                value = (vectors - vectors.matmul(q).matmul(q.T)).square().sum(-1)
            else:
                value = vectors.square().sum(-1)
            local = int(candidates[torch.argmax(value)])
            remaining[time].remove(local)
            order.append(time * groups + local)
            selected_vectors.append(projected[time, local])
            progressed = True
        if not progressed:
            break
    return order


def _round_robin_anchor_groups(anchors: Sequence[int], groups: int, budget: int) -> set[int]:
    chosen: set[int] = set()
    if budget <= 0 or not anchors:
        return chosen
    for group in range(groups):
        for unit in anchors:
            if len(chosen) >= budget:
                return chosen
            chosen.add(unit * groups + group)
    return chosen


class Layer8GroupSelector:
    def __init__(self, config: ExperimentConfig):
        config.validate()
        self.config = config

    def _scope(self, features: torch.Tensor, constraint: TimeConstraint) -> tuple[torch.Tensor, int, tuple[int, ...]]:
        unit_count = features.shape[0]
        bounds = constraint.range_unit_bounds(self.config.temporal_patch_size / self.config.sampling_fps, unit_count)
        if bounds is None:
            return features, 0, constraint.unit_indices(self.config.temporal_patch_size / self.config.sampling_fps, unit_count)
        lo, hi = bounds
        return features[lo : hi + 1], lo, (lo, hi)

    def select(self, features: torch.Tensor, constraint: TimeConstraint | None = None) -> SelectionResult:
        if features.ndim != 3:
            raise ValueError("Layer-8 group features must be [time, groups, hidden]")
        constraint = constraint or TimeConstraint(kind="global")
        scoped, offset, anchor_units = self._scope(features, constraint)
        unit_count, groups, _ = scoped.shape
        total = unit_count * groups
        budget = min(self.config.visual_token_budget, total)
        if unit_count == 0:
            empty = torch.empty(0, dtype=torch.long, device=features.device)
            return SelectionResult(empty, empty, (), (), self.config.selector_mode, self.config.visual_token_budget, 0)
        if self.config.selector_mode == "identity" or total <= budget:
            local = torch.arange(total, device=features.device)
            segments = (TemporalSegment(0, unit_count, 1.0),)
        elif self.config.selector_mode == "uniform":
            positions = torch.linspace(0, total - 1, budget, device=features.device)
            local = torch.unique(positions.round().long(), sorted=True)
            if local.numel() != budget:
                remaining = torch.tensor([i for i in range(total) if i not in set(local.tolist())], device=features.device)
                local = torch.sort(torch.cat([local, remaining[: budget - local.numel()]]))[0]
            segments = (TemporalSegment(0, unit_count, 1.0),)
        else:
            segments, normalized, _ = segment_units(
                scoped,
                common_energy=self.config.common_energy,
                min_common_energy=self.config.min_common_energy,
                min_units=self.config.min_segment_units,
                max_units=self.config.max_segment_units,
            )
            local_anchors = tuple(sorted({unit - offset for unit in anchor_units if offset <= unit < offset + unit_count}))
            mandatory = _round_robin_anchor_groups(local_anchors, groups, min(budget, len(local_anchors) * groups))
            selected = set(mandatory)
            # The contract guarantees at least one real group from every visual time unit.
            for unit in range(unit_count):
                if len(selected) >= budget:
                    break
                if not any(index // groups == unit for index in selected):
                    center = (groups // 2 + unit * 17) % groups
                    selected.add(unit * groups + center)
            remaining_budget = budget - len(selected)
            capacities = []
            minima = []
            for segment in segments:
                occupied = sum(segment.start <= index // groups < segment.stop for index in selected)
                capacities.append(segment.units * groups - occupied)
                minima.append(0)
            weights = [segment.units * math.sqrt(max(1.0, segment.median_rank)) for segment in segments]
            allocations = _largest_remainder(remaining_budget, weights, minima, capacities)
            for segment, allocation in zip(segments, allocations):
                self._select_segment(normalized, segment, allocation, selected, groups)
            if len(selected) != budget:
                raise RuntimeError(f"selector produced {len(selected)} groups for exact budget {budget}")
            local = torch.tensor(sorted(selected), device=features.device, dtype=torch.long)
        global_groups = local + offset * groups
        token_indices = (global_groups[:, None] * 4 + torch.arange(4, device=features.device)).reshape(-1)
        counts = torch.bincount(local // groups, minlength=unit_count).tolist()
        return SelectionResult(
            group_indices=global_groups,
            token_indices=token_indices,
            groups_per_unit=tuple(int(value) for value in counts),
            segments=segments,
            mode=self.config.selector_mode,
            requested_budget=self.config.visual_token_budget,
            effective_budget=int(global_groups.numel()),
        )

    def _select_segment(
        self,
        normalized: torch.Tensor,
        segment: TemporalSegment,
        allocation: int,
        selected: set[int],
        groups: int,
    ) -> None:
        if allocation <= 0:
            return
        values = normalized[segment.start : segment.stop]
        basis, common, residual = _basis_and_scores(values, self.config.common_energy)
        delta = _delta_scores(values, basis, self.config.group_grid)
        candidates = [segment.start * groups + i for i in range(segment.units * groups)]
        available = {index for index in candidates if index not in selected}
        before = len(selected)
        common_n = int(round(allocation * (0.5 if self.config.selector_mode == "svd_cr" else 0.4)))
        residual_n = allocation - common_n if self.config.selector_mode == "svd_cr" else int(round(allocation * 0.3))
        delta_n = allocation - common_n - residual_n

        occupied = segment.units * groups - len(available)
        css = [segment.start * groups + index for index in _css_order(values, basis, common_n + occupied)]
        self._take(css, common_n, available, selected)
        residual_order = self._score_order(residual, segment.start, groups)
        self._take(residual_order, residual_n, available, selected)
        if delta_n:
            delta_order = self._score_order(delta, segment.start, groups)
            self._take(delta_order, delta_n, available, selected)

        missing = allocation - (len(selected) - before)
        if missing:
            combined = torch.maximum(_percentile(common), _percentile(residual))
            if self.config.selector_mode == "svd_crd":
                combined = torch.maximum(combined, _percentile(delta))
            self._take(self._score_order(combined, segment.start, groups), missing, available, selected)
        if len(selected) - before != allocation:
            raise RuntimeError("segment selector could not satisfy its exact allocation")

    @staticmethod
    def _score_order(scores: torch.Tensor, start: int, groups: int) -> list[int]:
        order = torch.argsort(scores.flatten(), descending=True, stable=True).tolist()
        return [start * groups + int(index) for index in order]

    @staticmethod
    def _take(order: Iterable[int], amount: int, available: set[int], selected: set[int]) -> None:
        for index in order:
            if amount <= 0:
                return
            if index in available:
                selected.add(index)
                available.remove(index)
                amount -= 1

