"""Anomaly score distribution tracking for empirical density comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class DistributionTracker:
    normal_scores: list[float] = field(default_factory=list)
    attack_scores: list[float] = field(default_factory=list)

    def record_score(self, score: float, is_attack: bool) -> None:
        if is_attack:
            self.attack_scores.append(float(score))
        else:
            self.normal_scores.append(float(score))

    def density_data(self, bins: int = 30) -> dict[str, list[float]]:
        all_scores = self.normal_scores + self.attack_scores
        if not all_scores:
            return {"bin_centers": [], "normal_density": [], "attack_density": []}

        min_val = min(all_scores)
        max_val = max(all_scores)
        if min_val == max_val:
            max_val += 1.0

        bin_edges = np.linspace(min_val, max_val, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        norm_counts, _ = np.histogram(self.normal_scores, bins=bin_edges) if self.normal_scores else (np.zeros(bins), None)
        atk_counts, _ = np.histogram(self.attack_scores, bins=bin_edges) if self.attack_scores else (np.zeros(bins), None)

        norm_density = (norm_counts / len(self.normal_scores)).tolist() if self.normal_scores else [0.0] * bins
        atk_density = (atk_counts / len(self.attack_scores)).tolist() if self.attack_scores else [0.0] * bins

        return {
            "bin_centers": [round(float(c), 4) for c in bin_centers],
            "normal_density": [round(float(d), 4) for d in norm_density],
            "attack_density": [round(float(d), 4) for d in atk_density],
        }

