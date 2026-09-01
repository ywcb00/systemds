# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------

from typing import Callable, Iterable, Optional

import numpy as np


def _operator_signature(entry) -> frozenset:
    """Set of operator class names in an entry's DAG, ignoring hyperparameters."""
    try:
        return frozenset(
            node.operation.__name__
            for node in entry.dag.nodes
            if getattr(node, "operation", None) is not None
        )
    except Exception:
        return frozenset()


def _dag_size(entry) -> int:
    try:
        return sum(
            1
            for node in entry.dag.nodes
            if getattr(node, "operation", None) is not None
        )
    except Exception:
        return 0


def rank_by_robustness(
    entries: Iterable,
    *,
    performance_metric_name: str = "accuracy",
    neighbourhood_weight: float = 0.5,
    sharpness: int = 4,
    one_se_parsimony: bool = True,
    cache_scores: bool = True,
    score_attr: str = "robustness_score",
):
    entries = list(entries)
    if not entries:
        return [], []

    def perf_of(entry):
        if entry is None:
            return None
        try:
            score = float(entry.val_score[performance_metric_name])
        except (KeyError, TypeError, ValueError):
            return None
        return score if np.isfinite(score) else None

    indexed_entries = [
        (index, entry, score)
        for index, entry in enumerate(entries)
        if (score := perf_of(entry)) is not None
    ]
    if not indexed_entries:
        return [], []

    original_indices, entries, performance = zip(*indexed_entries)
    entries = list(entries)
    perf = np.array(performance, dtype=float)
    sizes = np.array([_dag_size(e) if e is not None else 0 for e in entries], float)

    smoothed = perf
    if neighbourhood_weight > 0.0 and len(entries) > 1:
        signatures = [
            _operator_signature(e) if e is not None else frozenset() for e in entries
        ]
        vocabulary = sorted({op for sig in signatures for op in sig})
        if vocabulary:
            position = {op: i for i, op in enumerate(vocabulary)}
            membership = np.zeros((len(entries), len(vocabulary)), dtype=np.float32)
            for row, sig in enumerate(signatures):
                for op in sig:
                    membership[row, position[op]] = 1.0
            intersection = membership @ membership.T
            counts = membership.sum(1)
            union = counts[:, None] + counts[None, :] - intersection
            jaccard = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            weights = jaccard**sharpness
            denominator = weights.sum(1)
            neighbourhood = np.divide(
                (weights * perf[None, :].astype(np.float32)).sum(1),
                denominator,
                out=perf.astype(np.float32).copy(),
                where=denominator > 0,
            )
            smoothed = (
                1.0 - neighbourhood_weight
            ) * perf + neighbourhood_weight * neighbourhood

    if cache_scores:
        for entry, score in zip(entries, smoothed):
            if entry is not None:
                setattr(entry, score_attr, float(score))

    if one_se_parsimony and len(entries) > 1:
        standard_error = float(smoothed.std()) / np.sqrt(len(smoothed))
        threshold = float(smoothed.max()) - standard_error
        keys = [
            (True, -sz, float(s)) if s >= threshold else (False, 0.0, float(s))
            for s, sz in zip(smoothed, sizes)
        ]
    else:
        keys = [(True, 0.0, float(s)) for s in smoothed]

    local_indices = sorted(range(len(entries)), key=lambda i: keys[i], reverse=True)
    sorted_entries = [entries[i] for i in local_indices]
    sorted_indices = [original_indices[i] for i in local_indices]

    return sorted_entries, sorted_indices


def rank_by_tradeoff(
    entries: Iterable,
    *,
    weights=(1.0, 0.0),
    performance_metric_name: str = "accuracy",
    runtime_accessor: Optional[Callable[[object], float]] = None,
    cache_scores: bool = True,
    score_attr: str = "tradeoff_score",
):
    entries = list(entries)
    if not entries:
        return [], []

    performance_score_accessor = lambda entry: getattr(entry, "val_score")[
        performance_metric_name
    ]

    if runtime_accessor is None:

        def runtime_accessor(entry):
            if hasattr(entry, "runtime"):
                return getattr(entry, "runtime")
            rep = getattr(entry, "representation_time", 0.0)
            task = getattr(entry, "task_time", 0.0)
            return rep + task

    performance = [
        float(performance_score_accessor(e)) if e is not None else 0.0 for e in entries
    ]
    runtimes = [float(runtime_accessor(e)) if e is not None else 0.0 for e in entries]

    perf_min, perf_max = min(performance), max(performance)
    run_min, run_max = min(runtimes), max(runtimes)

    def safe_normalize(values, vmin, vmax):
        if vmax - vmin == 0.0:
            return [1.0] * len(values)
        return [(v - vmin) / (vmax - vmin) for v in values]

    norm_perf = safe_normalize(performance, perf_min, perf_max)
    norm_run = safe_normalize(runtimes, run_min, run_max)
    norm_run = [1.0 - r for r in norm_run]

    acc_w, run_w = weights
    total_w = (acc_w or 0.0) + (run_w or 0.0)
    if total_w == 0.0:
        acc_w = 1.0
        run_w = 0.0
    else:
        acc_w /= total_w
        run_w /= total_w

    scores = [acc_w * a + run_w * r for a, r in zip(norm_perf, norm_run)]

    if cache_scores:
        for entry, score in zip(entries, scores):
            if entry is None:
                continue
            if hasattr(entry, score_attr):
                setattr(entry, score_attr, score)
            else:
                setattr(entry, score_attr, score)

    sorted_entries = sorted(
        entries,
        key=lambda e: e.tradeoff_score if hasattr(e, "tradeoff_score") else 0.0,
        reverse=True,
    )

    sorted_indices = [
        i
        for i, _ in sorted(
            enumerate(entries),
            key=lambda pair: pair[1].tradeoff_score if pair is not None else None,
            reverse=True,
        )
    ]

    return sorted_entries, sorted_indices
