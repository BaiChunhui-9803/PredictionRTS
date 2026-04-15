"""
KG Beam Search — Chain Reasoning Prediction Engine

Performs beam search over the knowledge graph to predict multi-step
state-action trajectories, starting from a given state.

Core idea:
  At each step, expand all active beams by considering candidate actions
  (from KG quality ranking) and their transition probabilities (from
  transitions dict).  Keep only the top-B beams by a configurable scoring
  metric, pruning branches whose cumulative probability drops below a
  threshold.

Usage:
    from src.decision.kg_beam_search import beam_search_predict, find_optimal_action

    # Run beam search
    results = beam_search_predict(kg, transitions, start_state=42,
                                  beam_width=3, max_steps=5)

    # Quick optimal-action recommendation
    action, info = find_optimal_action(kg, transitions, state=42)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from src.decision.knowledge_graph import DecisionKnowledgeGraph


@dataclass
class BeamSearchResult:
    step: int
    state: int
    action: str
    cumulative_probability: float
    quality_score: float
    win_rate: float
    avg_step_reward: float
    avg_future_reward: float
    beam_id: int
    parent_idx: Optional[int]


class _BeamNode:
    __slots__ = (
        "state",
        "action",
        "cum_prob",
        "score",
        "step",
        "win_rate",
        "avg_step_reward",
        "avg_future_reward",
        "path",
        "parent_idx",
        "visited_counts",
    )

    def __init__(
        self,
        state: int,
        action: Optional[str],
        cum_prob: float,
        score: float,
        step: int,
        win_rate: float = 0.0,
        avg_step_reward: float = 0.0,
        avg_future_reward: float = 0.0,
        path: Optional[List[_BeamNode]] = None,
        parent_idx: Optional[int] = None,
        visited_counts: Optional[Dict[int, int]] = None,
    ):
        self.state = state
        self.action = action
        self.cum_prob = cum_prob
        self.score = score
        self.step = step
        self.win_rate = win_rate
        self.avg_step_reward = avg_step_reward
        self.avg_future_reward = avg_future_reward
        self.path = path if path is not None else []
        self.parent_idx = parent_idx
        self.visited_counts = visited_counts if visited_counts is not None else {}


def _get_score(
    quality: Dict[str, Any],
    score_mode: str,
) -> float:
    if score_mode == "future_reward":
        return quality.get("avg_future_reward", 0.0)
    if score_mode == "win_rate":
        return quality.get("win_rate", 0.0)
    return quality.get("quality_score", 0.0)


def beam_search_predict(
    kg: DecisionKnowledgeGraph,
    transitions: Dict[int, Dict[str, Dict]],
    start_state: int,
    beam_width: int = 3,
    max_steps: int = 5,
    min_visits: int = 1,
    min_cum_prob: float = 0.01,
    score_mode: str = "quality",
    max_state_revisits: int = 2,
    discount_factor: float = 0.9,
) -> List[BeamSearchResult]:
    """
    Beam search over the KG transition graph.

    Args:
        kg: DecisionKnowledgeGraph instance.
        transitions: {state: {action: {"next_states": {s: count}, ...}, ...}}
        start_state: Starting state ID.
        beam_width: Number of beams to keep at each step.
        max_steps: Maximum prediction horizon.
        min_visits: Minimum visit count for an action to be considered.
        min_cum_prob: Prune branches whose cumulative probability < threshold.
        score_mode: "quality" | "future_reward" | "win_rate"
        max_state_revisits: Max times a single state can appear in one beam path.
            Set to 1 to completely forbid revisits, 2 to allow one revisit, etc.
        discount_factor: Per-step discount applied to cumulative probability.
            1.0 = no discount, 0.9 = 10% decay per step.

    Returns:
        Flat list of BeamSearchResult for every expanded node (across all beams),
        with beam_id and parent_idx so that callers can reconstruct the tree.
    """
    candidates_per_action: int = max(beam_width * 3, 9)

    all_nodes: List[BeamSearchResult] = []

    def _record(node: _BeamNode, beam_id: int, parent_idx: Optional[int]):
        all_nodes.append(
            BeamSearchResult(
                step=node.step,
                state=node.state,
                action=node.action or "",
                cumulative_probability=round(node.cum_prob, 6),
                quality_score=round(node.score, 4),
                win_rate=round(node.win_rate, 4),
                avg_step_reward=round(node.avg_step_reward, 4),
                avg_future_reward=round(node.avg_future_reward, 4),
                beam_id=beam_id,
                parent_idx=parent_idx,
            )
        )

    root = _BeamNode(
        state=start_state,
        action=None,
        cum_prob=1.0,
        score=0.0,
        step=0,
        path=[],
        parent_idx=None,
        visited_counts={start_state: 1},
    )
    _record(root, beam_id=0, parent_idx=None)

    beam_heads: List[Tuple[int, _BeamNode]] = [(0, root)]

    for step in range(max_steps):
        expanded: List[Tuple[int, _BeamNode]] = []

        for head_idx, beam in beam_heads:
            top_actions = kg.get_top_k_actions(
                state=beam.state,
                k=candidates_per_action,
                min_visits=min_visits,
                metric="quality_score",
            )

            if not top_actions:
                continue

            for action, quality in top_actions:
                state_trans = transitions.get(beam.state, {})
                if not state_trans or state_trans.get("__terminal__"):
                    continue

                trans_info = state_trans.get(action, {})
                if not trans_info:
                    continue

                next_states = trans_info.get("next_states", {})
                if not next_states:
                    continue

                total_count = sum(next_states.values())
                score = _get_score(quality, score_mode)

                for ns, count in sorted(
                    next_states.items(), key=lambda x: x[1], reverse=True
                ):
                    current_visits = beam.visited_counts.get(ns, 0)
                    if current_visits >= max_state_revisits:
                        continue

                    prob = count / total_count if total_count > 0 else 0.0
                    cum_prob = beam.cum_prob * prob * (discount_factor ** (step + 1))
                    if cum_prob < min_cum_prob:
                        continue

                    ns_quality = kg.get_action_quality(beam.state, action)
                    new_visited = dict(beam.visited_counts)
                    new_visited[ns] = new_visited.get(ns, 0) + 1
                    node = _BeamNode(
                        state=ns,
                        action=action,
                        cum_prob=cum_prob,
                        score=score,
                        step=step + 1,
                        win_rate=ns_quality["win_rate"] if ns_quality else 0.0,
                        avg_step_reward=ns_quality["avg_step_reward"]
                        if ns_quality
                        else 0.0,
                        avg_future_reward=ns_quality["avg_future_reward"]
                        if ns_quality
                        else 0.0,
                        path=beam.path + [beam],
                        parent_idx=None,
                        visited_counts=new_visited,
                    )
                    expanded.append((head_idx, node))

        if not expanded:
            break

        expanded.sort(key=lambda x: (x[1].score, x[1].cum_prob), reverse=True)

        trimmed = expanded[:beam_width]

        beam_heads = []
        for parent_head_idx, beam in trimmed:
            new_idx = len(all_nodes)
            beam_id = beam_heads.__len__()
            _record(beam, beam_id=beam_id, parent_idx=parent_head_idx)
            beam_heads.append((new_idx, beam))

    return all_nodes


def find_optimal_action(
    kg: DecisionKnowledgeGraph,
    transitions: Dict[int, Dict[str, Dict]],
    state: int,
    beam_width: int = 3,
    max_steps: int = 5,
    min_visits: int = 1,
    min_cum_prob: float = 0.01,
    score_mode: str = "quality",
    max_state_revisits: int = 2,
    discount_factor: float = 0.9,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Recommend the best first action from *state* using beam search.

    Returns:
        (action, info_dict)  where info_dict contains:
          - recommended_action
          - expected_cumulative_reward: mean avg_future_reward along best beam
          - expected_win_rate: mean win_rate along best beam
          - best_beam_cum_prob: cumulative probability of best beam
          - best_beam_length: number of steps in best beam
          - all_results: full beam search results
    """
    top_actions = kg.get_top_k_actions(
        state=state,
        k=5,
        min_visits=min_visits,
        metric="quality_score",
    )

    if not top_actions:
        return None, {"reason": "no_actions"}

    best_action = top_actions[0][0]
    best_quality = top_actions[0][1]

    results = beam_search_predict(
        kg,
        transitions,
        state,
        beam_width=beam_width,
        max_steps=max_steps,
        min_visits=min_visits,
        min_cum_prob=min_cum_prob,
        score_mode=score_mode,
        max_state_revisits=max_state_revisits,
        discount_factor=discount_factor,
    )

    if not results:
        return best_action, {
            "recommended_action": best_action,
            "expected_cumulative_reward": best_quality.get("avg_future_reward", 0.0),
            "expected_win_rate": best_quality.get("win_rate", 0.0),
            "best_beam_cum_prob": 0.0,
            "best_beam_length": 0,
            "all_results": [],
            "reason": "no_transitions",
        }

    real_paths = get_beam_paths(results)

    if real_paths:
        best_path = max(real_paths, key=lambda p: p[-1].cumulative_probability)
        avg_reward = (
            float(np.mean([r.avg_future_reward for r in best_path[1:]]))
            if len(best_path) > 1
            else 0.0
        )
        avg_wr = (
            float(np.mean([r.win_rate for r in best_path[1:]]))
            if len(best_path) > 1
            else 0.0
        )
        cum_prob = best_path[-1].cumulative_probability
        beam_len = len(best_path) - 1
    else:
        avg_reward = best_quality.get("avg_future_reward", 0.0)
        avg_wr = best_quality.get("win_rate", 0.0)
        cum_prob = 0.0
        beam_len = 0

    return best_action, {
        "recommended_action": best_action,
        "expected_cumulative_reward": round(avg_reward, 4),
        "expected_win_rate": round(avg_wr, 4),
        "best_beam_cum_prob": round(cum_prob, 6),
        "best_beam_length": beam_len,
        "all_results": results,
    }


def get_beam_paths(
    results: List[BeamSearchResult],
) -> List[List[BeamSearchResult]]:
    """
    Trace actual root-to-leaf paths via parent_idx.

    Returns:
        List of paths, each path is [root, step1, step2, ..., leaf].
    """
    if not results:
        return []

    parent_indices = {r.parent_idx for r in results if r.parent_idx is not None}
    all_indices = set(range(len(results)))
    leaf_indices = sorted(all_indices - parent_indices)

    paths: List[List[BeamSearchResult]] = []
    for leaf_idx in leaf_indices:
        path: List[BeamSearchResult] = []
        current = leaf_idx
        while current is not None:
            path.append(results[current])
            current = results[current].parent_idx
        path.reverse()
        paths.append(path)

    return paths
