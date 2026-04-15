#!/usr/bin/env python
"""
Experience Transition Graph Interactive Web Visualizer

Launch:
    streamlit run scripts/visualize_kg_web.py

Dependencies:
    pip install pyvis
"""

import sys
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set, Tuple

import base64
import io
import json
import time

import subprocess
import requests
import yaml
import streamlit as st
import numpy as np
import networkx as nx
from pyvis.network import Network
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata
from sklearn.manifold import MDS

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ROOT_DIR
from src.decision.knowledge_graph import DecisionKnowledgeGraph
from src.decision.kg_beam_search import (
    beam_search_predict,
    find_optimal_action,
    get_beam_paths,
    BeamSearchResult,
)
from src.decision.beam_matcher import match_beam_paths
from src.decision.chain_rollout import chain_rollout, RolloutResult, RolloutNode

KG_DIR = ROOT_DIR / "cache" / "knowledge_graph"
NPY_DIR = ROOT_DIR / "cache" / "npy"
DATA_DIR = ROOT_DIR / "data"


@st.cache_data
def load_kg_catalog() -> List[Dict]:
    path = ROOT_DIR / "configs" / "kg_catalog.yaml"
    if not path.exists():
        st.error(f"经验转移图目录不存在: {path}")
        st.stop()
    with open(path, encoding="utf-8") as f:
        catalog = yaml.safe_load(f)
    return catalog.get("knowledge_graphs", [])


@st.cache_data
def load_kg(kg_file: str) -> Tuple[Dict, float, float]:
    path = KG_DIR / kg_file

    if not path.exists():
        st.error(
            f"文件不存在: {path}\n请先运行 `python scripts/build_knowledge_graph.py`"
        )
        st.stop()

    with open(path, "rb") as f:
        kg_data = pickle.load(f)

    quality_scores = [
        stats.quality_score
        for actions in kg_data["state_action_map"].values()
        for stats in actions.values()
    ]
    quality_min = min(quality_scores) if quality_scores else -60.0
    quality_max = max(quality_scores) if quality_scores else 80.0

    return kg_data, quality_min, quality_max


@st.cache_data
def load_transitions(transitions_file: str) -> Dict:
    path = KG_DIR / transitions_file
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_kg_object(kg_file: str) -> DecisionKnowledgeGraph:
    path = KG_DIR / kg_file
    return DecisionKnowledgeGraph.load(str(path))


@st.cache_resource
def load_episode_data(map_id: str, data_id: str) -> Dict:
    import csv as _csv

    base = DATA_DIR / map_id / data_id
    states, actions, outcomes, scores = [], [], [], []

    node_log = base / "graph" / "node_log.txt"
    if node_log.exists():
        with open(node_log, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    states.append([int(p) for p in parts])

    action_log = base / "action_log.csv"
    if action_log.exists():
        with open(action_log, "r") as f:
            reader = _csv.reader(f)
            for row in reader:
                if row:
                    raw = row[0]
                    actions.append([raw[j : j + 2] for j in range(0, len(raw), 2)])

    result_file = base / "game_result.txt"
    if result_file.exists():
        with open(result_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                outcomes.append(parts[0] if parts else "Unknown")
                try:
                    sc = float(parts[2]) if len(parts) > 2 else 0.0
                    pn = float(parts[3]) if len(parts) > 3 else 0.0
                    scores.append(sc + pn)
                except (ValueError, IndexError):
                    scores.append(0.0)

    return {
        "states": states,
        "actions": actions,
        "outcomes": outcomes,
        "scores": scores,
        "n_episodes": len(states),
    }


@st.cache_resource
def load_distance_matrix_np(map_id: str, data_id: str):
    path = NPY_DIR / f"state_distance_matrix_{map_id}_{data_id}.npy"
    if path.exists():
        return np.load(str(path))
    return None


@st.cache_resource
def load_state_hp_data(map_id: str, data_id: str) -> Dict:
    import json as _json

    base = DATA_DIR / map_id / data_id
    sn_file = base / "graph" / "state_node.txt"

    reverse_dict: Dict = {}
    if sn_file.exists():
        with open(sn_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                key = eval(parts[0])
                sid = int(parts[1])
                score = float(parts[2])
                if sid not in reverse_dict:
                    reverse_dict[sid] = {"cluster": key, "score": score}

    primary_file = base / "bktree" / "primary_bktree.json"
    primary_cluster_ids = []
    if primary_file.exists():
        with open(primary_file, "r") as f:
            root = _json.load(f)
        _collect_cluster_ids(root, primary_cluster_ids)

    sub_node_map: Dict[Tuple[int, int], Dict] = {}

    def _search_sub(node):
        cid = node["cluster_id"]
        state = node.get("state", {})
        sub_node_map[(primary_id, cid)] = state
        for child in node.get("children", {}).values():
            _search_sub(child)

    for primary_id in primary_cluster_ids:
        sec_file = base / "bktree" / f"secondary_bktree_{primary_id}.json"
        if not sec_file.exists():
            continue
        try:
            with open(sec_file, "r") as f:
                sec_root = _json.load(f)
            _search_sub(sec_root)
        except Exception:
            pass

    hp_lookup: Dict[int, Dict] = {}
    for sid, info in reverse_dict.items():
        cluster = info["cluster"]
        if isinstance(cluster, tuple) and len(cluster) == 2:
            state_data = sub_node_map.get(cluster, {})
            if state_data:
                hp_lookup[sid] = {
                    "cluster": cluster,
                    "score": info["score"],
                    "red_army": state_data.get("red_army", []),
                    "blue_army": state_data.get("blue_army", []),
                }

    return {
        "reverse_dict": reverse_dict,
        "hp_lookup": hp_lookup,
        "n_states": len(reverse_dict),
    }


def _collect_cluster_ids(node, result_list):
    result_list.append(node["cluster_id"])
    for child in node.get("children", {}).values():
        _collect_cluster_ids(child, result_list)


def build_graph_data(
    kg_data: Dict,
    transitions: Dict,
    min_visits: int = 1,
    min_quality: float = -100,
    max_quality: float = 100,
    max_nodes: int = 200,
    focus_state: Optional[int] = None,
    focus_hops: int = 2,
    focus_forward: bool = True,
    focus_backward: bool = True,
) -> Tuple[List[Dict], List[Dict], Dict[int, Dict], bool]:
    sam = kg_data["state_action_map"]

    state_stats: Dict[int, Dict] = {}
    for key, actions in sam.items():
        state_id = key if isinstance(key, int) else key[0]
        total_visits = sum(a.visits for a in actions.values())
        best_quality = max((a.quality_score for a in actions.values()), default=0)
        best_win_rate = max((a.win_rate for a in actions.values()), default=0)
        n_actions = len(actions)
        state_stats[state_id] = {
            "total_visits": total_visits,
            "best_quality": best_quality,
            "best_win_rate": best_win_rate,
            "n_actions": n_actions,
            "is_terminal": state_id not in transitions
            or transitions[state_id].get("__terminal__", False),
        }

    all_states: Set[int] = set(state_stats.keys())

    if focus_state is not None:
        if focus_state not in all_states:
            return [], [], state_stats, False
        selected = set()
        frontier = {focus_state}
        for _ in range(focus_hops):
            next_frontier = set()
            for s in frontier:
                if focus_forward and s in transitions:
                    if transitions[s].get("__terminal__"):
                        pass
                    else:
                        for a_info in transitions[s].values():
                            for ns in a_info.get("next_states", {}):
                                if ns in all_states:
                                    next_frontier.add(ns)
                if focus_backward:
                    for other_s in all_states:
                        if other_s in transitions and not transitions[other_s].get(
                            "__terminal__"
                        ):
                            for a_info in transitions[other_s].values():
                                if (
                                    s in a_info.get("next_states", {})
                                    and other_s not in selected
                                ):
                                    next_frontier.add(other_s)
            selected |= frontier
            frontier = next_frontier - selected
        selected |= frontier
        filtered_states = selected & all_states
    else:
        filtered_states = all_states

    edges: List[Dict] = []
    for state_id in filtered_states:
        key = state_id if not kg_data["use_context"] else (state_id, ())
        actions = sam.get(key, {})
        for action, stats in actions.items():
            if stats.visits < min_visits:
                continue
            if stats.quality_score < min_quality or stats.quality_score > max_quality:
                continue

            if state_id in transitions and action in transitions[state_id]:
                next_states = transitions[state_id][action].get("next_states", {})
                total_count = sum(next_states.values())
                for next_state, count in next_states.items():
                    prob = count / total_count if total_count > 0 else 0
                    edges.append(
                        {
                            "from": state_id,
                            "to": next_state,
                            "action": action,
                            "visits": stats.visits,
                            "quality_score": stats.quality_score,
                            "win_rate": stats.win_rate,
                            "avg_step_reward": stats.avg_step_reward,
                            "avg_future_reward": stats.avg_future_reward,
                            "transition_count": count,
                            "transition_prob": prob,
                        }
                    )
            else:
                edges.append(
                    {
                        "from": state_id,
                        "to": -1,
                        "action": action,
                        "visits": stats.visits,
                        "quality_score": stats.quality_score,
                        "win_rate": stats.win_rate,
                        "avg_step_reward": stats.avg_step_reward,
                        "avg_future_reward": stats.avg_future_reward,
                        "transition_count": 0,
                        "transition_prob": 0,
                    }
                )

    valid_edges = [e for e in edges if e["to"] in all_states]

    if len(filtered_states) > max_nodes:
        sorted_states = sorted(
            filtered_states,
            key=lambda s: state_stats[s]["total_visits"],
            reverse=True,
        )[:max_nodes]
        top_set = set(sorted_states)
        valid_edges = [
            e for e in valid_edges if e["from"] in top_set and e["to"] in top_set
        ]
        filtered_states = top_set

    nodes = []
    for s in filtered_states:
        st_info = state_stats.get(s, {})
        nodes.append(
            {
                "id": s,
                "total_visits": st_info.get("total_visits", 0),
                "best_quality": st_info.get("best_quality", 0),
                "best_win_rate": st_info.get("best_win_rate", 0),
                "n_actions": st_info.get("n_actions", 0),
                "is_terminal": st_info.get("is_terminal", False),
            }
        )

    final_state_ids = {n["id"] for n in nodes}
    valid_edges = [
        e
        for e in valid_edges
        if e["from"] in final_state_ids and e["to"] in final_state_ids
    ]

    return nodes, valid_edges, state_stats, True


def render_pyvis_html(
    nodes: List[Dict],
    edges: List[Dict],
    state_stats: Dict[int, Dict],
    highlight_terminal: bool = True,
    edge_smooth_type: str = "continuous",
    edge_roundness: float = 0.25,
    layout_algorithm: str = "barnes_hut",
    freeze_layout: bool = False,
    canvas_height: int = 750,
) -> str:
    from pyvis.network import Network

    net = Network(
        height="100%",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
        notebook=False,
        select_menu=False,
        filter_menu=False,
    )

    _LAYOUT_DEFAULT_PARAMS = {
        "barnes_hut": {
            "gravity": -8000,
            "central_gravity": 0.3,
            "spring_length": 250,
            "spring_strength": 0.001,
            "damping": 0.09,
            "overlap": 0,
        },
        "force_atlas_2based": {
            "gravity": -50,
            "central_gravity": 0.01,
            "spring_length": 100,
            "spring_strength": 0.08,
            "damping": 0.4,
            "overlap": 0,
        },
        "repulsion": {
            "node_distance": 100,
            "central_gravity": 0.2,
            "spring_length": 200,
            "spring_strength": 0.05,
            "damping": 0.09,
        },
        "hrepulsion": {
            "node_distance": 120,
            "central_gravity": 0.0,
            "spring_length": 100,
            "spring_strength": 0.01,
            "damping": 0.09,
        },
    }
    net.set_options("""
    {
      "tooltip": {
        "style": "white-space: pre-line; font-family: monospace; background: #222; color: #fff; padding: 8px; border-radius: 4px;"
      }
    }
    """)

    from pyvis.physics import Physics

    physics = Physics()
    physics_fn = {
        "barnes_hut": physics.use_barnes_hut,
        "force_atlas_2based": physics.use_force_atlas_2based,
        "repulsion": physics.use_repulsion,
        "hrepulsion": physics.use_hrepulsion,
    }
    physics_fn[layout_algorithm](_LAYOUT_DEFAULT_PARAMS[layout_algorithm])
    net.options["physics"] = json.loads(physics.to_json())
    if freeze_layout:
        net.options["physics"]["enabled"] = False

    smooth_opt: dict = {"type": edge_smooth_type}
    if edge_smooth_type == "manual_arc":
        smooth_opt["roundness"] = edge_roundness
    net.options["edges"] = {"smooth": smooth_opt}

    max_visits = max((n["total_visits"] for n in nodes), default=1) or 1
    max_edge_visits = max((e["visits"] for e in edges), default=1) or 1

    edge_pair_count = defaultdict(int)
    edge_pair_index = defaultdict(int)
    for e in edges:
        edge_pair_count[(e["from"], e["to"])] += 1

    for n in nodes:
        sid = n["id"]
        visits = n["total_visits"]
        win_rate = n["best_win_rate"]
        quality = n["best_quality"]

        size = 10 + 30 * (visits / max_visits)

        is_terminal = n.get("is_terminal", False)
        node_shape = "dot"

        if is_terminal:
            color = "rgba(255, 80, 80, 0.9)"
            node_shape = "diamond"
        elif highlight_terminal and win_rate >= 0.5:
            color = f"rgba({int(255 * (1 - win_rate))}, {int(200 + 55 * win_rate)}, 80, 0.9)"
        elif highlight_terminal and win_rate <= 0.1 and visits > 5:
            color = f"rgba(255, {int(80 + 100 * win_rate)}, 80, 0.9)"
        else:
            r = int(100 + 155 * (1 - max(0, min(1, (win_rate + 0.2) / 1.2))))
            g = int(100 + 155 * max(0, min(1, (win_rate + 0.2) / 1.2)))
            color = f"rgba({r}, {g}, 180, 0.85)"

        title_parts = [
            f"<b>State {sid}</b>",
            f"Total Visits: {visits}",
            f"Available Actions: {n['n_actions']}",
            f"Best Win Rate: {win_rate * 100:.1f}%",
            f"Best Quality: {quality:.1f}",
        ]
        if is_terminal:
            title_parts.insert(1, "⚠️ Terminal State")
        title = "\n".join(title_parts)

        net.add_node(
            sid,
            label=str(sid),
            shape=node_shape if is_terminal else "dot",
            size=size if not is_terminal else size * 1.3,
            color=color,
            title=title,
            borderWidth=3 if is_terminal else 2,
            borderWidthSelected=4 if is_terminal else 4,
        )

    for e in edges:
        width = 1 + 4 * (e["visits"] / max_edge_visits)
        quality_norm = max(0, min(1, (e["quality_score"] + 40) / 60))
        r = int(255 * (1 - quality_norm))
        g = int(100 + 155 * quality_norm)
        b = int(150 + 105 * quality_norm)
        edge_color = f"rgba({r}, {g}, {b}, 0.7)"

        label = f"{e['action']} ({e['transition_prob'] * 100:.0f}%)"
        hover = "\n".join(
            [
                f"{e['from']} -> {e['to']}",
                f"Action: {e['action']}",
                f"Visits: {e['visits']}",
                f"Quality: {e['quality_score']:.1f}",
                f"Win Rate: {e['win_rate'] * 100:.1f}%",
                f"Avg Step Reward: {e['avg_step_reward']:.2f}",
                f"Avg Future Reward: {e['avg_future_reward']:.2f}",
                f"Transition Prob: {e['transition_prob'] * 100:.1f}%",
            ]
        )

        if edge_smooth_type == "continuous":
            smooth = {"type": "continuous"}
        else:
            key = (e["from"], e["to"])
            total = edge_pair_count[key]
            idx = edge_pair_index[key]
            edge_pair_index[key] += 1
            if total <= 1:
                smooth = {"type": "continuous"}
            else:
                roundness = (0.15 + 0.7 * idx / max(total - 1, 1)) * edge_roundness
                direction = "curvedCW" if idx % 2 == 0 else "curvedCCW"
                smooth = {"type": direction, "roundness": roundness}

        net.add_edge(
            e["from"],
            e["to"],
            label=label,
            title=hover,
            width=width,
            color=edge_color,
            arrows="to",
            font={"size": 8, "color": "#aaaaaa", "align": "middle"},
            smooth=smooth,
        )

    raw_html = net.generate_html()
    resize_js = """
<script>
(function() {
    document.querySelectorAll('center').forEach(function(el) { el.remove(); });
    document.documentElement.style.cssText = 'height:100%; margin:0; overflow:hidden;';
    document.body.style.cssText = 'height:100%; margin:0; padding:0; overflow:hidden;';
    var card = document.querySelector('.card');
    if (card) card.style.cssText = 'height:100%; width:100%; padding:0; margin:0;';
    var c = document.getElementById('mynetwork');
    if (!c) return;
    c.style.cssText = 'width:100%; height:100%; position:relative; float:none; border:none;';
    window.addEventListener('resize', function() {
        if (window.network) {
            window.network.setSize('100%', '100%');
            window.network.redraw();
        }
    });
    setTimeout(function() {
        if (window.network) {
            window.network.setSize('100%', '100%');
            window.network.redraw();
        }
    }, 1500);
})();
</script>"""
    return raw_html.replace("</body>", resize_js + "</body>")


_BEAM_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
]

_COMPOSITE_WEIGHTS = (0.30, 0.30, 0.30, 0.10)


def _compute_path_metrics(path: List[BeamSearchResult]) -> Dict[str, Any]:
    non_root = path[1:] if len(path) > 1 else path
    length = len(non_root)
    if length == 0:
        return {
            "length": 0,
            "first_action": "",
            "first_state": path[0].state if path else None,
            "last_state": path[0].state if path else None,
            "confidence": 1.0,
            "avg_quality": 0.0,
            "avg_win_rate": 0.0,
            "last_quality": path[0].quality_score if path else 0.0,
            "last_win_rate": path[0].win_rate if path else 0.0,
            "q_trend": "→",
            "wr_trend": "→",
            "avg_reward": 0.0,
        }
    cum_prob = path[-1].cumulative_probability
    avg_quality = float(np.mean([r.quality_score for r in non_root]))
    avg_win_rate = float(np.mean([r.win_rate for r in non_root]))
    avg_reward = float(np.mean([r.avg_future_reward for r in non_root]))
    last_quality = path[-1].quality_score
    last_win_rate = path[-1].win_rate
    mid = length // 2
    if mid > 0:
        q_first = np.mean([r.quality_score for r in non_root[:mid]])
        q_second = np.mean([r.quality_score for r in non_root[mid:]])
        wr_first = np.mean([r.win_rate for r in non_root[:mid]])
        wr_second = np.mean([r.win_rate for r in non_root[mid:]])
        q_trend = (
            "↑"
            if q_second > q_first * 1.05
            else ("↓" if q_second < q_first * 0.95 else "→")
        )
        wr_trend = (
            "↑"
            if wr_second > wr_first * 1.05
            else ("↓" if wr_second < wr_first * 0.95 else "→")
        )
    else:
        q_trend = "→"
        wr_trend = "→"
    return {
        "length": length,
        "first_action": path[1].action if len(path) > 1 else "",
        "first_state": path[1].state if len(path) > 1 else path[0].state,
        "last_state": path[-1].state,
        "confidence": cum_prob,
        "avg_quality": avg_quality,
        "avg_win_rate": avg_win_rate,
        "last_quality": last_quality,
        "last_win_rate": last_win_rate,
        "q_trend": q_trend,
        "wr_trend": wr_trend,
        "avg_reward": avg_reward,
    }


def _compute_composite_scores(
    beam_paths: List[List[BeamSearchResult]],
) -> Tuple[List[float], List[Dict[str, Any]]]:
    path_metrics = [_compute_path_metrics(p) for p in beam_paths]
    raw_confs = [m["confidence"] for m in path_metrics]
    raw_quals = [m["avg_quality"] for m in path_metrics]
    raw_wrs = [m["avg_win_rate"] for m in path_metrics]
    raw_rewards = [m["avg_reward"] for m in path_metrics]

    def _norm(vals):
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn) if mx > mn else 0.5 for v in vals]

    norm_confs = _norm(raw_confs)
    norm_quals = _norm(raw_quals)
    norm_wrs = _norm(raw_wrs)
    norm_rewards = _norm(raw_rewards)

    w_conf, w_qual, w_wr, w_rwd = _COMPOSITE_WEIGHTS
    composites = [
        w_conf * nc + w_qual * nq + w_wr * nw + w_rwd * nr
        for nc, nq, nw, nr in zip(norm_confs, norm_quals, norm_wrs, norm_rewards)
    ]
    return composites, path_metrics


def _build_rec_rows(
    sorted_indices: List[int],
    beam_paths: List[List[BeamSearchResult]],
    composites: List[float],
    path_metrics: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for display_order, orig_idx in enumerate(sorted_indices):
        m = path_metrics[orig_idx]
        rows.append(
            {
                "#": display_order + 1,
                "Path": orig_idx,
                "长度": m["length"],
                "首步动作": m["first_action"],
                "首步目标": m["first_state"],
                "末状态": m["last_state"],
                "置信度": f"{m['confidence']:.4f}",
                "平均Quality": f"{m['avg_quality']:.1f}",
                "平均胜率": f"{m['avg_win_rate']:.2%}",
                "末Quality": f"{m['last_quality']:.1f}",
                "末胜率": f"{m['last_win_rate']:.2%}",
                "Q趋势": m["q_trend"],
                "WR趋势": m["wr_trend"],
                "平均奖励": f"{m['avg_reward']:.1f}",
                "综合评分": f"{composites[orig_idx]:.3f}",
            }
        )
    return rows


def _build_path_detail_rows(
    beam_paths: List[List[BeamSearchResult]],
) -> List[Dict[str, Any]]:
    sorted_paths = sorted(
        enumerate(beam_paths),
        key=lambda x: tuple((r.state, r.action) for r in x[1]),
    )
    all_rows = []
    prev_path_states: List[int] = []
    for orig_idx, path in sorted_paths:
        cur_states = [r.state for r in path]
        fork_point = 0
        for i in range(min(len(cur_states), len(prev_path_states))):
            if cur_states[i] != prev_path_states[i]:
                break
            fork_point = i + 1
        for i, r in enumerate(path):
            if i < fork_point and orig_idx > 0:
                fork_mark = "↑"
            elif i == fork_point and orig_idx > 0:
                fork_mark = "➤"
            else:
                fork_mark = ""
            all_rows.append(
                {
                    "Path": orig_idx,
                    "分叉": fork_mark,
                    "Step": r.step,
                    "State": r.state,
                    "Action": r.action,
                    "Cum Prob": f"{r.cumulative_probability:.4f}",
                    "Quality": f"{r.quality_score:.2f}",
                    "Win Rate": f"{r.win_rate:.2%}",
                    "Step Reward": f"{r.avg_step_reward:.3f}",
                    "Future Reward": f"{r.avg_future_reward:.3f}",
                }
            )
        prev_path_states = cur_states
    return all_rows


def render_beam_tree(
    results: List[BeamSearchResult],
    highlight_indices: Optional[Set[int]] = None,
) -> str:
    """Generate a standalone pyvis tree for beam search paths."""
    if not results:
        return "<p>No results.</p>"

    g = nx.DiGraph()
    max_beam = max(r.beam_id for r in results)
    color_map = {i: _BEAM_COLORS[i % len(_BEAM_COLORS)] for i in range(max_beam + 1)}

    has_highlight = highlight_indices is not None and len(highlight_indices) > 0

    idx_map = {}
    for i, r in enumerate(results):
        idx_map[i] = i
        label = f"S{r.state}\nQ={r.quality_score:.1f}"

        if has_highlight:
            is_hl = i in highlight_indices
            if is_hl:
                node_color = "#FFD700"
                node_border = 4
                node_opacity = 1.0
                font_size = 16
            else:
                node_color = color_map[r.beam_id]
                node_border = 1
                node_opacity = 0.35
                font_size = 12
        else:
            node_color = color_map[r.beam_id]
            node_border = 3 if r.step == 0 else 1
            node_opacity = 1.0
            font_size = 14 if r.step == 0 else 13

        g.add_node(
            i,
            label=label,
            color=node_color,
            font={"size": font_size, "color": "white"},
            size=20 if r.step == 0 else 15,
            borderWidth=node_border,
            opacity=node_opacity,
            title=(
                f"State: {r.state}\n"
                f"Action: {r.action}\n"
                f"Step: {r.step}\n"
                f"Quality: {r.quality_score:.2f}\n"
                f"Win Rate: {r.win_rate:.2%}\n"
                f"Step Reward: {r.avg_step_reward:.3f}\n"
                f"Future Reward: {r.avg_future_reward:.3f}\n"
                f"Cum Prob: {r.cumulative_probability:.4f}"
            ),
        )

    for i, r in enumerate(results):
        if r.parent_idx is not None and r.parent_idx in idx_map:
            if has_highlight:
                edge_hl = i in highlight_indices and r.parent_idx in highlight_indices
                edge_color = "#FFD700" if edge_hl else "#666666"
                edge_width = 3 if edge_hl else 0.5
                edge_opacity = 1.0 if edge_hl else 0.3
            else:
                edge_color = color_map[r.beam_id]
                edge_width = 1
                edge_opacity = 1.0

            g.add_edge(
                idx_map[r.parent_idx],
                idx_map[i],
                title=f"{r.action} ({r.cumulative_probability:.3f})",
                label=f"{r.action} ({r.cumulative_probability:.1%})",
                color=edge_color,
                width=edge_width,
                font={"size": 10, "color": "#cccccc"},
                arrows="to",
                smooth={"type": "continuous"},
                opacity=edge_opacity,
            )

    net = Network(height="500px", width="100%", directed=True, notebook=False)

    for nid, ndata in g.nodes(data=True):
        net.add_node(nid, **ndata)
    for src, tgt, edata in g.edges(data=True):
        net.add_edge(src, tgt, **edata)

    options_dict = {
        "physics": {
            "enabled": False,
            "hierarchicalRepulsion": {
                "nodeDistance": 180,
                "centralGravity": 0.0,
                "springLength": 200,
                "springConstant": 0.01,
                "damping": 0.09,
            },
        },
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "LR",
                "sortMethod": "directed",
                "levelSeparation": 250,
                "nodeSpacing": 180,
                "blockShifting": True,
                "edgeMinimization": True,
            },
        },
        "edges": {"smooth": {"type": "continuous", "roundness": 0.2}},
        "interaction": {"hover": True, "tooltipDelay": 50},
    }
    net.set_options(json.dumps(options_dict))

    raw_html = net.generate_html(notebook=False)

    resize_js = """<script>
(function() {
    var style = document.createElement('style');
    style.innerHTML = 'html, body { margin:0; padding:0; height:100%; overflow:hidden; } '
        + '.card { height:100%; padding:0; margin:0; } '
        + '#mynetwork { height:100%; width:100%; }';
    document.head.appendChild(style);
    var titles = document.querySelectorAll('center h1');
    titles.forEach(function(el) { el.remove(); });
    var centers = document.querySelectorAll('center');
    centers.forEach(function(el) { if(el.children.length === 0) el.remove(); });
    setTimeout(function() {
        if (window.network) {
            window.network.setSize('100%', '100%');
            window.network.redraw();
        }
    }, 500);
})();
</script>"""
    return raw_html.replace("</body>", resize_js + "</body>")


def main():
    st.set_page_config(
        page_title="Experience Transition Graph Visualizer",
        page_icon="🕸️",
        layout="wide",
    )

    st.title("🕸️ Experience Transition Graph Explorer")
    st.markdown("交互式浏览经验转移图 + 图上束搜索规划。")

    _TAB_OPTIONS = ["转移图可视化", "束搜索规划", "滚动推演", "原始数据", "实时对局"]
    _TAB_ICONS = ["🕸️", "🔮", "🎮", "📊", "⚡"]

    with st.sidebar:
        _sel = st.segmented_control(
            "功能",
            _TAB_OPTIONS,
            default=_TAB_OPTIONS[0],
            key="tab_selector",
        )
        if _sel:
            _sel_val = _sel[0] if isinstance(_sel, list) else _sel
            st.session_state._tab_value = _sel_val
        _sel_val = st.session_state.get("_tab_value", _TAB_OPTIONS[0])
        active_tab = _TAB_OPTIONS.index(_sel_val)
        st.session_state.active_tab = active_tab

        st.divider()

        catalog = load_kg_catalog()
        if not catalog:
            st.error("经验转移图目录为空，请检查 configs/kg_catalog.yaml")
            st.stop()

        maps: Dict[str, List[Dict]] = {}
        for entry in catalog:
            maps.setdefault(entry["map_id"], []).append(entry)

        map_ids = list(maps.keys())
        selected_map = st.selectbox("地图", map_ids)

        map_entries = maps[selected_map]
        kg_names = [e["name"] for e in map_entries]
        selected_kg_idx = st.selectbox(
            "经验转移图",
            options=range(len(map_entries)),
            format_func=lambda i: kg_names[i],
        )
        kg_entry = map_entries[selected_kg_idx]
        st.caption(
            f"📋 类型: {kg_entry.get('type', '-')}  |  "
            f"窗口: {kg_entry.get('context_window', 0)}  |  "
            f"data_id: {kg_entry.get('data_id', '-')}"
        )

        kg_data, quality_min, quality_max = load_kg(kg_entry["file"])
        transitions = load_transitions(kg_entry.get("transitions", ""))
        kg_obj = load_kg_object(kg_entry["file"])

        if "quality_low" not in st.session_state:
            st.session_state.quality_low = quality_min
        if "quality_high" not in st.session_state:
            st.session_state.quality_high = quality_max

        st.divider()

        _prev_tab = st.session_state.get("_prev_tab", -1)
        tab_just_switched = _prev_tab != active_tab
        st.session_state._prev_tab = active_tab

        if not tab_just_switched:
            if active_tab == 0:
                focus_enabled = st.checkbox(
                    "🎯 聚焦模式",
                    value=True,
                    help="仅展示指定状态的邻居子图",
                    key="viz_focus",
                )
                focus_state = None
                focus_hops = 2
                focus_forward = True
                focus_backward = True
                if focus_enabled:
                    focus_state = st.number_input(
                        "聚焦状态 ID",
                        min_value=0,
                        max_value=99999,
                        value=0,
                        key="viz_focus_state",
                    )
                    focus_hops = st.slider("扩展跳数", 1, 5, 2, key="viz_focus_hops")
                    focus_direction = st.radio(
                        "扩展方向",
                        ["双选", "作为源节点", "作为目标节点"],
                        index=0,
                        horizontal=True,
                        help="双选：同时扩展前后；源节点：仅后继；目标节点：仅前驱",
                        key="viz_focus_dir",
                    )
                    focus_forward = focus_direction in ("双选", "作为源节点")
                    focus_backward = focus_direction in ("双选", "作为目标节点")

                st.divider()

                min_visits = st.slider(
                    "最小访问次数",
                    1,
                    50,
                    1,
                    help="过滤低频 state-action 对",
                    key="viz_min_visits",
                )

                st.markdown("**Quality Score 范围**")
                quality_range = st.slider(
                    "Quality Score",
                    min_value=float(quality_min),
                    max_value=float(quality_max),
                    value=(
                        float(st.session_state.quality_low),
                        float(st.session_state.quality_high),
                    ),
                    step=0.5,
                    label_visibility="collapsed",
                    help="过滤 state-action 对的 Quality Score 范围",
                    key="viz_quality",
                )
                st.session_state.quality_low = quality_range[0]
                st.session_state.quality_high = quality_range[1]

                min_quality, max_quality = quality_range

                max_nodes = st.slider(
                    "最大节点数",
                    20,
                    500,
                    200,
                    help="限制渲染规模，避免卡顿",
                    key="viz_max_nodes",
                )

                st.divider()
                highlight_terminal = st.checkbox(
                    "高亮终端状态",
                    value=True,
                    help="Win终端=绿色, Loss终端=红色",
                    key="viz_hl_term",
                )

                st.divider()
                st.subheader("🎨 渲染设置")

                _EDGE_SMOOTH_OPTIONS = [
                    "continuous",
                    "manual_arc",
                ]
                _EDGE_SMOOTH_LABELS = {
                    "continuous": "直线",
                    "manual_arc": "弧线",
                }
                edge_smooth_type = st.selectbox(
                    "边样式",
                    options=_EDGE_SMOOTH_OPTIONS,
                    index=0,
                    format_func=lambda t: _EDGE_SMOOTH_LABELS[t],
                    help="直线: 简洁清晰; 弧线: 同一对节点间的多条边自动扇形分散",
                    key="viz_edge_smooth",
                )
                edge_roundness = st.slider(
                    "弯曲度",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    disabled=(edge_smooth_type != "manual_arc"),
                    help="手动弧线模式下，弧线弯曲的基础倍率",
                    key="viz_edge_round",
                )

                _LAYOUT_OPTIONS = [
                    "force_atlas_2based",
                    "barnes_hut",
                    "repulsion",
                    "hrepulsion",
                ]
                _LAYOUT_LABELS = {
                    "force_atlas_2based": "Force Atlas 2 (默认)",
                    "barnes_hut": "Barnes-Hut",
                    "repulsion": "Repulsion",
                    "hrepulsion": "Hierarchical Repulsion",
                }
                st.caption("**布局算法**")
                col_layout, col_btn = st.columns([5, 2])
                with col_layout:
                    layout_algorithm = st.selectbox(
                        "布局算法",
                        options=_LAYOUT_OPTIONS,
                        index=0,
                        format_func=lambda a: _LAYOUT_LABELS[a],
                        label_visibility="collapsed",
                        help="不同力导向算法影响节点分布方式",
                        key="viz_layout",
                    )
                with col_btn:
                    render_clicked = st.button(
                        "🔄 渲染", use_container_width=True, key="viz_render_btn"
                    )

                if "render_key" not in st.session_state:
                    st.session_state.render_key = 0
                if render_clicked:
                    st.session_state.render_key += 1

                freeze_layout = st.checkbox(
                    "🔒 冻结布局（拖拽不回弹）", value=False, key="viz_freeze"
                )

                st.divider()
                st.caption("数据来源: `cache/knowledge_graph/`")

            elif active_tab == 1:
                c1, c2 = st.columns(2)
                with c1:
                    st.number_input(
                        "起始状态 ID",
                        min_value=0,
                        max_value=99999,
                        value=0,
                        key="pred_state",
                    )
                with c2:
                    st.selectbox(
                        "评分策略",
                        options=["quality", "future_reward", "win_rate"],
                        format_func={
                            "quality": "Quality Score",
                            "future_reward": "Future Reward",
                            "win_rate": "Win Rate",
                        }.get,
                        key="sm",
                    )

                c3, c4 = st.columns(2)
                with c3:
                    st.slider("Beam Width", 1, 10, 3, key="beam_w")
                with c4:
                    st.slider("最大步数", 1, 15, 5, key="max_s")

                c5, c6 = st.columns(2)
                with c5:
                    st.slider("最低访问次数", 1, 10, 1, key="mv_pred")
                with c6:
                    st.slider(
                        "最大状态重复",
                        1,
                        5,
                        2,
                        key="msr",
                        help="每条 beam 路径中同一状态最多出现的次数。设为 1 则完全禁止重复访问。",
                    )

                c7, c8 = st.columns(2)
                with c7:
                    st.slider(
                        "累积概率阈值",
                        0.001,
                        0.1,
                        0.01,
                        step=0.001,
                        format="%.3f",
                        key="mcp",
                    )
                with c8:
                    st.slider(
                        "折扣因子",
                        0.5,
                        1.0,
                        0.9,
                        step=0.05,
                        format="%.2f",
                        key="df",
                        help="每步累积概率乘以此值的步数次幂。1.0=无折扣，越低越惩罚过深路径。",
                    )

                st.button(
                    "🔮 开始规划",
                    type="primary",
                    use_container_width=True,
                    key="pred_btn",
                )

            elif active_tab == 2:
                st.caption("单步规划参数（每步束搜索）")
                c1, c2 = st.columns(2)
                with c1:
                    st.number_input(
                        "起始状态 ID",
                        min_value=0,
                        max_value=99999,
                        value=0,
                        key="roll_state",
                    )
                with c2:
                    st.selectbox(
                        "评分策略",
                        options=["quality", "future_reward", "win_rate"],
                        format_func={
                            "quality": "Quality Score",
                            "future_reward": "Future Reward",
                            "win_rate": "Win Rate",
                        }.get,
                        key="roll_sm",
                    )

                c3, c4 = st.columns(2)
                with c3:
                    st.slider("Beam Width", 1, 10, 3, key="roll_bw")
                with c4:
                    st.slider("前瞻步数", 1, 15, 5, key="roll_la")

                c5, c6 = st.columns(2)
                with c5:
                    st.slider("最低访问次数", 1, 10, 1, key="roll_mv")
                with c6:
                    st.slider(
                        "最大状态重复",
                        1,
                        5,
                        2,
                        key="roll_msr",
                        help="同一条推演路径中同一状态最多出现的次数。设为 1 则完全禁止重复访问。",
                    )

                c7, c8 = st.columns(2)
                with c7:
                    st.slider(
                        "累积概率阈值",
                        0.001,
                        0.1,
                        0.01,
                        step=0.001,
                        format="%.3f",
                        key="roll_mcp",
                    )
                with c8:
                    st.slider(
                        "折扣因子",
                        0.5,
                        1.0,
                        0.9,
                        step=0.05,
                        format="%.2f",
                        key="roll_df",
                        help="每步累积概率乘以此值的步数次幂。1.0=无折扣，越低越惩罚过深路径。",
                    )

                st.caption("滚动推演参数")
                c9, c10, c11 = st.columns([6, 4, 1.5])
                with c9:
                    st.selectbox(
                        "动作选择",
                        options=list(_ACTION_STRATEGY_LABELS.keys()),
                        format_func=lambda x: _ACTION_STRATEGY_LABELS[x],
                        key="roll_as",
                    )
                with c10:
                    st.selectbox(
                        "状态转移",
                        options=list(_NEXT_STATE_MODE_LABELS.keys()),
                        format_func=lambda x: _NEXT_STATE_MODE_LABELS[x],
                        key="roll_nsm",
                    )
                with c11:
                    st.text_input("seed", value="42", key="roll_seed")

                c12, c13 = st.columns(2)
                with c12:
                    st.slider(
                        "ε",
                        0.01,
                        0.5,
                        0.1,
                        step=0.01,
                        format="%.2f",
                        key="roll_eps",
                        disabled=(st.session_state.get("roll_as") != "epsilon_greedy"),
                    )
                with c13:
                    st.slider("最大推演步数", 1, 100, 50, key="roll_mrs")

                st.markdown("**推演模式**")
                st.radio(
                    "mode",
                    options=["单步推演", "多步推演"],
                    index=0,
                    horizontal=True,
                    key="roll_mode",
                    label_visibility="collapsed",
                )

                if st.session_state.get("roll_mode") == "多步推演":
                    st.toggle("启用备选路径", value=False, key="roll_backup")
                    if st.session_state.get("roll_backup", False):
                        st.slider(
                            "备选评分阈值",
                            0.0,
                            1.0,
                            0.3,
                            step=0.05,
                            format="%.2f",
                            key="roll_backup_st",
                        )
                        st.slider(
                            "模糊匹配距离阈值",
                            0.0,
                            1.0,
                            0.2,
                            step=0.05,
                            format="%.2f",
                            key="roll_backup_dt",
                        )

                st.button(
                    "🎲 开始推演",
                    type="primary",
                    use_container_width=True,
                    key="roll_btn",
                )

            elif active_tab == 4:
                _render_live_game_sidebar(kg_entry)

    st.markdown(f"### {_TAB_ICONS[active_tab]} {_TAB_OPTIONS[active_tab]}")

    if tab_just_switched:
        st.spinner("")
        st.rerun()

    if active_tab == 0:
        _render_visualization_tab(
            kg_data,
            transitions,
            kg_entry,
            min_visits,
            min_quality,
            max_quality,
            max_nodes,
            focus_state,
            focus_hops,
            focus_forward,
            focus_backward,
            focus_enabled,
            highlight_terminal,
            edge_smooth_type,
            edge_roundness,
            layout_algorithm,
            freeze_layout,
        )

    elif active_tab == 1:
        _render_prediction_tab(kg_data, transitions, kg_entry, kg_obj)

    elif active_tab == 2:
        _render_rollout_tab(kg_data, transitions, kg_entry, kg_obj)

    elif active_tab == 3:
        _render_raw_data_tab(kg_entry)

    elif active_tab == 4:
        _render_live_game_content()


def _render_visualization_tab(
    kg_data,
    transitions,
    kg_entry,
    min_visits,
    min_quality,
    max_quality,
    max_nodes,
    focus_state,
    focus_hops,
    focus_forward,
    focus_backward,
    focus_enabled,
    highlight_terminal,
    edge_smooth_type,
    edge_roundness,
    layout_algorithm,
    freeze_layout,
):
    n_states = len(kg_data.get("unique_states", set()))
    n_actions = len(kg_data.get("unique_actions", set()))
    total_visits = kg_data.get("total_visits", 0)

    col1, col2, col3 = st.columns(3)
    col1.metric("状态节点", n_states)
    col2.metric("动作种类", n_actions)
    col3.metric("总访问次数", total_visits)

    with st.spinner("构建图谱..."):
        nodes, edges, state_stats, state_found = build_graph_data(
            kg_data,
            transitions,
            min_visits=min_visits,
            min_quality=min_quality,
            max_quality=max_quality,
            max_nodes=max_nodes,
            focus_state=focus_state,
            focus_hops=focus_hops,
            focus_forward=focus_forward,
            focus_backward=focus_backward,
        )

    if focus_enabled and not state_found:
        st.toast(f"状态 {focus_state} 不存在于经验转移图中", icon="⚠️")
        st.info("当前渲染: **0 个节点**, **0 条边**")
        st.stop()

    st.info(f"当前渲染: **{len(nodes)} 个节点**, **{len(edges)} 条边**")

    if not nodes:
        st.warning("没有满足条件的节点，请放宽筛选条件。")
        st.stop()

    with st.spinner("渲染可视化（首次可能需要几秒）..."):
        html = render_pyvis_html(
            nodes,
            edges,
            state_stats,
            highlight_terminal,
            edge_smooth_type=edge_smooth_type,
            edge_roundness=edge_roundness,
            layout_algorithm=layout_algorithm,
            freeze_layout=freeze_layout,
            canvas_height=750,
        )

    b64 = base64.b64encode(html.encode()).decode()
    data_uri = (
        f"data:text/html;charset=utf-8;base64,{b64}#v={st.session_state.render_key}"
    )
    st.markdown(
        f'''<div style="resize:both; overflow:hidden; width:100%; height:800px; max-width:100%; border:2px solid #444; border-radius:4px;">
        <iframe src="{data_uri}" width="100%" height="100%" style="border:none; display:block;"></iframe>
      </div>''',
        unsafe_allow_html=True,
    )

    st.divider()

    with st.expander("📊 节点统计表（按访问量排序）"):
        table_data = sorted(nodes, key=lambda x: x["total_visits"], reverse=True)
        st.dataframe(
            [
                {
                    "State": n["id"],
                    "Visits": n["total_visits"],
                    "Actions": n["n_actions"],
                    "Best Win Rate": f"{n['best_win_rate'] * 100:.1f}%",
                    "Best Quality": f"{n['best_quality']:.1f}",
                }
                for n in table_data
            ],
            use_container_width=True,
        )

    with st.expander("📊 边统计表（按质量排序 Top 50）"):
        sorted_edges = sorted(edges, key=lambda x: x["quality_score"], reverse=True)[
            :50
        ]
        st.dataframe(
            [
                {
                    "From": e["from"],
                    "To": e["to"],
                    "Action": e["action"],
                    "Visits": e["visits"],
                    "Quality": f"{e['quality_score']:.1f}",
                    "Win Rate": f"{e['win_rate'] * 100:.1f}%",
                    "Step Reward": f"{e['avg_step_reward']:.2f}",
                    "Future Reward": f"{e['avg_future_reward']:.2f}",
                    "Trans Prob": f"{e['transition_prob'] * 100:.1f}%",
                }
                for e in sorted_edges
            ],
            use_container_width=True,
        )

    with st.expander("🔍 状态搜索"):
        search_id = st.number_input(
            "输入状态 ID", min_value=0, max_value=99999, value=0, key="search"
        )
        if st.button("搜索", key="search_btn"):
            if search_id in state_stats:
                info = state_stats[search_id]
                st.json(
                    {
                        "state_id": search_id,
                        "total_visits": info["total_visits"],
                        "n_available_actions": info["n_actions"],
                        "best_quality_score": round(info["best_quality"], 2),
                        "best_win_rate": round(info["best_win_rate"], 4),
                    }
                )

                key = search_id if not kg_data["use_context"] else (search_id, ())
                actions = kg_data["state_action_map"].get(key, {})
                if actions:
                    st.subheader("该状态的所有动作统计")
                    action_rows = []
                    for act, stats in sorted(
                        actions.items(), key=lambda x: x[1].quality_score, reverse=True
                    ):
                        action_rows.append(
                            {
                                "Action": act,
                                "Visits": stats.visits,
                                "Quality": f"{stats.quality_score:.2f}",
                                "Win Rate": f"{stats.win_rate * 100:.1f}%",
                                "Avg Step Reward": f"{stats.avg_step_reward:.2f}",
                                "Avg Future Reward": f"{stats.avg_future_reward:.2f}",
                            }
                        )
                    st.dataframe(action_rows, use_container_width=True)

                    if search_id in transitions:
                        st.subheader("该状态的转移概率")
                        trans_rows = []
                        for act, t_info in transitions[search_id].items():
                            ns_dict = t_info.get("next_states", {})
                            total_c = sum(ns_dict.values())
                            for ns, c in sorted(
                                ns_dict.items(), key=lambda x: x[1], reverse=True
                            ):
                                trans_rows.append(
                                    {
                                        "Action": act,
                                        "Next State": ns,
                                        "Count": c,
                                        "Probability": f"{c / total_c * 100:.1f}%"
                                        if total_c
                                        else "N/A",
                                    }
                                )
                        st.dataframe(trans_rows, use_container_width=True)
                else:
                    st.info("该状态在当前 KG 中无动作记录。")
            else:
                st.warning(f"状态 {search_id} 不存在于经验转移图中。")


@st.cache_data(max_entries=200, show_spinner=False)
def _cached_beam_results(
    _kg_path: str,
    _trans_path: str,
    start_state: int,
    beam_width: int,
    max_steps: int,
    min_visits: int,
    min_cum_prob: float,
    score_mode: str,
    max_state_revisits: int,
    discount_factor: float,
):
    kg = DecisionKnowledgeGraph.load(str(KG_DIR / _kg_path))
    with open(str(KG_DIR / _trans_path), "rb") as f:
        transitions = pickle.load(f)

    action, info = find_optimal_action(
        kg,
        transitions,
        start_state,
        beam_width=beam_width,
        max_steps=max_steps,
        min_visits=min_visits,
        min_cum_prob=min_cum_prob,
        score_mode=score_mode,
        max_state_revisits=max_state_revisits,
        discount_factor=discount_factor,
    )

    if action is None:
        return {"action": None}

    results = info["all_results"]

    results_ser = [
        {
            "step": r.step,
            "state": r.state,
            "action": r.action,
            "cumulative_probability": r.cumulative_probability,
            "quality_score": r.quality_score,
            "win_rate": r.win_rate,
            "avg_step_reward": r.avg_step_reward,
            "avg_future_reward": r.avg_future_reward,
            "beam_id": r.beam_id,
            "parent_idx": r.parent_idx,
        }
        for r in results
    ]

    beam_paths = get_beam_paths(results)
    beam_paths_ser = []
    for path in beam_paths:
        beam_paths_ser.append(
            [
                {
                    "step": r.step,
                    "state": r.state,
                    "action": r.action,
                    "cumulative_probability": r.cumulative_probability,
                    "quality_score": r.quality_score,
                    "win_rate": r.win_rate,
                    "avg_step_reward": r.avg_step_reward,
                    "avg_future_reward": r.avg_future_reward,
                }
                for r in path
            ]
        )

    return {
        "action": action,
        "info": {
            "expected_cumulative_reward": info["expected_cumulative_reward"],
            "expected_win_rate": info["expected_win_rate"],
            "best_beam_cum_prob": info["best_beam_cum_prob"],
            "best_beam_length": info["best_beam_length"],
            "reason": info.get("reason"),
        },
        "results": results_ser,
        "beam_paths": beam_paths_ser,
    }


@st.cache_data(max_entries=200, show_spinner=False)
def _cached_rollout_results(
    _kg_path: str,
    _trans_path: str,
    start_state: int,
    score_mode: str,
    action_strategy: str,
    next_state_mode: str,
    beam_width: int,
    lookahead_steps: int,
    max_rollout_steps: int,
    min_visits: int,
    min_cum_prob: float,
    max_state_revisits: int,
    discount_factor: float,
    epsilon: float,
    rng_seed: Optional[int],
    rollout_mode: str = "single_step",
    enable_backup: bool = False,
    score_threshold: float = 0.3,
    distance_threshold: float = 0.2,
):
    kg = DecisionKnowledgeGraph.load(str(KG_DIR / _kg_path))
    with open(str(KG_DIR / _trans_path), "rb") as f:
        transitions = pickle.load(f)

    dist_matrix = None
    if enable_backup:
        map_id = _kg_path.split("_")[0] if "_" in _kg_path else ""
        data_id = _kg_path.split("_")[1] if "_" in _kg_path else ""
        if map_id and data_id:
            dm_path = NPY_DIR / f"state_distance_matrix_{map_id}_{data_id}.npy"
            if dm_path.exists():
                dist_matrix = np.load(str(dm_path))

    result = chain_rollout(
        kg,
        transitions,
        start_state,
        score_mode=score_mode,
        action_strategy=action_strategy,
        next_state_mode=next_state_mode,
        beam_width=beam_width,
        lookahead_steps=lookahead_steps,
        max_rollout_steps=max_rollout_steps,
        min_visits=min_visits,
        min_cum_prob=min_cum_prob,
        max_state_revisits=max_state_revisits,
        discount_factor=discount_factor,
        epsilon=epsilon,
        rng_seed=rng_seed,
        rollout_mode=rollout_mode,
        enable_backup=enable_backup,
        score_threshold=score_threshold,
        distance_threshold=distance_threshold,
        dist_matrix=dist_matrix,
    )

    nodes_ser = {}
    for nid, n in result.nodes.items():
        nodes_ser[str(nid)] = {
            "id": n.id,
            "parent_id": n.parent_id,
            "children_ids": n.children_ids,
            "state": n.state,
            "action": n.action,
            "beam_id": n.beam_id,
            "quality_score": n.quality_score,
            "win_rate": n.win_rate,
            "avg_future_reward": n.avg_future_reward,
            "avg_step_reward": n.avg_step_reward,
            "visits": n.visits,
            "transition_prob": n.transition_prob,
            "cumulative_probability": n.cumulative_probability,
            "rollout_depth": n.rollout_depth,
            "is_on_chosen_path": n.is_on_chosen_path,
            "is_terminal": n.is_terminal,
            "is_beam_root": n.is_beam_root,
        }

    return {
        "nodes": nodes_ser,
        "root_id": result.root_id,
        "chosen_path_ids": result.chosen_path_ids,
        "termination_reason": result.termination_reason,
        "beam_results_by_step": {
            str(k): v for k, v in result.beam_results_by_step.items()
        },
        "rollout_mode": result.rollout_mode,
        "plan_segments": result.plan_segments,
        "total_re_searches": result.total_re_searches,
        "total_backup_switches": result.total_backup_switches,
        "switch_points_by_segment": {
            str(k): v for k, v in result.switch_points_by_segment.items()
        },
    }


def _run_prediction(kg_data, transitions, kg_entry, kg):
    start_state = st.session_state.get("pred_state", 0)
    score_mode = st.session_state.get("sm", "quality")
    beam_width = st.session_state.get("beam_w", 3)
    max_steps = st.session_state.get("max_s", 5)
    min_cum_prob = st.session_state.get("mcp", 0.01)
    min_visits_pred = st.session_state.get("mv_pred", 1)
    max_state_revisits = st.session_state.get("msr", 2)
    discount_factor = st.session_state.get("df", 0.9)

    unique_states = kg_data.get("unique_states", set())
    if isinstance(unique_states, set):
        if start_state not in unique_states:
            st.toast(f"状态 {start_state} 不存在于经验转移图中", icon="⚠️")
            st.warning(
                f"状态 {start_state} 不存在于当前经验转移图中。请选择一个有效状态。"
            )
            return

    if start_state not in transitions:
        st.warning(f"状态 {start_state} 没有转移数据，无法进行规划。")
        return

    kg_file = kg_entry.get("file", "")
    trans_file = kg_entry.get("transitions", "")

    cached = _cached_beam_results(
        kg_file,
        trans_file,
        start_state,
        beam_width,
        max_steps,
        min_visits_pred,
        min_cum_prob,
        score_mode,
        max_state_revisits,
        discount_factor,
    )

    if cached["action"] is None:
        st.error("无法规划：该状态无可用动作。")
        return

    action = cached["action"]
    info = cached["info"]
    results = [BeamSearchResult(**r) for r in cached["results"]]

    beam_paths = get_beam_paths(results)
    if not beam_paths:
        st.info("无搜索结果。")
        return

    composites, path_metrics = _compute_composite_scores(beam_paths)

    sorted_indices = sorted(
        range(len(beam_paths)), key=lambda i: composites[i], reverse=True
    )

    if (
        "pred_selected_path" not in st.session_state
        or st.session_state.pred_selected_path >= len(beam_paths)
    ):
        st.session_state.pred_selected_path = sorted_indices[0] if sorted_indices else 0

    selected_path_idx = st.session_state.pred_selected_path
    selected_path = beam_paths[selected_path_idx]
    highlight_set = {i for i, r in enumerate(results) if r in selected_path}

    col_action, col_tree, col_rec = st.columns([0.15, 0.85, 0.8])

    with col_action:
        st.markdown("**🎯 最优动作推荐**")
        st.metric("推荐动作", action)
        st.metric("预期累积奖励", f"{info['expected_cumulative_reward']:.3f}")
        st.metric("预期胜率", f"{info['expected_win_rate']:.2%}")
        st.metric("最优路径累积概率", f"{info['best_beam_cum_prob']:.4f}")
        st.metric("最优路径步数", info["best_beam_length"])

        if info.get("reason") == "no_transitions":
            st.caption("注意：该状态无转移数据，推荐结果仅基于单步质量评分。")

    with col_tree:
        with st.spinner("生成路径树..."):
            tree_html = render_beam_tree(results, highlight_indices=highlight_set)

        tree_b64 = base64.b64encode(tree_html.encode()).decode()
        tree_uri = f"data:text/html;charset=utf-8;base64,{tree_b64}#v=1"
        json_b64 = base64.b64encode(
            _results_to_json(
                results, beam_width, max_steps, min_cum_prob, score_mode, start_state
            ).encode()
        ).decode()
        json_uri = f"data:application/json;base64,{json_b64}"
        st.markdown(
            f"**🌳 路径树图**  "
            f'<a href="{tree_uri}" '
            f'download="beam_tree_state{start_state}.html" '
            f'style="font-size:0.85em;color:#4CAF50;text-decoration:none;margin-right:12px;">📥 导出 HTML</a>'
            f'<a href="{json_uri}" '
            f'download="beam_tree_state{start_state}.json" '
            f'style="font-size:0.85em;color:#2196F3;text-decoration:none;">📥 导出 JSON</a>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'''<div style="overflow:hidden; width:100%; height:500px; border:2px solid #444; border-radius:4px;">
        <iframe src="{tree_uri}" width="100%" height="100%" style="border:none; display:block;"></iframe>
      </div>''',
            unsafe_allow_html=True,
        )
        st.caption("节点颜色对应不同 Beam；鼠标悬停查看详情。")

    with col_rec:
        st.markdown("**📊 路径推荐**")

        rec_rows = _build_rec_rows(sorted_indices, beam_paths, composites, path_metrics)

        event = st.dataframe(
            rec_rows,
            use_container_width=True,
            height=min(len(rec_rows) * 35 + 50, 400),
            on_select="rerun",
            selection_mode="single-row",
        )

        if event and event.selection.rows:
            new_path = rec_rows[event.selection.rows[0]]["Path"]
            if new_path != st.session_state.pred_selected_path:
                st.session_state.pred_selected_path = new_path
                st.rerun()

    col_detail, col_frag = st.columns([1, 0.8])

    with col_detail:
        st.markdown("**📋 路径详情**")

        all_rows = _build_path_detail_rows(beam_paths)

        st.dataframe(
            all_rows, use_container_width=True, height=min(len(all_rows) * 35 + 50, 500)
        )

    with col_frag:
        st.markdown("**🔍 片段匹配**")
        map_id = kg_entry.get("map_id", "")
        data_id = kg_entry.get("data_id", "")
        match_rows = []
        dist_mat = None

        if map_id and data_id:
            with st.spinner("匹配原始对局片段..."):
                ep_data = load_episode_data(map_id, data_id)
                dist_mat = load_distance_matrix_np(map_id, data_id)

            if ep_data["n_episodes"] > 0:
                match_results = match_beam_paths(
                    {i: path for i, path in enumerate(beam_paths)},
                    ep_data["states"],
                    ep_data["actions"],
                    ep_data["outcomes"],
                    ep_data["scores"],
                    dist_mat,
                    top_k=5,
                )

                for bid in sorted(match_results.keys()):
                    for rank, m in enumerate(match_results[bid]):
                        match_rows.append(
                            {
                                "Path": bid,
                                "Rank": rank + 1,
                                "Episode": m.episode_id,
                                "位置": f"{m.start_pos}~{m.end_pos}",
                                "状态相似度": f"{m.state_similarity:.1%}",
                                "动作匹配率": f"{m.action_match_rate:.0%}",
                                "综合置信度": f"{m.combined_score:.1%}",
                                "结果": m.outcome,
                                "得分": m.episode_score,
                            }
                        )

        if match_rows:
            st.dataframe(
                match_rows,
                use_container_width=True,
                height=min(len(match_rows) * 35 + 50, 500),
            )
            wins = sum(1 for r in match_rows if r["结果"] == "Win")
            total_m = len(match_rows)
            st.caption(
                f"共 {total_m} 条匹配 | Win: {wins} ({wins / total_m:.0%}) | "
                f"距离矩阵: {'有' if dist_mat is not None else '无(精确匹配)'}"
            )
        else:
            st.info("无匹配结果。原始对局数据可能未加载。")


_run_prediction = st.fragment(_run_prediction)


def _render_prediction_tab(kg_data, transitions, kg_entry, kg):
    st.markdown("### 从指定起始状态出发，用 Beam Search 在转移图上进行多步规划搜索。")
    _run_prediction(kg_data, transitions, kg_entry, kg)


def _run_episode_query(n_ep, outcomes, scores, ep_states, ep_actions):
    st.subheader("📋 Episode 查询")

    col_ep, col_btn = st.columns([4, 1])
    with col_ep:
        ep_id = st.number_input(
            "Episode ID", min_value=0, max_value=n_ep - 1, value=0, key="raw_ep"
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔍 查询", type="primary", use_container_width=True, key="raw_btn")

    if ep_id >= n_ep:
        st.error(f"Episode ID 超出范围 (0~{n_ep - 1})")
    else:
        outcome = outcomes[ep_id] if ep_id < len(outcomes) else "Unknown"
        score = scores[ep_id] if ep_id < len(scores) else 0.0
        states = ep_states[ep_id] if ep_id < len(ep_states) else []
        actions = ep_actions[ep_id] if ep_id < len(ep_actions) else []
        n_steps = len(states)

        st.markdown(
            f"**Episode #{ep_id}**  |  {outcome}  |  得分: {score}  |  步数: {n_steps}"
        )

        if n_steps > 0:
            rows = []
            for i in range(n_steps):
                act = actions[i] if i < len(actions) else ""
                rows.append(
                    {
                        "Step": i,
                        "State": states[i],
                        "Action": act,
                    }
                )

            st.dataframe(
                rows, use_container_width=True, height=min(n_steps * 35 + 50, 600)
            )


_run_episode_query = st.fragment(_run_episode_query)


def _run_distance_query(map_id, data_id):
    st.subheader("🔬 状态距离查询")

    dist_mat = load_distance_matrix_np(map_id, data_id)

    if dist_mat is None:
        st.warning("当前地图无距离矩阵缓存。")
        return

    n_states = dist_mat.shape[0]

    col_q1, col_q2 = st.columns([4, 1])
    with col_q1:
        query_state = st.number_input(
            "查询状态 ID",
            min_value=0,
            max_value=n_states - 1,
            value=0,
            key="dist_state",
        )
    with col_q2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button(
            "🔍 查询距离", type="primary", use_container_width=True, key="dist_btn"
        )

    col_m1, col_m2 = st.columns([1.5, 6])
    with col_m1:
        query_mode = st.segmented_control(
            "查询模式",
            options=["距离阈值", "Top-K"],
            key="dist_mode",
            label_visibility="collapsed",
        )
    with col_m2:
        threshold = 1.0
        top_k = 10
        if query_mode == "距离阈值":
            threshold = st.slider(
                "距离阈值",
                0.0,
                float(np.max(dist_mat)),
                1.0,
                step=0.1,
                format="%.1f",
                key="dist_thresh",
                label_visibility="collapsed",
            )
        else:
            top_k = st.slider(
                "Top-K", 1, 50, 10, key="dist_topk", label_visibility="collapsed"
            )

    if query_state >= n_states:
        st.error(f"状态 ID 超出范围（0~{n_states - 1}）")
    else:
        row = dist_mat[query_state]

        if query_mode == "距离阈值":
            indices = np.where(
                (row <= threshold) & (np.arange(n_states) != query_state)
            )[0]
            dists = row[indices]
            order = np.argsort(dists)
            indices = indices[order]
            dists = dists[order]
        else:
            sorted_idx = np.argsort(row)
            indices = sorted_idx[1 : top_k + 1]
            dists = row[indices]

        st.info(
            f"状态 {query_state} 在 {n_states} 个状态中"
            f"{'，阈值 ' + str(threshold) + ' 内' if query_mode == '距离阈值' else '，Top-' + str(top_k)}"
            f"有 {len(indices)} 个匹配"
        )

        dist_rows = [
            {"State": int(idx), "Distance": round(float(d), 4)}
            for idx, d in zip(indices, dists)
        ]
        st.dataframe(
            dist_rows,
            use_container_width=True,
            height=min(len(dist_rows) * 35 + 50, 500),
        )


_run_distance_query = st.fragment(_run_distance_query)


def _run_hp_query(map_id, data_id):
    st.subheader("❤️ 状态 HP 查询")

    hp_data = load_state_hp_data(map_id, data_id)
    n_hp_states = hp_data["n_states"]

    if n_hp_states == 0:
        st.warning(
            "无 HP 数据。请确认 data/ 目录下有 bktree/ 和 graph/state_node.txt。"
        )
        return

    col_hp1, col_hp2 = st.columns([4, 1])
    with col_hp1:
        hp_state_id = st.number_input(
            "查询状态 ID",
            min_value=0,
            max_value=n_hp_states - 1,
            value=0,
            key="hp_state",
        )
    with col_hp2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔍 查询 HP", type="primary", use_container_width=True, key="hp_btn")

    if hp_state_id >= n_hp_states:
        st.error(f"状态 ID 超出范围（0~{n_hp_states - 1}）")
    elif hp_state_id not in hp_data["hp_lookup"]:
        st.warning(f"状态 {hp_state_id} 在 BK-Tree 中无对应节点。")
    else:
        info = hp_data["hp_lookup"][hp_state_id]
        cluster = info["cluster"]
        score = info["score"]
        red = info["red_army"]
        blue = info["blue_army"]

        red_hp = [u[2] * 100 for u in red]
        blue_hp = [u[2] * 100 for u in blue]
        red_avg = sum(red_hp) / len(red_hp) if red_hp else 0
        blue_avg = sum(blue_hp) / len(blue_hp) if blue_hp else 0
        red_total = sum(red_hp)
        blue_total = sum(blue_hp)
        hp_diff = red_total - blue_total
        diff_label = (
            "红方优势" if hp_diff > 0 else "蓝方优势" if hp_diff < 0 else "持平"
        )

        st.markdown(
            f"**状态 {hp_state_id}**  |  聚类: `{cluster}`  |  Score: {score:.2f}"
        )

        c_h1, c_h2, c_h3, c_h4 = st.columns(4)
        c_h1.metric("红方单位数", len(red))
        c_h2.metric("红方平均 HP", f"{red_avg:.1f}%")
        c_h3.metric("蓝方单位数", len(blue))
        c_h4.metric("蓝方平均 HP", f"{blue_avg:.1f}%")

        c_h5, c_h6 = st.columns(2)
        c_h5.metric("HP 差值", f"{hp_diff:+.1f}%", diff_label)
        c_h6.metric("红/蓝总 HP", f"{red_total:.1f}% / {blue_total:.1f}%")

        with st.expander("红方单位详情"):
            red_rows = [
                {
                    "单位": i + 1,
                    "X": f"{u[0]:.3f}",
                    "Y": f"{u[1]:.3f}",
                    "HP%": f"{u[2] * 100:.1f}%",
                }
                for i, u in enumerate(red)
            ]
            st.dataframe(red_rows, use_container_width=True)

        with st.expander("蓝方单位详情"):
            blue_rows = [
                {
                    "单位": i + 1,
                    "X": f"{u[0]:.3f}",
                    "Y": f"{u[1]:.3f}",
                    "HP%": f"{u[2] * 100:.1f}%",
                }
                for i, u in enumerate(blue)
            ]
            st.dataframe(blue_rows, use_container_width=True)


_run_hp_query = st.fragment(_run_hp_query)


@st.cache_data(max_entries=10, show_spinner="正在计算 MDS 降维...")
def compute_mds(_map_id: str, _data_id: str):
    cache_path = NPY_DIR / f"mds_{_map_id}_{_data_id}.npy"
    if cache_path.exists():
        return np.load(str(cache_path))

    dist_path = NPY_DIR / f"state_distance_matrix_{_map_id}_{_data_id}.npy"
    if not dist_path.exists():
        return None
    dist_mat = np.load(str(dist_path))
    n = dist_mat.shape[0]

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        normalized_stress=False,
        random_state=42,
        n_init=4,
        max_iter=300,
        eps=1e-6,
        n_jobs=-1,
    )
    coords = mds.fit_transform(dist_mat)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), coords)
    return coords


def _build_hp_diff_array(hp_data: Dict, n_states: int) -> np.ndarray:
    hp_diff = np.full(n_states, np.nan)
    for sid, info in hp_data["hp_lookup"].items():
        red = info.get("red_army", [])
        blue = info.get("blue_army", [])
        if red or blue:
            red_total = sum(u[2] * 100 for u in red)
            blue_total = sum(u[2] * 100 for u in blue)
            hp_diff[sid] = red_total - blue_total
    return hp_diff


def plot_mds_terrain(
    coords: np.ndarray,
    hp_diff: np.ndarray,
    state_stats: Optional[Dict[int, Dict]] = None,
):
    import plotly.graph_objects as go

    x, y = coords[:, 0], coords[:, 1]
    valid_mask = ~np.isnan(hp_diff)
    n_states = len(x)
    grid_n = int(min(200, np.sqrt(n_states) * 3))
    grid_n = max(grid_n, 50)

    traces = []

    if valid_mask.sum() > 10:
        x_v, y_v, z_v = x[valid_mask], y[valid_mask], hp_diff[valid_mask]

        x_margin = (x_v.max() - x_v.min()) * 0.05
        y_margin = (y_v.max() - y_v.min()) * 0.05
        xi = np.linspace(x_v.min() - x_margin, x_v.max() + x_margin, grid_n)
        yi = np.linspace(y_v.min() - y_margin, y_v.max() + y_margin, grid_n)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        zi = griddata((x_v, y_v), z_v, (xi_grid, yi_grid), method="linear")

        traces.append(
            go.Contour(
                z=zi,
                x=xi,
                y=yi,
                colorscale="RdBu",
                reversescale=False,
                opacity=0.65,
                contours=dict(start=-100, end=100, size=5),
                hovertemplate="MDS X: %{x:.1f}<br>MDS Y: %{y:.1f}<br>HP差: %{z:.1f}<extra></extra>",
                showscale=True,
                colorbar=dict(
                    title=dict(text="HP差值(红-蓝)", font=dict(size=12)), len=0.85
                ),
                line=dict(width=0),
            )
        )

    n_valid = int(valid_mask.sum())
    if n_valid > 0:
        traces.append(
            go.Scattergl(
                x=x[valid_mask],
                y=y[valid_mask],
                mode="markers",
                marker=dict(
                    size=4,
                    color=hp_diff[valid_mask],
                    colorscale="RdBu",
                    reversescale=False,
                    showscale=False,
                    opacity=0.7,
                    line=dict(width=0),
                ),
                customdata=np.column_stack(
                    [np.arange(n_states)[valid_mask], hp_diff[valid_mask]]
                ),
                hovertemplate="State: %{customdata[0]}<br>HP差: %{customdata[1]:.1f}<extra></extra>",
                name="有HP数据",
            )
        )

    n_missing = n_states - n_valid
    if n_missing > 0:
        traces.append(
            go.Scattergl(
                x=x[~valid_mask],
                y=y[~valid_mask],
                mode="markers",
                marker=dict(size=2, color="gray", opacity=0.25, line=dict(width=0)),
                customdata=np.arange(n_states)[~valid_mask],
                hovertemplate="State: %{customdata[0]} (无HP数据)<extra></extra>",
                name="无HP数据",
            )
        )

    title = f"状态地形图 ({n_states} 个状态)"
    if n_missing > 0:
        title += f"  [HP: {n_valid}, 缺失: {n_missing}]"

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(title="MDS X", gridcolor="#e0e0e0"),
        yaxis=dict(title="MDS Y", gridcolor="#e0e0e0", scaleanchor="x", scaleratio=1),
        font=dict(family="SimHei, Microsoft YaHei, sans-serif"),
        margin=dict(l=60, r=30, t=50, b=50),
        dragmode="pan",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _run_mds_terrain(map_id, data_id, n_states_kg):
    st.subheader("🗺️ 状态地形图（MDS 降维 + HP 插值）")

    dist_path = NPY_DIR / f"state_distance_matrix_{map_id}_{data_id}.npy"
    if not dist_path.exists():
        st.warning("当前地图无距离矩阵缓存，无法计算 MDS。")
        return

    dist_mat = np.load(str(dist_path))
    n_states_dist = dist_mat.shape[0]

    c1, c2, c3 = st.columns([1, 1, 1])
    c1.metric("状态数", n_states_dist)
    c2.metric("距离矩阵", f"{n_states_dist}x{n_states_dist}")

    cache_path = NPY_DIR / f"mds_{map_id}_{data_id}.npy"
    has_cache = cache_path.exists()

    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        do_compute = st.button(
            f"🗺️ {'加载' if has_cache else '计算'}地形图",
            type="primary",
            use_container_width=True,
            key="mds_btn",
        )

    if not do_compute:
        if has_cache:
            st.info(f"已有缓存: `cache/npy/mds_{map_id}_{data_id}.npy`，点击加载。")
        else:
            n_est = max(1, int(n_states_dist * n_states_dist * 8e-9))
            st.info(f"首次计算预计需要 ~{n_est} 分钟，计算结果将缓存到本地。")
        return

    coords = compute_mds(map_id, data_id)
    if coords is None:
        st.error("MDS 计算失败：无法加载距离矩阵。")
        return

    hp_data = load_state_hp_data(map_id, data_id)
    hp_diff = _build_hp_diff_array(hp_data, n_states_dist)

    with st.spinner("正在生成地形图..."):
        fig = plot_mds_terrain(coords, hp_diff)

    st.markdown(
        '<div class="mds-sq" style="height:0;overflow:hidden"></div>'
        "<style>"
        ".stVerticalBlock:has(.mds-sq) [data-testid='stPlotlyChart'] {"
        "  width: min(100%, 85vh) !important;"
        "  aspect-ratio: 1/1 !important;"
        "  margin: 0 auto !important;"
        "}"
        ".stVerticalBlock:has(.mds-sq) .js-plotly-plot,"
        ".stVerticalBlock:has(.mds-sq) .plot-container,"
        ".stVerticalBlock:has(.mds-sq) .svg-container {"
        "  width: 100% !important; height: 100% !important;"
        "}"
        "</style>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, height=700)
    st.caption(
        "X/Y: MDS 降维坐标 | 颜色: HP 差值（红方优势=红色, 蓝方优势=蓝色） | 支持拖拽/缩放/悬停查看"
    )


_run_mds_terrain = st.fragment(_run_mds_terrain)


_ACTION_STRATEGY_LABELS = {
    "best_beam": "Best Beam",
    "best_subtree_quality": "Best Subtree Quality",
    "best_subtree_winrate": "Best Subtree WinRate",
    "highest_transition_prob": "Highest Trans. Prob",
    "random_beam": "Random Beam",
    "epsilon_greedy": "Epsilon-Greedy",
}

_NEXT_STATE_MODE_LABELS = {
    "sample": "概率采样",
    "highest_prob": "最高概率",
}

BRIDGE_API_URL = "http://localhost:8000"

PROJECT_ROOT = Path(__file__).parent.parent


def _render_live_game_sidebar(kg_entry: Optional[Dict] = None):
    kg_file = kg_entry.get("file", "") if kg_entry else ""
    kg_name = kg_entry.get("name", "") if kg_entry else ""
    kg_data_dir = kg_entry.get("data_dir", "") if kg_entry else ""
    if kg_file:
        st.caption(f"KG: {kg_name}")
        if kg_data_dir:
            st.caption(f"路径: {kg_data_dir}/bktree/")
            import json, glob as _glob

            _bkt_dir = PROJECT_ROOT / kg_data_dir / "bktree"
            _pri = _bkt_dir / "primary_bktree.json"
            _pri_cnt = 0
            if _pri.exists():
                try:
                    _d = json.load(open(str(_pri), "r"))
                    _stk = [_d]
                    while _stk:
                        _n = _stk.pop()
                        _pri_cnt += 1
                        _stk.extend(_n.get("children", {}).values())
                except Exception:
                    pass
            _sec_files = sorted(_glob.glob(str(_bkt_dir / "secondary_bktree_*.json")))
            _sec_cnt = len(_sec_files)
            _sn_path = PROJECT_ROOT / kg_data_dir / "graph" / "state_node.txt"
            _sn_cnt = 0
            if _sn_path.exists():
                try:
                    _sn_cnt = sum(1 for _ in open(str(_sn_path), "r") if _.strip())
                except Exception:
                    pass
            with st.expander("BKTree 详情", expanded=False):
                st.caption(f"Primary 节点: {_pri_cnt}")
                st.caption(f"Secondary 树: {_sec_cnt}")
                st.caption(f"State 映射: {_sn_cnt} 条")
                st.caption(
                    f"map_id: {kg_entry.get('map_id', '-')} | data_id: {kg_entry.get('data_id', '-')}"
                )

    col_port, col_btn = st.columns([3, 2])
    with col_port:
        port = st.number_input(
            "API 端口",
            min_value=1024,
            max_value=65535,
            value=8000,
            key="live_port",
        )
    _start_clicked = False
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            "一键启动", type="primary", use_container_width=True, key="live_start"
        ):
            _start_clicked = True

    st.caption("窗口位置")
    wc1, wc2, wc3, wc4 = st.columns(4)
    with wc1:
        st.number_input("X", value=2600, key="live_wx", label_visibility="collapsed")
    with wc2:
        st.number_input("Y", value=50, key="live_wy", label_visibility="collapsed")
    with wc3:
        st.number_input("W", value=640, key="live_ww", label_visibility="collapsed")
    with wc4:
        st.number_input("H", value=480, key="live_wh", label_visibility="collapsed")

    ap_port = int(st.session_state.get("live_port", 8000))
    api_base = f"http://localhost:{ap_port}"

    st.divider()
    start_paused = st.toggle(
        "启动后暂停",
        value=False,
        key="live_start_paused",
        help="开启后服务启动时自动进入暂停状态，需手动恢复",
    )

    st.subheader("Beam Search 参数")

    c1, c2 = st.columns(2)
    with c1:
        st.selectbox(
            "评分策略",
            options=["quality", "future_reward", "win_rate"],
            format_func={
                "quality": "Quality Score",
                "future_reward": "Future Reward",
                "win_rate": "Win Rate",
            }.get,
            key="live_sm",
        )
    with c2:
        st.slider("Beam Width", 1, 10, 3, key="live_bw")

    c3, c4 = st.columns(2)
    with c3:
        st.slider("前瞻步数", 1, 15, 5, key="live_la")
    with c4:
        st.slider("最低访问次数", 1, 10, 1, key="live_mv")

    c5, c6 = st.columns(2)
    with c5:
        st.slider(
            "最大状态重复",
            1,
            5,
            2,
            key="live_msr",
            help="同一条推演路径中同一状态最多出现的次数。",
        )
    with c6:
        st.slider(
            "累积概率阈值",
            0.001,
            0.1,
            0.01,
            step=0.001,
            format="%.3f",
            key="live_mcp",
        )

    c7, c8 = st.columns(2)
    with c7:
        st.slider(
            "折扣因子",
            0.5,
            1.0,
            0.9,
            step=0.05,
            format="%.2f",
            key="live_df",
            help="每步累积概率乘以此值的步数次幂。1.0=无折扣。",
        )
    with c8:
        st.selectbox(
            "动作选择",
            options=list(_ACTION_STRATEGY_LABELS.keys()),
            format_func=lambda x: _ACTION_STRATEGY_LABELS[x],
            key="live_as",
        )

    c9, c10 = st.columns(2)
    with c9:
        st.slider(
            "ε",
            0.01,
            0.5,
            0.1,
            step=0.01,
            format="%.2f",
            key="live_eps",
            disabled=(st.session_state.get("live_as") != "epsilon_greedy"),
        )
    with c10:
        st.slider("最大推演步数", 1, 100, 50, key="live_mrs")

    st.markdown("**推演模式**")
    live_mode = st.radio(
        "mode",
        options=["单步推演", "多步推演"],
        index=0,
        horizontal=True,
        key="live_mode",
        label_visibility="collapsed",
    )

    live_backup = st.toggle("启用备选路径", value=False, key="live_backup")
    if live_backup:
        bc1, bc2 = st.columns(2)
        with bc1:
            st.slider(
                "备选评分阈值",
                0.0,
                1.0,
                0.3,
                step=0.05,
                format="%.2f",
                key="live_backup_st",
            )
        with bc2:
            st.slider(
                "模糊匹配距离阈值",
                0.0,
                1.0,
                0.2,
                step=0.05,
                format="%.2f",
                key="live_backup_dt",
            )

    if _start_clicked:
        if "live_proc" in st.session_state and st.session_state.live_proc is not None:
            proc = st.session_state.live_proc
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_live_game.py"),
            "--mode",
            "all",
            "--port",
            str(port),
        ]
        if kg_file:
            cmd.extend(["--kg_file", kg_file])
        if kg_data_dir:
            cmd.extend(["--data_dir", kg_data_dir])
        cmd.extend(
            [
                "--window_x",
                str(st.session_state.get("live_wx", 50)),
                "--window_y",
                str(st.session_state.get("live_wy", 50)),
                "--window_w",
                str(st.session_state.get("live_ww", 640)),
                "--window_h",
                str(st.session_state.get("live_wh", 480)),
            ]
        )
        p = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32"
            else 0,
        )
        st.session_state.live_proc = p
        _beam_params = {
            "beam_width": st.session_state.get("live_bw", 3),
            "max_steps": st.session_state.get("live_mrs", 50),
            "score_mode": st.session_state.get("live_sm", "quality"),
            "min_visits": st.session_state.get("live_mv", 1),
            "min_cum_prob": st.session_state.get("live_mcp", 0.01),
            "max_state_revisits": st.session_state.get("live_msr", 2),
            "discount_factor": st.session_state.get("live_df", 0.9),
        }
        _ap_payload = {
            "enabled": True,
            "mode": "single_step" if live_mode == "单步推演" else "multi_step",
            "lookahead_steps": st.session_state.get("live_la", 5),
            "action_strategy": st.session_state.get("live_as", "best_beam"),
            "score_mode": st.session_state.get("live_sm", "quality"),
            "beam_width": st.session_state.get("live_bw", 3),
            "max_steps": st.session_state.get("live_mrs", 50),
            "min_visits": st.session_state.get("live_mv", 1),
            "min_cum_prob": st.session_state.get("live_mcp", 0.01),
            "max_state_revisits": st.session_state.get("live_msr", 2),
            "discount_factor": st.session_state.get("live_df", 0.9),
            "enable_backup": live_backup,
            "epsilon": st.session_state.get("live_eps", 0.1),
        }
        with st.spinner("等待服务启动..."):
            for _ in range(30):
                time.sleep(0.5)
                try:
                    r = requests.get(f"{api_base}/game/status", timeout=2)
                    if r.status_code == 200:
                        requests.post(
                            f"{api_base}/game/beam_params",
                            json=_beam_params,
                            timeout=5,
                        )
                        requests.post(
                            f"{api_base}/game/autopilot",
                            json=_ap_payload,
                            timeout=5,
                        )
                        if start_paused:
                            requests.post(
                                f"{api_base}/game/control",
                                json={"command": "pause"},
                                timeout=5,
                            )
                        st.success(
                            "服务已启动，自动决策已开启"
                            + (" (已暂停)" if start_paused else "")
                        )
                        break
                except Exception:
                    continue
            else:
                st.warning("服务启动超时，请检查日志")

    try:
        r = requests.get(f"{api_base}/game/autopilot/status", timeout=2)
        if r.status_code == 200:
            data = r.json()
            if data.get("enabled"):
                stats = data.get("stats", {})
                st.markdown(
                    f"**运行中** | {data.get('mode', '')} | "
                    f"决策:{stats.get('total_decisions', 0)} | "
                    f"重规划:{stats.get('total_replans', 0)} | "
                    f"偏离:{stats.get('total_divergences', 0)} | "
                    f"{data.get('plan_progress', '0/0')}"
                )
            else:
                st.caption("待机")
    except Exception:
        pass


def _build_live_game_html(port: int = 8000) -> str:
    _html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
:root{--bg:#0e1117;--sf:#262730;--sf2:#1e1f26;--bd:#3d3f4a;--tx:#fafafa;--tx2:#a3a8b8;
--ac:#4fc3f7;--grn:#4caf50;--ylw:#ffc107;--red:#f44336;--rd:8px}
.light{--bg:#f5f5f5;--sf:#ffffff;--sf2:#eeeeee;--bd:#d0d0d0;--tx:#1a1a1a;--tx2:#666666;
--ac:#0288d1;--grn:#2e7d32;--ylw:#f57f17;--red:#d32f2f;--rd:8px}
.light .console-body{background:#fff;color:#333}
.light .log-line{border-bottom-color:rgba(0,0,0,.08)}
.light .log-ts{color:#999}
.light .lv-info{color:#666}.light .lv-success{color:#2e7d32}.light .lv-warn{color:#e65100}.light .lv-error{color:#c62828}
.light .src-api{background:#e3f2fd;color:#1565c0}
.light .src-game{background:#eceff1;color:#546e7a}
.light .badge-gray{background:#e0e0e0;color:#616161}
.light .badge-green{background:#e8f5e9;color:#2e7d32}
.light .badge-yellow{background:#fff8e1;color:#f57f17}
.light .btn-primary{background:#0288d1;color:#fff}
.light .btn-ghost{background:#eee;color:#333;border-color:#ccc}
.light .toggle .slider{background:#bbb}
.light .toast-ok{background:#e8f5e9;color:#1b5e20}
.light .lv-debug{color:#00796b}
.light .src-debug{background:#00796b;color:#b2dfdb}
.light .toast-err{background:#ffebee;color:#b71c1c}
.toast-wrap{position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:9999;display:flex;flex-direction:column;gap:6px;align-items:center;pointer-events:none}
.toast{padding:8px 20px;border-radius:6px;font-size:13px;font-weight:600;pointer-events:auto;animation:toast-in .2s ease;box-shadow:0 2px 8px rgba(0,0,0,.3)}
.toast-ok{background:#1b5e20;color:#a5d6a7}
.toast-err{background:#b71c1c;color:#ffcdd2}
@keyframes toast-in{from{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:translateY(0)}}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Source Sans Pro',-apple-system,sans-serif;background:var(--bg);color:var(--tx);font-size:14px;line-height:1.6;overflow:hidden;height:100vh}
.root{display:flex;height:100vh}
.left{flex:1;overflow-y:auto;padding:16px;min-width:0;border-right:1px solid var(--bd)}
.right{width:480px;display:flex;flex-direction:column;flex-shrink:0}
.console-header{padding:10px 14px;background:var(--sf);border-bottom:1px solid var(--bd);display:flex;justify-content:space-between;align-items:center;flex-shrink:0;position:relative}
.console-header span{font-size:13px;font-weight:600;color:var(--tx2);text-transform:uppercase;letter-spacing:.5px}
.filter-panel{position:absolute;top:100%;right:0;z-index:100;background:var(--sf);border:1px solid var(--bd);border-radius:var(--rd);padding:12px;min-width:200px;display:none;box-shadow:0 4px 12px rgba(0,0,0,.3)}
.filter-panel.show{display:block}
.filter-panel h4{font-size:12px;color:var(--tx2);margin:0 0 8px;font-weight:600}
.filter-group{margin-bottom:10px}
.filter-group:last-child{margin-bottom:0}
.filter-group label{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--tx);padding:2px 0;cursor:pointer}
.filter-group label:hover{color:var(--ac)}
.filter-group input[type=checkbox]{accent-color:var(--ac)}
.console-body{flex:1;overflow-y:auto;padding:8px 10px;font-family:'Cascadia Code','Fira Code','Consolas',monospace;font-size:12px;line-height:1.8}
.log-line{display:flex;gap:8px;padding:1px 0;border-bottom:1px solid rgba(61,63,74,.3)}
.log-ts{color:#666;flex-shrink:0;width:60px;text-align:right}
.log-src{flex-shrink:0;min-width:36px;padding:0 8px;font-weight:700;text-align:center;border-radius:3px;white-space:nowrap}
.src-api{background:#1565c0;color:#90caf9}
.src-game{background:#37474f;color:#b0bec5}
.log-msg{flex:1;word-break:break-all}
.lv-info{color:var(--tx2)}.lv-success{color:#a5d6a7}.lv-warn{color:#fff9c4}.lv-error{color:#ef9a9a}.lv-debug{color:#80cbc4}
.console-footer{padding:6px 14px;background:var(--sf);border-top:1px solid var(--bd);display:flex;justify-content:space-between;align-items:center;flex-shrink:0;font-size:12px;color:var(--tx2)}
.card{background:var(--sf);border:1px solid var(--bd);border-radius:var(--rd);padding:14px;margin-bottom:10px}
.card-title{font-size:13px;font-weight:600;color:var(--tx2);text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px}
.metrics{display:flex;gap:10px;flex-wrap:wrap}
.metric{flex:1;min-width:70px;background:var(--sf2);border-radius:6px;padding:8px 10px}
.metric-label{font-size:11px;color:var(--tx2);margin-bottom:2px}
.metric-value{font-size:18px;font-weight:700}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600}
.badge-green{background:#1b5e20;color:#a5d6a7}
.badge-yellow{background:#f57f17;color:#fff9c4}
.badge-gray{background:#424242;color:#9e9e9e}
.btn{border:none;border-radius:6px;padding:7px 14px;font-size:13px;font-weight:600;cursor:pointer;transition:opacity .15s}
.btn:hover{opacity:.85}.btn:active{opacity:.7}
.btn-primary{background:#4fc3f7;color:#fff}
.btn-blue-dark{background:#0288d1;color:#fff}
.btn-step{background:#4fc3f7;color:#fff}
.btn-danger{background:#f44336;color:#fff}
.btn-warn{background:#ff9800;color:#fff}
.btn-ok{background:#4caf50;color:#fff}
.btn-ghost{background:var(--sf2);color:var(--tx);border:1px solid var(--bd)}
.btn-sm{padding:5px 16px;font-size:12px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.row-between{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.card{background:var(--sf2);border-radius:8px;padding:12px;margin-bottom:10px}
.card-title{font-size:13px;font-weight:600;color:var(--tx2);margin-bottom:8px}
.toggle-wrap{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--tx2)}
.toggle{position:relative;width:36px;height:20px;cursor:pointer}
.toggle input{opacity:0;width:0;height:0}
.toggle .slider{position:absolute;inset:0;background:#424242;border-radius:10px;transition:.2s}
.toggle .slider:before{content:"";position:absolute;width:16px;height:16px;left:2px;top:2px;background:#fff;border-radius:50%;transition:.2s}
.toggle input:checked+.slider{background:var(--ac)}
.toggle input:checked+.slider:before{transform:translateX(16px)}
.ep-item{border:1px solid var(--bd);border-radius:6px;padding:8px 10px;margin-bottom:6px;background:var(--sf)}
.ep-item:hover{border-color:var(--ac)}
.ep-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.ep-title{font-size:13px;font-weight:600;color:var(--tx)}
.ep-badge{padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600}
.ep-badge-win{background:#1b5e20;color:#a5d6a7}
.ep-badge-loss{background:#b71c1c;color:#ef9a9a}
.ep-badge-interrupted{background:#424242;color:#9e9e9e}
.ep-badge-dogfall{background:#e65100;color:#ffe0b2}
.ep-meta{font-size:12px;color:var(--tx2);margin-bottom:4px}
.ep-flow{display:flex;flex-wrap:wrap;gap:2px;font-size:11px;margin-top:4px}
.ep-flow-item{padding:2px 5px;border-radius:3px;background:var(--sf2);color:var(--tx2);white-space:nowrap;font-size:11px}
.ep-flow-arrow{color:var(--ac);padding:0 1px}
.ep-events{font-size:10px;color:var(--tx2);margin-top:3px;max-height:60px;overflow-y:auto}
.ep-page-btn{padding:2px 8px;border-radius:4px;border:1px solid var(--bd);background:var(--sf);color:var(--tx);cursor:pointer;font-size:11px}
.ep-page-btn:hover{border-color:var(--ac)}
.ep-page-btn.active{background:var(--ac);color:#fff;border-color:var(--ac)}
.ep-details{margin-top:6px;border:1px solid var(--bd);border-radius:4px;overflow:hidden}
.ep-details summary{padding:5px 8px;font-size:11px;color:var(--tx2);cursor:pointer;background:var(--sf);user-select:none}
.ep-details summary:hover{color:var(--ac)}
.ep-plan-block{padding:6px 8px;border-bottom:1px solid var(--bd);font-size:11px}
.ep-plan-block:last-child{border-bottom:none}
.ep-plan-title{font-weight:600;color:var(--ac);margin-bottom:4px}
.ep-plan-details{border-bottom:1px solid var(--bd)}
.ep-plan-details:last-child{border-bottom:none}
.ep-plan-details summary{padding:5px 8px;font-size:11px;color:var(--ac);cursor:pointer;background:var(--sf);user-select:none;font-weight:600}
.ep-plan-details summary:hover{opacity:.85}
.ep-plan-details .ep-plan-content{padding:6px 8px}
.ep-beam-path{margin-bottom:3px;padding:3px 5px;border-radius:3px;border:1px solid var(--bd);font-size:10px}
.ep-beam-path.chosen{border-color:rgba(13,71,161,.45);background:rgba(13,71,161,.06)}
.ep-beam-path .path-label{font-weight:600;margin-right:4px}
.ep-beam-path.chosen .path-label{color:#1565c0}
.ep-beam-path:not(.chosen) .path-label{color:var(--tx2)}
.ep-beam-path .path-metrics{float:right;color:var(--tx2);font-size:9px}
.ep-beam-path .path-metrics span{margin-left:6px}
.ep-beam-step{display:inline-block;padding:0 3px;border-radius:2px;background:var(--sf2);color:var(--tx2);white-space:nowrap}
.ep-beam-path .path-arrow{color:var(--ac);opacity:.6;margin:0 1px}
.ep-beam-table{width:100%;border-collapse:collapse;font-size:10px;margin-top:4px}
.ep-beam-table th{text-align:left;padding:2px 4px;color:var(--tx2);font-weight:600;border-bottom:1px solid var(--bd)}
.ep-beam-table td{padding:2px 4px;color:var(--tx)}
</style></head>
<body>

<div id="toast-container" class="toast-wrap"></div>

<div class="root">
<div class="left">

<div style="display:flex;align-items:stretch;gap:10px;margin-bottom:10px">
  <button class="btn btn-ghost btn-sm" style="padding:5px 16px;font-size:15px;border-radius:var(--rd)" onclick="toggleTheme()" id="theme-btn" title="切换主题">&#9728;</button>
  <div class="card" style="margin-bottom:0;display:flex;align-items:center;gap:4px;padding:4px 8px">
    <span style="font-size:10px;color:var(--tx2);white-space:nowrap">窗口</span>
    <input type="number" id="win-x" value="2600" style="width:50px;padding:2px 3px;font-size:10px;border:1px solid var(--bd);border-radius:var(--rd);background:var(--sf);color:var(--tx);text-align:center" title="X">
    <input type="number" id="win-y" value="50" style="width:48px;padding:2px 3px;font-size:10px;border:1px solid var(--bd);border-radius:var(--rd);background:var(--sf);color:var(--tx);text-align:center" title="Y">
    <span style="font-size:10px;color:var(--tx2)">&times;</span>
    <input type="number" id="win-w" value="640" style="width:52px;padding:2px 3px;font-size:10px;border:1px solid var(--bd);border-radius:var(--rd);background:var(--sf);color:var(--tx);text-align:center" title="宽">
    <input type="number" id="win-h" value="480" style="width:52px;padding:2px 3px;font-size:10px;border:1px solid var(--bd);border-radius:var(--rd);background:var(--sf);color:var(--tx);text-align:center" title="高">
    <button class="btn btn-ghost btn-sm" style="padding:2px 8px;font-size:10px;margin-left:2px" onclick="applyWindowPos()">应用</button>
  </div>
  <div class="card" style="margin-bottom:0;flex:1">
    <div class="row-between">
      <div style="display:flex;align-items:center;gap:12px">
        <span class="conn-dot" id="conn-dot"></span>
        <span style="font-size:12px;color:var(--tx2)" id="conn-text">检测中...</span>
        <span style="font-size:11px;color:var(--tx2)">|</span>
        <span style="font-size:12px;color:var(--tx2)">端口: __PORT__</span>
      </div>
      <div class="toggle-wrap">
        <label class="toggle"><input type="checkbox" id="auto-refresh" checked><span class="slider"></span></label>
        <select class="input" id="refresh-interval" style="width:55px;padding:3px 5px;font-size:11px">
          <option value="1000">1s</option>
          <option value="2000" selected>2s</option>
          <option value="3000">3s</option>
          <option value="5000">5s</option>
        </select>
      </div>
    </div>
  </div>
  <div class="card" style="margin-bottom:0">
    <div style="display:flex;align-items:center;gap:6px;justify-content:center">
      <button class="btn btn-warn btn-sm" onclick="ctrl('pause')">暂停</button>
      <button class="btn btn-ok btn-sm" onclick="ctrl('resume')">恢复</button>
      <span style="color:var(--bd);font-size:16px;margin:0 2px">|</span>
      <button class="btn btn-step btn-sm" onclick="ctrl('step')">步进</button>
      <button class="btn btn-blue-dark btn-sm" onclick="ctrl('run_episode')">单局运行</button>
      <span style="color:var(--bd);font-size:16px;margin:0 2px">|</span>
      <button class="btn btn-danger btn-sm" onclick="shutdownService()">停止服务</button>
    </div>
  </div>
</div>

<div class="card">
  <div class="card-title">游戏状态</div>
  <div class="metrics" style="margin-bottom:10px">
    <div class="metric"><div class="metric-label">状态</div><div class="metric-value"><span class="badge badge-gray" id="state">--</span></div></div>
    <div class="metric"><div class="metric-label">帧</div><div class="metric-value" id="frame">--</div></div>
    <div class="metric"><div class="metric-label">Episode</div><div class="metric-value" id="episode">--</div></div>
    <div class="metric"><div class="metric-label">我方</div><div class="metric-value" id="my-count">--</div></div>
    <div class="metric"><div class="metric-label">敌方</div><div class="metric-value" id="enemy-count">--</div></div>
    <div class="metric"><div class="metric-label">聚类</div><div class="metric-value" id="cluster" style="font-size:13px">--</div></div>
  </div>
  <div class="metrics">
    <div class="metric"><div class="metric-label">我方 HP</div><div class="metric-value" id="my-hp">--</div></div>
    <div class="metric"><div class="metric-label">敌方 HP</div><div class="metric-value" id="enemy-hp">--</div></div>
    <div class="metric" style="flex:2"><div class="metric-label">上一步动作</div><div class="metric-value" id="last-action" style="font-size:13px;color:var(--ac)">--</div></div>
    <div class="metric"><div class="metric-label">KG</div><div class="metric-value" id="kg-status" style="font-size:11px">--</div></div>
    <div class="metric"><div class="metric-label">Buffer</div><div class="metric-value" id="history-buffer" style="font-size:11px;color:var(--ac)">--</div></div>
  </div>
</div>

<div class="card" id="episodes-card">
  <div class="card-title" style="display:flex;justify-content:space-between;align-items:center">
    <span>对局记录</span>
    <div style="display:flex;gap:6px;align-items:center">
      <select id="ep-sort" class="input" style="padding:3px 6px;font-size:11px;border-radius:4px;border:1px solid var(--bd);background:var(--sf);color:var(--tx)" onchange="epCurrentPage=1;renderLocalEpisodes()">
        <option value="id_desc">最新优先</option>
        <option value="id_asc">最早优先</option>
        <option value="score_desc">得分降序</option>
        <option value="score_asc">得分升序</option>
      </select>
      <input id="ep-search" type="text" placeholder="搜索..." style="padding:3px 8px;font-size:11px;border-radius:4px;border:1px solid var(--bd);background:var(--sf);color:var(--tx);width:80px" oninput="debounceSearch()">
      <button class="btn btn-ghost btn-sm" style="padding:3px 8px;font-size:11px" onclick="loadEpisodes()">刷新</button>
      <button class="btn btn-ghost btn-sm" style="padding:3px 8px;font-size:11px;color:#f44336" onclick="clearEpisodes()">清空</button>
    </div>
  </div>
  <div id="episodes-list" style="max-height:540px;overflow-y:auto"></div>
  <div id="ep-pagination" style="display:flex;justify-content:center;gap:4px;margin-top:8px;font-size:12px"></div>
  <div style="text-align:center;font-size:11px;color:var(--tx2);margin-top:4px"><span id="ep-count">0</span> 条记录</div>
</div>

</div>

<div class="right">
  <div class="console-header">
    <span>控制台日志</span>
    <div style="display:flex;gap:6px">
      <div style="position:relative">
        <button class="btn btn-ghost btn-sm" onclick="toggleFilter()">过滤</button>
        <div class="filter-panel" id="filter-panel">
          <div class="filter-group">
            <h4>日志级别</h4>
            <label><input type="checkbox" data-filter="level" value="info" checked onchange="applyFilters()"> Info</label>
            <label><input type="checkbox" data-filter="level" value="success" checked onchange="applyFilters()"> Success</label>
            <label><input type="checkbox" data-filter="level" value="warn" checked onchange="applyFilters()"> Warn</label>
            <label><input type="checkbox" data-filter="level" value="error" checked onchange="applyFilters()"> Error</label>
            <label><input type="checkbox" data-filter="level" value="debug" onchange="applyFilters()"> Debug</label>
          </div>
          <div class="filter-group">
            <h4>来源</h4>
            <label><input type="checkbox" data-filter="source" value="game" checked onchange="applyFilters()"> GAME</label>
            <label><input type="checkbox" data-filter="source" value="api" checked onchange="applyFilters()"> API</label>
            <label><input type="checkbox" data-filter="source" value="autopilot" checked onchange="applyFilters()"> Autopilot</label>
          </div>
          <div class="filter-group">
            <h4>消息类型</h4>
            <label><input type="checkbox" data-filter="type" value="info" checked onchange="applyFilters()"> Info</label>
            <label><input type="checkbox" data-filter="type" value="action" checked onchange="applyFilters()"> Action</label>
            <label><input type="checkbox" data-filter="type" value="fallback" checked onchange="applyFilters()"> Fallback</label>
            <label><input type="checkbox" data-filter="type" value="result" checked onchange="applyFilters()"> Result</label>
            <label><input type="checkbox" data-filter="type" value="episode" checked onchange="applyFilters()"> Episode</label>
            <label><input type="checkbox" data-filter="type" value="control" checked onchange="applyFilters()"> Control</label>
          </div>
          <div style="display:flex;gap:6px;margin-top:10px;border-top:1px solid var(--bd);padding-top:8px">
            <button class="btn btn-primary btn-sm" style="flex:1;padding:4px 0;font-size:11px" onclick="saveFilterConfig()">保存配置</button>
            <button class="btn btn-ghost btn-sm" style="flex:1;padding:4px 0;font-size:11px" onclick="resetFilterConfig()">重置默认</button>
          </div>
        </div>
      </div>
      <button class="btn btn-ghost btn-sm" onclick="clearLogs()">清空</button>
    </div>
  </div>
  <div class="console-body" id="console-body">
    <div style="text-align:center;color:#666;padding:40px 0">等待日志...</div>
  </div>
  <div class="console-footer">
    <span id="log-count">0 条</span>
    <span id="log-scroll-hint" style="cursor:pointer;color:var(--ac)" onclick="scrollConsoleBottom()">↓ 滚动到底部</span>
  </div>
</div>
</div>

<script>
const API='http://localhost:__PORT__';
let timer=null, logTimer=null, latestSeq=0, renderedMaxSeq=0, recommendedAction='';
let userScrolled=false;

const LV_COLORS={info:'lv-info',success:'lv-success',warn:'lv-warn',error:'lv-error',debug:'lv-debug'};
const SRC_CLS={api:'src-api',game:'src-game'};
let _filterLevels=new Set(['info','success','warn','error']);
let _filterSources=new Set(['game','api','autopilot']);
let _filterTypes=new Set(['info','action','fallback','result','episode','control']);
function _extractType(msg){
  if(!msg)return 'info';
  if(msg.startsWith('执行动作'))return 'action';
  if(msg.startsWith('回退策略'))return 'fallback';
  if(msg.startsWith('判定'))return 'result';
  if(msg.startsWith('Episode'))return 'episode';
  if(msg.startsWith('步进')||msg.startsWith('游戏暂停')||msg.startsWith('游戏恢复')||msg.startsWith('游戏停止'))return 'control';
  return 'info';
}
function _isFiltered(log){if(log.level==='debug')return !_filterLevels.has('debug');return !_filterLevels.has(log.level||'info')||!_filterSources.has(log.source||'game')||!_filterTypes.has(_extractType(log.message))}
function toggleFilter(){document.getElementById('filter-panel').classList.toggle('show')}
function _syncCheckboxesFromSets(){
  document.querySelectorAll('#filter-panel input[type=checkbox]').forEach(function(cb){
    var t=cb.getAttribute('data-filter'),v=cb.value;
    if(t==='level')cb.checked=_filterLevels.has(v);
    else if(t==='source')cb.checked=_filterSources.has(v);
    else cb.checked=_filterTypes.has(v);
  });
}
function _syncSetsFromCheckboxes(){
  _filterLevels=new Set();_filterSources=new Set();_filterTypes=new Set();
  document.querySelectorAll('#filter-panel input[type=checkbox]').forEach(function(cb){
    if(cb.checked){var t=cb.getAttribute('data-filter'),v=cb.value;if(t==='level')_filterLevels.add(v);else if(t==='source')_filterSources.add(v);else _filterTypes.add(v)}
  });
}
var FILTER_STORAGE_KEY='live_filter_cfg';
var FILTER_DEFAULTS={levels:['info','success','warn','error'],sources:['game','api','autopilot'],types:['info','action','fallback','result','episode','control']};
function loadFilterConfig(){
  var raw=localStorage.getItem(FILTER_STORAGE_KEY);
  if(!raw)return;
  try{
    var cfg=JSON.parse(raw);
    _filterLevels=new Set(cfg.levels||FILTER_DEFAULTS.levels);
    _filterSources=new Set(cfg.sources||FILTER_DEFAULTS.sources);
    _filterTypes=new Set(cfg.types||FILTER_DEFAULTS.types);
    _syncCheckboxesFromSets();
    document.querySelectorAll('#console-body .log-line').forEach(function(el){
      el.style.display=(_filterLevels.has(el.getAttribute('data-level'))&&_filterSources.has(el.getAttribute('data-source'))&&_filterTypes.has(el.getAttribute('data-type')))?'':'none';
    });
  }catch(e){}
}
function saveFilterConfig(){
  _syncSetsFromCheckboxes();
  localStorage.setItem(FILTER_STORAGE_KEY,JSON.stringify({levels:Array.from(_filterLevels),sources:Array.from(_filterSources),types:Array.from(_filterTypes)}));
  toast('过滤配置已保存');
}
function resetFilterConfig(){
  localStorage.removeItem(FILTER_STORAGE_KEY);
  _filterLevels=new Set(FILTER_DEFAULTS.levels);
  _filterSources=new Set(FILTER_DEFAULTS.sources);
  _filterTypes=new Set(FILTER_DEFAULTS.types);
  _syncCheckboxesFromSets();
  document.querySelectorAll('#console-body .log-line').forEach(function(el){el.style.display=''});
  toast('已重置为默认');
}
function applyFilters(){
  _syncSetsFromCheckboxes();
  document.querySelectorAll('#console-body .log-line').forEach(function(el){
    el.style.display=(_filterLevels.has(el.getAttribute('data-level'))&&_filterSources.has(el.getAttribute('data-source'))&&_filterTypes.has(el.getAttribute('data-type')))?'':'none';
  });
}
document.addEventListener('click',function(e){if(!e.target.closest('.filter-panel')&&!e.target.closest('[onclick*="toggleFilter"]')){document.getElementById('filter-panel').classList.remove('show')}});

async function apiGet(p){const r=await fetch(API+p);if(!r.ok)throw new Error(r.status);return r.json()}
async function apiPost(p,b){const r=await fetch(API+p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});if(!r.ok){const t=await r.text().catch(()=>'');throw new Error(t||r.status)}return r.json()}

function toast(m,ok=true){const c=document.getElementById('toast-container'),d=document.createElement('div');d.className='toast '+(ok?'toast-ok':'toast-err');d.textContent=m;c.appendChild(d);setTimeout(()=>d.remove(),3000)}

function setConn(ok){const d=document.getElementById('conn-dot'),t=document.getElementById('conn-text');d.className='conn-dot '+(ok?'conn-ok':'conn-err');t.textContent=ok?'已连接':'连接失败'}
function setStatus(s){const el=document.getElementById('state');if(s.running&&!s.paused){el.textContent='运行中';el.className='badge badge-green'}else if(s.paused){el.textContent='暂停';el.className='badge badge-yellow'}else{el.textContent='已停止';el.className='badge badge-gray'}document.getElementById('frame').textContent=s.frame||0;document.getElementById('episode').textContent=s.episode||0;document.getElementById('my-count').textContent=s.my_count!=null?s.my_count:'-';document.getElementById('enemy-count').textContent=s.enemy_count!=null?s.enemy_count:'-';document.getElementById('cluster').textContent=s.state_cluster||'-';const kgEl=document.getElementById('kg-status');if(s.kg_loaded){const f=s.kg_file||'';kgEl.textContent=f.split('/').pop().replace('.pkl','');kgEl.style.color='#4caf50'}else{kgEl.textContent='未加载';kgEl.style.color='#f44336'}const bufEl=document.getElementById('history-buffer');if(s.history_episodes!=null){bufEl.textContent=s.history_episodes+' eps / '+s.history_frames+' frames / '+s.history_capacity+' cap'}else{bufEl.textContent='--'}}
function setObs(o){if(o.error)return;document.getElementById('my-hp').textContent=o.my_total_hp||0;document.getElementById('enemy-hp').textContent=o.enemy_total_hp||0;document.getElementById('last-action').textContent=o.last_action||'-'}

async function refresh(){try{const s=await apiGet('/game/status');setStatus(s);setConn(true);if(!logTimer)startLogTimer();if(s.running){try{const o=await apiGet('/game/observation');setObs(o)}catch(e){}}}catch(e){setConn(false)}}

function startTimer(){stopTimer();const ms=parseInt(document.getElementById('refresh-interval').value)||2000;timer=setInterval(refresh,ms)}
function stopTimer(){if(timer){clearInterval(timer);timer=null}}
document.getElementById('auto-refresh').addEventListener('change',function(){if(this.checked)startTimer();else stopTimer()});
document.getElementById('refresh-interval').addEventListener('change',function(){if(document.getElementById('auto-refresh').checked)startTimer()});

const consoleBody=document.getElementById('console-body');
consoleBody.addEventListener('scroll',function(){const el=this;userScrolled=(el.scrollTop+el.clientHeight)<el.scrollHeight-30});
function scrollConsoleBottom(){consoleBody.scrollTop=consoleBody.scrollHeight;userScrolled=false}

function appendLogLine(log){
  if(log.seq!==undefined&&log.seq<=renderedMaxSeq)return;
  if(log.seq!==undefined&&log.seq>renderedMaxSeq)renderedMaxSeq=log.seq;
  const div=document.createElement('div');
  div.className='log-line';
  div.setAttribute('data-level',log.level||'info');
  div.setAttribute('data-source',log.source||'game');
  div.setAttribute('data-type',_extractType(log.message));
  if(_isFiltered(log)){div.style.display='none'}
  const lvCls=LV_COLORS[log.level]||'lv-info';
  const srcCls=SRC_CLS[log.source]||'src-game';
  div.innerHTML=`<span class="log-ts">${log.ts||''}</span><span class="log-src ${srcCls}">${(log.source||'').toUpperCase()}</span><span class="log-msg ${lvCls}">${escHtml(log.message||'')}</span>`;
  consoleBody.appendChild(div);
  while(consoleBody.childElementCount>200){
    var first=consoleBody.firstChild;
    if(first.style.display==='none'){consoleBody.removeChild(first)}
    else{var vc=0;for(var c=consoleBody.firstChild;c;c=c.nextSibling){if(c.style.display!=='none')vc++}if(vc<=200)break;consoleBody.removeChild(first)}
  }
  if(!userScrolled)scrollConsoleBottom();
}
function escHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}

async function fetchLogs(){
  try{
    const r=await apiGet('/game/logs?after_seq='+latestSeq);
    const logs=r.logs||[];
    if(logs.length===0)return;
    if(consoleBody.querySelector('[style*="text-align:center"]'))consoleBody.innerHTML='';
    logs.forEach(l=>appendLogLine(l));
    if(r.latest_seq>latestSeq)latestSeq=r.latest_seq;
    document.getElementById('log-count').textContent=consoleBody.childElementCount+' 条';
  }catch(e){}
}

function startLogTimer(){stopLogTimer();logTimer=setInterval(fetchLogs,1500)}
function stopLogTimer(){if(logTimer){clearInterval(logTimer);logTimer=null}}

async function clearLogs(){try{await apiPost('/game/logs/clear',{})}catch(e){}consoleBody.innerHTML='<div style="text-align:center;color:#666;padding:40px 0">已清空</div>';latestSeq=0;renderedMaxSeq=0;document.getElementById('log-count').textContent='0 条'}

async function ctrl(cmd){try{await apiPost('/game/control',{command:cmd});toast('已发送: '+cmd);await refresh()}catch(e){toast('失败: '+e.message,false)}}
function toggleTheme(){const d=document.documentElement;d.classList.toggle('light');const isLight=d.classList.contains('light');document.getElementById('theme-btn').innerHTML=isLight?'&#9728;':'&#9790;';localStorage.setItem('live_theme',isLight?'light':'dark')}
function loadWindowPos(){var s=localStorage.getItem('live_win_pos');if(!s)return;try{var p=JSON.parse(s);document.getElementById('win-x').value=p.x||2600;document.getElementById('win-y').value=p.y||50;document.getElementById('win-w').value=p.w||640;document.getElementById('win-h').value=p.h||480}catch(e){}}
async function applyWindowPos(){var x=parseInt(document.getElementById('win-x').value)||50;var y=parseInt(document.getElementById('win-y').value)||50;var w=parseInt(document.getElementById('win-w').value)||640;var h=parseInt(document.getElementById('win-h').value)||480;localStorage.setItem('live_win_pos',JSON.stringify({x:x,y:y,w:w,h:h}));try{var r=await apiPost('/game/window_pos',{x:x,y:y,w:w,h:h});if(r.ok)toast('窗口位置已更新');else toast('更新失败: '+(r.error||''),false)}catch(e){toast('请求失败: '+e.message,false)}}
async function shutdownService(){stopTimer();stopLogTimer();setConn(false);latestSeq=0;renderedMaxSeq=0;consoleBody.innerHTML='<div style="text-align:center;color:#666;padding:40px 0">等待日志...</div>';document.getElementById('log-count').textContent='0 条';document.getElementById('conn-text').textContent='已停止';try{await fetch(API+'/game/autopilot',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:false})})}catch(e){}try{await fetch(API+'/game/shutdown',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})}catch(e){}toast('服务已停止')}

var epCurrentPage=1,epPerPage=10,epSearchTimer=null,allEpisodes=[];
function debounceSearch(){clearTimeout(epSearchTimer);epSearchTimer=setTimeout(function(){epCurrentPage=1;renderLocalEpisodes()},400)}
async function loadEpisodes(){
  try{
    var data=await apiGet('/game/episodes?page=1&per_page=999');
    var newEps=data.episodes||[];
    newEps.forEach(function(ep){
      var idx=allEpisodes.findIndex(function(e){return e.id===ep.id});
      if(idx>=0){
        if((!ep.plans||ep.plans.length===0)&&allEpisodes[idx].plans&&allEpisodes[idx].plans.length>0){
          ep.plans=allEpisodes[idx].plans;
        }
        allEpisodes[idx]=ep
      }else{allEpisodes.push(ep)}
    });
    var agentIds=[];
    newEps.forEach(function(ep){if(ep.source==='agent_buffer')agentIds.push(ep.id)});
    if(agentIds.length>0){try{await apiPost('/game/episodes/ack',{ids:agentIds})}catch(e){}}
    renderLocalEpisodes();
  }catch(e){}
}
function renderLocalEpisodes(){
  var sort=document.getElementById('ep-sort').value;
  var search=document.getElementById('ep-search').value.trim().toLowerCase();
  var filtered=allEpisodes.slice();
  if(search){filtered=filtered.filter(function(ep){return ep.result.toLowerCase().indexOf(search)>=0||ep.mode.toLowerCase().indexOf(search)>=0})}
  if(sort==='id_desc')filtered.sort(function(a,b){return b.id-a.id});
  else if(sort==='id_asc')filtered.sort(function(a,b){return a.id-b.id});
  else if(sort==='score_desc')filtered.sort(function(a,b){return(b.score||0)-(a.score||0)});
  else if(sort==='score_asc')filtered.sort(function(a,b){return(a.score||0)-(b.score||0)});
  var total=filtered.length;
  var start=(epCurrentPage-1)*epPerPage;
  var pageItems=filtered.slice(start,start+epPerPage);
  renderEpisodes({episodes:pageItems,total:total,page:epCurrentPage,per_page:epPerPage,total_pages:Math.max(1,Math.ceil(total/epPerPage))});
}
var epAutoTimer=null;
function startEpAutoRefresh(){stopEpAutoRefresh();epAutoTimer=setInterval(loadEpisodes,5000)}
function stopEpAutoRefresh(){if(epAutoTimer){clearInterval(epAutoTimer);epAutoTimer=null}}
async function clearEpisodes(){try{await apiPost('/game/episodes/clear');allEpisodes=[];renderEpisodes({episodes:[],total:0,page:1,per_page:epPerPage,total_pages:1});toast('对局记录已清空')}catch(e){toast('清空失败: '+e.message,false)}}

function renderEpisodes(data){
  var list=document.getElementById('episodes-list');
  var pgn=document.getElementById('ep-pagination');
  var cnt=document.getElementById('ep-count');
  var eps=data.episodes||[];
  cnt.textContent=data.total||0;
  if(eps.length===0){list.innerHTML='<div style="text-align:center;color:var(--tx2);padding:20px 0;font-size:12px">暂无对局记录</div>';pgn.innerHTML='';return}
  var html='';
  for(var i=0;i<eps.length;i++){
    var ep=eps[i];
    var isMulti=ep.mode==='multi_step';
    var badgeCls='ep-badge-interrupted';
    if(ep.result==='Win')badgeCls='ep-badge-win';
    else if(ep.result==='Loss')badgeCls='ep-badge-loss';
    else if(ep.result==='Dogfall')badgeCls='ep-badge-dogfall';
    var flowHtml='';
    if(ep.markov_flow&&ep.markov_flow.length>0){
      var maxShow=Math.min(ep.markov_flow.length,20);
      for(var j=0;j<maxShow;j++){
        var item=ep.markov_flow[j];
        var ev=(ep.events&&ep.events[j])?ep.events[j]:{event_type:'no_action'};
        var et=ev.event_type||'no_action';
        var bg,bd,tc;
        if(et==='kg_plan'){
          if(isMulti){bg='rgba(13,71,161,.18)';bd='2px dashed rgba(13,71,161,.5)';tc='#42a5f5'}
          else{bg='rgba(13,71,161,.12)';bd='1px solid rgba(13,71,161,.3)';tc='#64b5f6'}
        }else if(et==='kg_follow'){
          bg='rgba(27,94,32,.10)';bd='1px solid rgba(27,94,32,.25)';tc='#81c784';
        }else if(et==='diverge'){
          bg='rgba(245,127,23,.15)';bd='1px solid rgba(245,127,23,.3)';tc='#ffb74d';
        }else if(et==='fallback'){
          bg='rgba(183,28,28,.15)';bd='1px solid rgba(183,28,28,.3)';tc='#ef9a9a';
        }else if(et==='backup_switch'){
          bg='rgba(123,31,162,.15)';bd='1px solid rgba(123,31,162,.3)';tc='#ce93d8';
        }else{
          bg='var(--sf2)';bd='1px solid var(--bd)';tc='var(--tx2)';
        }
        var lbl={kg_plan:'规划',kg_follow:'跟随',diverge:'偏离',fallback:'回退',backup_switch:'备选',external:'外部',no_action:'-'}[et]||et;
        if(j>0)flowHtml+='<span class="ep-flow-arrow">&rarr;</span>';
        var sid=item[0]!=null?(Array.isArray(item[0])?'M('+item[0].join(',')+')':'S'+item[0]):'S?';
        var act=item[1]||'';
        if(act.startsWith('action_'))act=act.replace(/^action_/,'').substring(0,6);
        flowHtml+='<span class="ep-flow-item" style="background:'+bg+';border:'+bd+';color:'+tc+'" title="'+lbl+'">'+sid+'<span style="opacity:.7;margin-left:2px">'+escHtml(act)+'</span></span>';
      }
      if(ep.markov_flow.length>20)flowHtml+='<span class="ep-flow-item" style="color:var(--ylw);background:none;border:none">+'+(ep.markov_flow.length-20)+'</span>';
    }
    var scoreStr=ep.score!==undefined&&ep.score!==0?(ep.score>0?'+':'')+ep.score.toFixed(0):'-';
    html+='<div class="ep-item">';
    html+='<div class="ep-header"><span class="ep-title">#'+ep.id+'</span><span class="ep-badge '+badgeCls+'">'+(ep.result||'?')+'</span></div>';
    html+='<div class="ep-meta">得分: '+scoreStr+' | 步数: '+ep.steps+' | '+(ep.mode==='multi_step'?'多步':'单步')+' '+(ep.match_mode?'('+ep.match_mode+')':'')+' | '+(ep.timestamp||'')+'</div>';
    if(flowHtml)html+='<div class="ep-flow">'+flowHtml+'</div>';
    if(ep.plans&&ep.plans.length>0){
      html+='<details class="ep-details"><summary>推演详情 ('+ep.plans.length+' 次规划)</summary>';
      for(var p=0;p<ep.plans.length;p++){
        var pl=ep.plans[p];
        var trigger=pl.trigger||'';
        var trigLbl={diverge:'偏离触发',exhausted:'用尽重规划',single_step:'单步规划'}[trigger]||trigger||pl.mode;
        html+='<details class="ep-plan-details"><summary>规划 #'+(p+1)+' — S'+pl.state_id+' ('+trigLbl+')</summary>';
        html+='<div class="ep-plan-content">';
        if(pl.beam_paths&&pl.beam_paths.length>0){
          for(var pi=0;pi<pl.beam_paths.length;pi++){
            var path=pl.beam_paths[pi];
            var chosen=path.chosen?'chosen':'';
            html+='<div class="ep-beam-path '+chosen+'">';
            html+='<span class="path-label">'+(path.chosen?'[选中] ':'')+path.rank+'</span>';
            html+='<span class="path-metrics"><span>CumP:'+(path.cum_prob*100).toFixed(1)+'%</span><span>'+(path.steps.length-1)+'步</span></span>';
            html+='<br>';
            for(var si=0;si<path.steps.length;si++){
              var st=path.steps[si];
              if(si===0){
                html+='<span class="ep-beam-step" style="font-weight:600">S'+st.state+'</span>';
              }else{
                html+='<span class="path-arrow">&rarr;</span>';
                if(st.action&&st.action!=='')html+='<span class="ep-beam-step" style="color:var(--ac)">'+st.action+'</span>';
                html+='<span class="ep-beam-step">S'+st.state+'</span>';
              }
            }
            html+='</div>';
          }
        }else if(pl.action_plan&&pl.action_plan.length>0){
          html+='<div class="ep-beam-path chosen">';
          html+='<span class="path-label">[选中] 1</span>';
          html+='<br>';
          for(var a=0;a<pl.action_plan.length;a++){
            if(a===0){
              html+='<span class="ep-beam-step" style="font-weight:600">S'+pl.planned_states[a]+'</span>';
            }else{
              html+='<span class="path-arrow">&rarr;</span>';
              html+='<span class="ep-beam-step" style="color:var(--ac)">'+pl.action_plan[a]+'</span>';
              html+='<span class="ep-beam-step">S'+pl.planned_states[a]+'</span>';
            }
          }
          html+='</div>';
        }
        if(pl.beam_results&&pl.beam_results.length>0){
          html+='<details style="margin-top:4px"><summary style="font-size:10px;color:var(--tx2);cursor:pointer">Beam Search ('+pl.beam_results.length+' 节点)</summary>';
          html+='<table class="ep-beam-table"><tr><th>Step</th><th>State</th><th>Action</th><th>Beam</th><th>WR</th><th>Quality</th><th>CumP</th></tr>';
          for(var b=0;b<pl.beam_results.length;b++){
            var br=pl.beam_results[b];
            var wrStr=(br.win_rate*100).toFixed(1)+'%';
            var qsStr=br.quality_score.toFixed(1);
            var cpStr=br.cumulative_probability.toFixed(4);
            html+='<tr><td>'+br.step+'</td><td>S'+br.state+'</td><td>'+(br.action||'-')+'</td><td>B'+br.beam_id+'</td><td>'+wrStr+'</td><td>'+qsStr+'</td><td>'+cpStr+'</td></tr>';
          }
          html+='</table></details>';
        }
        html+='</div>';
        html+='</details>';
      }
      html+='</details>';
    }
    html+='</div>';
  }
  list.innerHTML=html;
  var totalPages=data.total_pages||1;
  if(totalPages<=1){pgn.innerHTML='';return}
  var phtml='';
  for(var p=1;p<=totalPages;p++){
    if(p===epCurrentPage)phtml+='<button class="ep-page-btn active">'+p+'</button>';
    else phtml+='<button class="ep-page-btn" onclick="epCurrentPage='+p+';renderLocalEpisodes()">'+p+'</button>';
  }
  pgn.innerHTML=phtml;
}

async function init(){
  if(localStorage.getItem('live_theme')!=='dark'){document.documentElement.classList.add('light');document.getElementById('theme-btn').innerHTML='&#9728;'}
  try{await refreshAutopilotStatus()}catch(e){}
  loadFilterConfig();
  loadWindowPos();
  await refresh();
  startTimer();
  startLogTimer();
  fetchLogs();
  loadEpisodes();
}
init();
</script>
</body></html>"""
    return _html.replace("__PORT__", str(port))


def _render_live_game_content():
    port = int(st.session_state.get("live_port", 8000))
    html = _build_live_game_html(port)
    st.components.v1.html(html, height=950, scrolling=False)


def _run_rollout(kg_data, transitions, kg_entry, kg):
    start_state = st.session_state.get("roll_state", 0)
    score_mode = st.session_state.get("roll_sm", "quality")
    beam_width = st.session_state.get("roll_bw", 3)
    lookahead_steps = st.session_state.get("roll_la", 5)
    min_cum_prob = st.session_state.get("roll_mcp", 0.01)
    min_visits_roll = st.session_state.get("roll_mv", 1)
    max_state_revisits = st.session_state.get("roll_msr", 2)
    discount_factor = st.session_state.get("roll_df", 0.9)
    action_strategy = st.session_state.get("roll_as", "best_beam")
    next_state_mode = st.session_state.get("roll_nsm", "sample")
    epsilon = st.session_state.get("roll_eps", 0.1)
    max_rollout_steps = st.session_state.get("roll_mrs", 50)
    rng_seed_str = st.session_state.get("roll_seed", "42")
    rollout_mode = st.session_state.get("roll_mode", "单步推演")
    enable_backup = st.session_state.get("roll_backup", False)
    backup_score_threshold = st.session_state.get("roll_backup_st", 0.3)
    backup_dist_threshold = st.session_state.get("roll_backup_dt", 0.2)

    unique_states = kg_data.get("unique_states", set())
    if isinstance(unique_states, set):
        if start_state not in unique_states:
            st.warning(
                f"状态 {start_state} 不存在于当前经验转移图中。请选择一个有效状态。"
            )
            return

    if start_state not in transitions:
        st.warning(f"状态 {start_state} 没有转移数据，无法进行推演。")
        return

    kg_file = kg_entry.get("file", "")
    trans_file = kg_entry.get("transitions", "")

    try:
        rng_seed = int(rng_seed_str) if rng_seed_str.strip() else None
    except ValueError:
        rng_seed = None

    actual_rollout_mode = "multi_step" if rollout_mode == "多步推演" else "single_step"

    cached = _cached_rollout_results(
        kg_file,
        trans_file,
        start_state,
        score_mode,
        action_strategy,
        next_state_mode,
        beam_width,
        lookahead_steps,
        max_rollout_steps,
        min_visits_roll,
        min_cum_prob,
        max_state_revisits,
        discount_factor,
        epsilon,
        rng_seed,
        rollout_mode=actual_rollout_mode,
        enable_backup=enable_backup,
        score_threshold=backup_score_threshold,
        distance_threshold=backup_dist_threshold,
    )

    nodes = cached["nodes"]
    chosen_path_ids = cached["chosen_path_ids"]
    termination_reason = cached["termination_reason"]

    if len(chosen_path_ids) <= 1:
        st.error("推演无结果：该状态无可用动作或为终端状态。")
        return

    path_nodes = [nodes[str(nid)] for nid in chosen_path_ids]
    n_steps = len(path_nodes) - 1

    step_data = []
    for i in range(n_steps):
        prev = path_nodes[i]
        curr = path_nodes[i + 1]
        child_actions = set()
        for cid in prev.get("children_ids", []):
            c = nodes.get(str(cid))
            if c and c.get("action"):
                child_actions.add(c["action"])
        step_data.append(
            {
                "step": i + 1,
                "state": prev["state"],
                "action": curr["action"],
                "next_state": curr["state"],
                "win_rate": curr["win_rate"],
                "quality_score": curr["quality_score"],
                "avg_future_reward": curr["avg_future_reward"],
                "avg_step_reward": curr["avg_step_reward"],
                "transition_prob": curr["transition_prob"],
                "cumulative_probability": curr["cumulative_probability"],
                "visits": curr["visits"],
                "n_candidates": len(child_actions),
                "is_terminal": curr["is_terminal"],
            }
        )

    st.markdown(f"**起始状态**: `{start_state}`")

    col_summary, col_mid, col_traj = st.columns([0.2, 0.6, 0.8])

    with col_summary:
        st.markdown("**📋 推演摘要**")
        last_step = step_data[-1]
        st.metric("总步数", n_steps)
        st.metric("终止原因", termination_reason)
        st.metric(
            "起止状态",
            f"{path_nodes[0]['state']} → {last_step['next_state']}",
        )
        avg_wr = sum(s["win_rate"] for s in step_data) / n_steps
        avg_qs = sum(s["quality_score"] for s in step_data) / n_steps
        st.metric("平均胜率", f"{avg_wr:.2%}")
        st.metric("平均 Quality", f"{avg_qs:.1f}")

        if actual_rollout_mode == "multi_step":
            st.divider()
            st.caption("**多步推演统计**")
            plan_segments = cached.get("plan_segments", [])
            st.metric("规划段数", len(plan_segments))
            st.metric("重新规划次数", cached.get("total_re_searches", 0))
            st.metric("备选切换次数", cached.get("total_backup_switches", 0))
            main_steps = 0
            total_planned = 0
            for seg in plan_segments:
                total_planned += len(seg.get("actions_planned", []))
                if seg.get("divergence_type") in ("none", "backup_switch"):
                    main_steps += len(seg.get("actions_planned", []))
                elif seg.get("divergence_step", -1) >= 0:
                    main_steps += seg["divergence_step"]
            if total_planned > 0:
                st.metric("路径命中率", f"{main_steps / total_planned:.0%}")

    with col_mid:
        st.markdown("**📈 胜率/质量趋势**")
        step_nums = [s["step"] for s in step_data]
        win_rates = [s["win_rate"] * 100 for s in step_data]
        qualities = [s["quality_score"] for s in step_data]

        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=step_nums,
                y=win_rates,
                mode="lines+markers",
                name="Win Rate (%)",
                line=dict(color="#4CAF50", width=2),
                marker=dict(size=4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=step_nums,
                y=qualities,
                mode="lines+markers",
                name="Quality Score",
                line=dict(color="#2196F3", width=2),
                marker=dict(size=4),
                yaxis="y2",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[step_nums[0], step_nums[-1]],
                y=[50, 50],
                mode="lines",
                name="50% 基准",
                line=dict(color="#4CAF50", width=1, dash="dash"),
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[step_nums[0], step_nums[-1]],
                y=[0, 0],
                mode="lines",
                name="0 基准",
                line=dict(color="#4CAF50", width=1, dash="dash"),
                hoverinfo="skip",
                yaxis="y2",
            )
        )
        fig.update_layout(
            height=300,
            margin=dict(l=50, r=50, t=30, b=40),
            xaxis=dict(title="Step", gridcolor="#e0e0e0"),
            yaxis=dict(title="Win Rate (%)", gridcolor="#e0e0e0", domain=[0, 1]),
            yaxis2=dict(
                title="Quality",
                gridcolor="#e0e0e0",
                overlaying="y",
                side="right",
                anchor="x",
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            font=dict(family="SimHei, Microsoft YaHei, sans-serif", size=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True, height=320)

        st.markdown("**🔍 片段匹配**")
        map_id = kg_entry.get("map_id", "")
        data_id = kg_entry.get("data_id", "")
        match_rows = []
        dist_mat = None

        if map_id and data_id:
            with st.spinner("匹配原始对局片段..."):
                ep_data = load_episode_data(map_id, data_id)
                dist_mat = load_distance_matrix_np(map_id, data_id)

            if ep_data["n_episodes"] > 0:
                state_seq = [pn["state"] for pn in path_nodes]
                action_seq = [s["action"] for s in step_data]

                rollout_beam_paths = {
                    0: [
                        BeamSearchResult(
                            step=i,
                            state=state_seq[i],
                            action=action_seq[i] if i < len(action_seq) else "",
                            cumulative_probability=path_nodes[i][
                                "cumulative_probability"
                            ],
                            quality_score=path_nodes[i]["quality_score"],
                            win_rate=path_nodes[i]["win_rate"],
                            avg_step_reward=path_nodes[i]["avg_step_reward"],
                            avg_future_reward=path_nodes[i]["avg_future_reward"],
                            beam_id=0,
                            parent_idx=i - 1 if i > 0 else None,
                        )
                        for i in range(len(state_seq))
                    ]
                }

                match_results = match_beam_paths(
                    rollout_beam_paths,
                    ep_data["states"],
                    ep_data["actions"],
                    ep_data["outcomes"],
                    ep_data["scores"],
                    dist_mat,
                    top_k=5,
                )

                for bid in sorted(match_results.keys()):
                    for rank, m in enumerate(match_results[bid]):
                        match_rows.append(
                            {
                                "Rank": rank + 1,
                                "Episode": m.episode_id,
                                "位置": f"{m.start_pos}~{m.end_pos}",
                                "状态相似度": f"{m.state_similarity:.1%}",
                                "动作匹配率": f"{m.action_match_rate:.0%}",
                                "综合置信度": f"{m.combined_score:.1%}",
                                "结果": m.outcome,
                                "得分": m.episode_score,
                            }
                        )

        if match_rows:
            st.dataframe(
                match_rows,
                use_container_width=True,
                height=min(len(match_rows) * 35 + 50, 300),
            )
            wins = sum(1 for r in match_rows if r["结果"] == "Win")
            total_m = len(match_rows)
            st.caption(
                f"共 {total_m} 条匹配 | Win: {wins} ({wins / total_m:.0%}) | "
                f"距离矩阵: {'有' if dist_mat is not None else '无(精确匹配)'}"
            )
        else:
            st.info("无匹配结果。原始对局数据可能未加载。")

    with col_traj:
        st.markdown("**📋 推演轨迹**")

        plan_segments = cached.get("plan_segments", [])
        step_segment_map = {}
        if actual_rollout_mode == "multi_step" and plan_segments:
            global_step = 0
            for seg_idx, seg in enumerate(plan_segments):
                actions = seg.get("actions_planned", [])
                div_step = seg.get("divergence_step", -1)
                div_type = seg.get("divergence_type", "none")
                for act_idx in range(len(actions)):
                    step_segment_map[global_step] = {
                        "segment": seg_idx,
                        "seg_type": seg.get("segment_type", ""),
                        "divergence": act_idx == div_step
                        if div_type != "none"
                        else False,
                        "div_type": div_type if act_idx == div_step else "",
                    }
                    global_step += 1

        _SEG_TYPE_LABELS = {
            "initial_plan": "初始规划",
            "re_search": "重新规划",
            "backup_switch": "备选切换",
        }

        traj_rows = []
        for idx, s in enumerate(step_data):
            seg_info = step_segment_map.get(idx, {})
            row = {
                "Step": s["step"],
                "State": s["state"],
                "Action": s["action"],
                "Next State": s["next_state"],
                "Win Rate": f"{s['win_rate']:.2%}",
                "Quality": f"{s['quality_score']:.1f}",
                "Future Reward": f"{s['avg_future_reward']:.2f}",
                "Cum. Prob": f"{s['cumulative_probability']:.4f}",
                "Trans. Prob": f"{s['transition_prob']:.2%}",
                "Visits": s["visits"],
                "Candidates": s["n_candidates"],
                "Terminal": "⭕" if s["is_terminal"] else "",
            }
            if actual_rollout_mode == "multi_step" and seg_info:
                row["推演段"] = seg_info.get("segment", "")
                seg_type = seg_info.get("seg_type", "")
                row["段类型"] = _SEG_TYPE_LABELS.get(seg_type, seg_type)
                if seg_info.get("divergence"):
                    div_t = seg_info.get("div_type", "")
                    div_label = {
                        "re_search": "🔄 重规划",
                        "backup_switch": "🔀 备选切换",
                        "no_valid_transition": "❌ 无转移",
                        "low_cum_prob": "⚠️ 低概率",
                    }.get(div_t, div_t)
                    row["段类型"] += f" [{div_label}]"
            traj_rows.append(row)

        st.dataframe(
            traj_rows,
            use_container_width=True,
            height=min(len(traj_rows) * 35 + 50, 600),
        )

    st.divider()
    st.subheader("🔍 束搜索追溯")

    step_options = []
    for i, s in enumerate(step_data):
        label = f"Step {s['step']}: S{s['state']} →({s['action']})→ S{s['next_state']}"
        if s["is_terminal"]:
            label += " [终端]"
        step_options.append(label)

    if not step_options:
        st.info("无推演步骤可供追溯。")
        return

    col_steps, col_tree, col_rec = st.columns([0.2, 0.5, 0.8])

    with col_steps:
        selected_step = st.radio(
            "选择步骤",
            options=list(range(len(step_options))),
            format_func=lambda x: step_options[x],
            key="beam_trace_step",
        )

    beam_results_by_step = cached.get("beam_results_by_step", {})
    raw_beam = beam_results_by_step.get(str(selected_step), [])

    beam_results_list = (
        [
            BeamSearchResult(
                step=r["step"],
                state=r["state"],
                action=r["action"],
                cumulative_probability=r["cumulative_probability"],
                quality_score=r["quality_score"],
                win_rate=r["win_rate"],
                avg_step_reward=r["avg_step_reward"],
                avg_future_reward=r["avg_future_reward"],
                beam_id=r["beam_id"],
                parent_idx=r["parent_idx"],
            )
            for r in raw_beam
        ]
        if raw_beam
        else []
    )

    sub_beam_paths = get_beam_paths(beam_results_list) if beam_results_list else []
    sub_composites, sub_path_metrics = (
        _compute_composite_scores(sub_beam_paths) if sub_beam_paths else ([], [])
    )
    sub_sorted_indices = (
        sorted(
            range(len(sub_beam_paths)),
            key=lambda i: sub_composites[i],
            reverse=True,
        )
        if sub_composites
        else []
    )

    _roll_sel_key = "roll_beam_selected_path"
    _roll_sel_step_key = "roll_beam_selected_step"
    prev_sel_step = st.session_state.get(_roll_sel_step_key, -1)
    if prev_sel_step != selected_step:
        if sub_sorted_indices:
            st.session_state[_roll_sel_key] = sub_sorted_indices[0]
        else:
            st.session_state[_roll_sel_key] = 0
        st.session_state[_roll_sel_step_key] = selected_step
    if _roll_sel_key not in st.session_state:
        st.session_state[_roll_sel_key] = (
            sub_sorted_indices[0] if sub_sorted_indices else 0
        )

    highlight_set = set()
    if sub_beam_paths and st.session_state[_roll_sel_key] < len(sub_beam_paths):
        sel_path = sub_beam_paths[st.session_state[_roll_sel_key]]
        highlight_set = {i for i, r in enumerate(beam_results_list) if r in sel_path}

    chosen_action = step_data[selected_step]["action"]
    chosen_next_state = step_data[selected_step]["next_state"]

    with col_tree:
        if beam_results_list:
            html = render_beam_tree(beam_results_list, highlight_indices=highlight_set)
            html_b64 = base64.b64encode(html.encode()).decode()
            html_uri = f"data:text/html;charset=utf-8;base64,{html_b64}#v=1"
            sd = step_data[selected_step]
            subtree_json = _results_to_json(
                beam_results_list,
                beam_width,
                lookahead_steps,
                min_cum_prob,
                score_mode,
                sd["state"],
            )
            json_b64 = base64.b64encode(subtree_json.encode()).decode()
            json_uri = f"data:application/json;base64,{json_b64}"
            st.markdown(
                f"**🌳 子树路径图**  "
                f'<a href="{html_uri}" '
                f'download="beam_subtree_step{sd["step"]}_S{sd["state"]}.html" '
                f'style="font-size:0.85em;color:#4CAF50;text-decoration:none;margin-right:12px;">📥 导出 HTML</a>'
                f'<a href="{json_uri}" '
                f'download="beam_subtree_step{sd["step"]}_S{sd["state"]}.json" '
                f'style="font-size:0.85em;color:#2196F3;text-decoration:none;">📥 导出 JSON</a>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="overflow:hidden;width:100%;height:520px;'
                f'border:1px solid #444;border-radius:4px;">'
                f'<iframe src="{html_uri}" width="100%" height="100%" '
                f'style="border:none;display:block;"></iframe>'
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("**🌳 子树路径图**")
            st.info("该步骤无束搜索子树数据。")

    with col_rec:
        st.markdown("**📊 路径推荐**")
        if sub_beam_paths:
            rec_rows = _build_rec_rows(
                sub_sorted_indices, sub_beam_paths, sub_composites, sub_path_metrics
            )
            event = st.dataframe(
                rec_rows,
                use_container_width=True,
                height=min(len(rec_rows) * 35 + 50, 500),
                on_select="rerun",
                selection_mode="single-row",
            )
            if event and event.selection.rows:
                new_path = rec_rows[event.selection.rows[0]]["Path"]
                old_path = st.session_state[_roll_sel_key]
                if new_path != old_path:
                    st.session_state[_roll_sel_key] = new_path
                    st.rerun()
        else:
            st.info("无路径数据。")

    st.divider()
    st.markdown("**📋 路径详情**")
    if sub_beam_paths:
        detail_rows = _build_path_detail_rows(sub_beam_paths)
        chosen_path_idx = st.session_state.get(_roll_sel_key, 0)
        st.caption(f"当前选中路径: **#{chosen_path_idx}**")
        st.dataframe(
            detail_rows,
            use_container_width=True,
            height=min(len(detail_rows) * 35 + 50, 500),
        )
    else:
        st.info("无路径数据。")

    st.divider()

    with st.expander("🌐 完整推演树全局视图", expanded=False):
        g = nx.DiGraph()
        chosen_set = set(chosen_path_ids)
        max_beam_id = 0
        for nid_key, n in nodes.items():
            bid = n.get("beam_id")
            if bid is not None and bid > max_beam_id:
                max_beam_id = bid

        for nid_key, n in nodes.items():
            nid = n["id"]
            is_chosen = nid in chosen_set
            is_term = n.get("is_terminal", False)

            if is_term:
                node_color = "rgba(255, 80, 80, 0.9)"
                shape = "diamond"
                bw = 3
            elif is_chosen:
                node_color = "#4FC3F7"
                shape = "dot"
                bw = 2
            else:
                node_color = "#999999"
                shape = "dot"
                bw = 1

            label = f"S{n['state']}"
            if n.get("action"):
                label += f"\n{n['action']}"

            depth_tag = f"D{n['rollout_depth']}" if n["rollout_depth"] >= 0 else ""
            title_parts = [
                f"Node ID: {nid}",
                f"State: {n['state']}",
                f"Action: {n.get('action', '-')}",
                f"Beam: {n.get('beam_id', '-')}",
                f"Depth: {depth_tag}",
                f"Quality: {n['quality_score']:.1f}",
                f"Win Rate: {n['win_rate']:.2%}",
                f"Cum Prob: {n['cumulative_probability']:.4f}",
                f"Chosen: {'Yes' if is_chosen else 'No'}",
            ]
            if is_term:
                title_parts.insert(1, "Terminal")
            if n.get("is_beam_root"):
                title_parts.append("Beam Root")

            g.add_node(
                nid,
                label=label,
                color=node_color,
                shape=shape,
                size=20 if is_chosen else 12,
                borderWidth=bw,
                title="\n".join(title_parts),
                font={"size": 10, "color": "white" if is_chosen else "#cccccc"},
            )

        for nid_key, n in nodes.items():
            parent_id = n.get("parent_id")
            if parent_id is not None and str(parent_id) in nodes:
                child_nid = n["id"]
                is_chosen_edge = parent_id in chosen_set and child_nid in chosen_set
                edge_color = "#4FC3F7" if is_chosen_edge else "#666666"
                edge_width = 2 if is_chosen_edge else 0.5
                action_label = n.get("action", "")
                g.add_edge(
                    parent_id,
                    child_nid,
                    label=action_label,
                    color=edge_color,
                    width=edge_width,
                    arrows="to",
                    font={"size": 8, "color": "#aaaaaa"},
                    smooth={"type": "continuous"},
                )

        net_full = Network(height="700px", width="100%", directed=True, notebook=False)
        for nid, ndata in g.nodes(data=True):
            net_full.add_node(nid, **ndata)
        for src, tgt, edata in g.edges(data=True):
            net_full.add_edge(src, tgt, **edata)

        options_full = {
            "physics": {"enabled": False},
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "direction": "LR",
                    "sortMethod": "directed",
                    "levelSeparation": 200,
                    "nodeSpacing": 120,
                    "blockShifting": True,
                    "edgeMinimization": True,
                }
            },
            "edges": {"smooth": {"type": "continuous", "roundness": 0.15}},
            "interaction": {"hover": True, "tooltipDelay": 50},
        }
        net_full.set_options(json.dumps(options_full))

        raw_html_full = net_full.generate_html(notebook=False)
        resize_full = """<script>
(function() {
    document.querySelectorAll('center').forEach(function(el) { el.remove(); });
    document.documentElement.style.cssText = 'height:100%; margin:0; overflow:hidden;';
    document.body.style.cssText = 'height:100%; margin:0; padding:0; overflow:hidden;';
    var card = document.querySelector('.card');
    if (card) card.style.cssText = 'height:100%; width:100%; padding:0; margin:0;';
    var c = document.getElementById('mynetwork');
    if (!c) return;
    c.style.cssText = 'width:100%; height:100%; position:relative; float:none; border:none;';
    setTimeout(function() {
        if (window.network) {
            window.network.setSize('100%', '100%');
            window.network.redraw();
        }
    }, 500);
})();
</script>"""
        full_tree_html = raw_html_full.replace("</body>", resize_full + "</body>")
        st.components.v1.html(full_tree_html, height=720)

        st.caption(
            "蓝色=主路径 | 灰色=探索分支 | 红色菱形=终端 | "
            f"总节点: {len(nodes)} | 主路径: {len(chosen_path_ids)} 节点"
        )

    with st.expander("💾 导出推演树 JSON", expanded=False):
        tree_json_nodes = []
        for nid_key in sorted(
            nodes.keys(), key=lambda k: int(k) if str(k).isdigit() else k
        ):
            n = nodes[nid_key]
            tree_json_nodes.append(
                {
                    "id": n["id"],
                    "parent_id": n["parent_id"],
                    "children_ids": n["children_ids"],
                    "state": n["state"],
                    "action": n["action"],
                    "beam_id": n["beam_id"],
                    "quality_score": n["quality_score"],
                    "win_rate": n["win_rate"],
                    "avg_future_reward": n["avg_future_reward"],
                    "avg_step_reward": n["avg_step_reward"],
                    "visits": n["visits"],
                    "transition_prob": n["transition_prob"],
                    "cumulative_probability": n["cumulative_probability"],
                    "rollout_depth": n["rollout_depth"],
                    "is_on_chosen_path": n["is_on_chosen_path"],
                    "is_terminal": n["is_terminal"],
                    "is_beam_root": n["is_beam_root"],
                }
            )

        tree_json = json.dumps(
            {
                "meta": {
                    "start_state": start_state,
                    "action_strategy": action_strategy,
                    "score_mode": score_mode,
                    "beam_width": beam_width,
                    "lookahead_steps": lookahead_steps,
                    "max_rollout_steps": max_rollout_steps,
                    "rng_seed": rng_seed,
                    "total_nodes": len(nodes),
                    "chosen_path_length": len(chosen_path_ids),
                    "termination_reason": termination_reason,
                },
                "chosen_path_ids": chosen_path_ids,
                "nodes": tree_json_nodes,
            },
            indent=2,
            ensure_ascii=False,
        )

        b64 = base64.b64encode(tree_json.encode("utf-8")).decode()
        dl_href = f'<a href="data:application/json;base64,{b64}" download="rollout_tree_S{start_state}.json">📥 下载 rollout_tree_S{start_state}.json</a>'
        st.markdown(dl_href, unsafe_allow_html=True)
        st.caption(
            f"JSON 大小: {len(tree_json):,} 字符 ({len(tree_json_nodes)} 个节点)"
        )
        st.code(tree_json[:2000] + ("..." if len(tree_json) > 2000 else ""))


_run_rollout = st.fragment(_run_rollout)


def _render_rollout_tab(kg_data, transitions, kg_entry, kg):
    st.markdown("### 从指定起始状态出发，按策略逐步滚动推演至终端状态。")
    _run_rollout(kg_data, transitions, kg_entry, kg)


def _render_raw_data_tab(kg_entry):
    map_id = kg_entry.get("map_id", "")
    data_id = kg_entry.get("data_id", "")

    if not map_id or not data_id:
        st.warning("当前 KG 条目缺少 map_id 或 data_id。")
        return

    ep_data = load_episode_data(map_id, data_id)
    n_ep = ep_data["n_episodes"]
    outcomes = ep_data["outcomes"]
    scores = ep_data["scores"]
    ep_states = ep_data["states"]
    ep_actions = ep_data["actions"]

    if n_ep == 0:
        st.error("未加载到原始对局数据。请确认 data/ 目录下有对应文件。")
        return

    n_win = sum(1 for o in outcomes if o == "Win")
    n_loss = sum(1 for o in outcomes if o == "Loss")
    avg_len = sum(len(s) for s in ep_states) / n_ep if n_ep > 0 else 0
    avg_score = sum(scores) / len(scores) if scores else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总 Episodes", n_ep)
    c2.metric("Win", n_win, f"{n_win / n_ep:.1%}" if n_ep else "")
    c3.metric("Loss", n_loss, f"{n_loss / n_ep:.1%}" if n_ep else "")
    c4.metric("平均步数 / 平均得分", f"{avg_len:.1f}", f"{avg_score:.1f}")

    c_ep, c_dist = st.columns(2)
    with c_ep:
        _run_episode_query(n_ep, outcomes, scores, ep_states, ep_actions)
    with c_dist:
        _run_distance_query(map_id, data_id)

    c_hp, c_mds = st.columns(2)
    with c_hp:
        _run_hp_query(map_id, data_id)
    with c_mds:
        _run_mds_terrain(map_id, data_id, len(ep_data.get("states", [])))


def _results_to_json(
    results: list,
    beam_width: int,
    max_steps: int,
    min_cum_prob: float,
    score_mode: str,
    start_state: int,
) -> str:
    nodes = []
    for i, r in enumerate(results):
        nodes.append(
            {
                "index": i,
                "step": r.step,
                "state": r.state,
                "action": r.action,
                "cumulative_probability": r.cumulative_probability,
                "quality_score": r.quality_score,
                "win_rate": r.win_rate,
                "avg_step_reward": r.avg_step_reward,
                "avg_future_reward": r.avg_future_reward,
                "beam_id": r.beam_id,
                "parent_idx": r.parent_idx,
            }
        )
    edges = [
        {"source": r.parent_idx, "target": idx}
        for idx, r in enumerate(results)
        if r.parent_idx is not None
    ]
    return json.dumps(
        {
            "meta": {
                "start_state": start_state,
                "beam_width": beam_width,
                "max_steps": max_steps,
                "min_cum_prob": min_cum_prob,
                "score_mode": score_mode,
                "total_nodes": len(results),
            },
            "nodes": nodes,
            "edges": edges,
        },
        indent=2,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    main()
