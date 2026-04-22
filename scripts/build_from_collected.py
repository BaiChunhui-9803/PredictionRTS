#!/usr/bin/env python
"""
Build ETG (Experience Transition Graph) from collected replay data.

Usage:
    python scripts/build_from_collected.py \
        --input output/collected_data/collected_data_*.pkl \
        --bktree-dir output/collected_data \
        --output-dir cache/knowledge_graph/MarineMicro_MvsM_4_augmented/ \
        --validate
"""

import sys
import os
import json
import pickle
import argparse
import logging
from typing import List, Dict, Optional, Tuple, Any, Set
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ROOT_DIR
from src.structure.BKTree_sc2 import (
    ClusterNode,
    BKTree,
    classify_new_state,
    get_max_cluster_id,
)
from src.structure.custom_distance_sc2 import CustomDistance
from src.data.loader import DataLoader
from src.decision.knowledge_graph import DecisionKnowledgeGraph

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ACTION_MAP = {
    "a": "action_ATK_nearest",
    "b": "action_ATK_clu_nearest",
    "c": "action_ATK_nearest_weakest",
    "d": "action_ATK_clu_nearest_weakest",
    "e": "action_ATK_threatening",
    "f": "action_DEF_clu_nearest",
    "g": "action_MIX_gather",
    "h": "action_MIX_lure",
    "i": "action_MIX_sacrifice_lure",
    "j": "do_randomly",
    "k": "do_nothing",
}


def load_collected_data(path: str) -> Dict[str, Any]:
    logger.info(f"Loading collected data from {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(
        f"  Episodes: {data['total_episodes']}, "
        f"Source: {data.get('source_action_log', '?')}"
    )
    return data


def build_cluster_mapping(
    episodes: List[Dict], bktree_dir: str
) -> Tuple[Dict[Tuple[int, int], int], Dict[int, Tuple[int, int]]]:
    state_to_node: Dict[Tuple[int, int], int] = {}
    node_to_state: Dict[int, Tuple[int, int]] = {}
    next_node_id = 0

    for ep in episodes:
        for frame in ep["frames"]:
            cluster = frame.get("state_cluster")
            if cluster is None:
                continue
            key = (int(cluster[0]), int(cluster[1]))
            if key not in state_to_node:
                state_to_node[key] = next_node_id
                node_to_state[next_node_id] = key
                next_node_id += 1

    logger.info(
        f"  Cluster mapping: {len(state_to_node)} unique state clusters -> node_ids 0~{next_node_id - 1}"
    )

    state_node_path = Path(bktree_dir) / "state_node.txt"
    if state_node_path.exists():
        logger.info(f"  Loading state_node.txt from {state_node_path}")
        count_loaded = 0
        with open(state_node_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                try:
                    key = eval(parts[0])
                    nid = int(parts[1])
                    if key not in state_to_node:
                        state_to_node[key] = nid
                        node_to_state[nid] = key
                        count_loaded += 1
                except Exception:
                    continue
        logger.info(f"  Loaded {count_loaded} additional entries from state_node.txt")

    return state_to_node, node_to_state


def build_episodes_arrays(
    episodes: List[Dict], state_to_node: Dict[Tuple[int, int], int]
) -> Tuple[List[List[int]], List[List[str]], List[List[float]], List[str]]:
    state_episodes: List[List[int]] = []
    action_episodes: List[List[str]] = []
    reward_episodes: List[List[float]] = []
    outcome_episodes: List[str] = []

    skipped = 0
    for ep in episodes:
        if not ep["frames"]:
            skipped += 1
            continue

        states: List[int] = []
        actions: List[str] = []
        rewards: List[float] = []
        prev_score: Optional[float] = None

        for frame in ep["frames"]:
            cluster = frame.get("state_cluster")
            if cluster is None:
                continue
            key = (int(cluster[0]), int(cluster[1]))
            nid = state_to_node.get(key)
            if nid is None:
                continue
            states.append(nid)

            action_code = frame.get("action_code", "")
            if action_code and len(action_code) >= 2:
                letter = action_code[1].lower()
                action_name = ACTION_MAP.get(letter, "do_nothing")
            else:
                action_name = "do_nothing"
            actions.append(action_name)

            current_score = float(frame.get("hp_my", 0)) - float(
                frame.get("hp_enemy", 0)
            )
            if prev_score is None:
                rewards.append(0.0)
            else:
                rewards.append(current_score - prev_score)
            prev_score = current_score

        if states:
            state_episodes.append(states)
            action_episodes.append(actions)
            reward_episodes.append(rewards)
            outcome_episodes.append(ep.get("result", "Unknown"))

    logger.info(
        f"  Built {len(state_episodes)} valid episodes (skipped {skipped} empty)"
    )
    return state_episodes, action_episodes, reward_episodes, outcome_episodes


def build_transitions(
    state_episodes: List[List[int]],
    action_episodes: List[List[str]],
    outcome_episodes: List[str],
    output_dir: Path,
    unique_states: Optional[Set[int]] = None,
) -> Dict:
    logger.info("Building transitions...")
    transitions: Dict = defaultdict(dict)

    for ep_idx in range(len(state_episodes)):
        states = state_episodes[ep_idx]
        actions = action_episodes[ep_idx] if ep_idx < len(action_episodes) else []
        outcome = (
            outcome_episodes[ep_idx] if ep_idx < len(outcome_episodes) else "Unknown"
        )

        for t in range(len(states) - 1):
            state = states[t]
            action = actions[t] if t < len(actions) else "do_nothing"
            next_state = states[t + 1]

            if action not in transitions[state]:
                transitions[state][action] = {
                    "next_states": defaultdict(int),
                    "wins": 0,
                    "total": 0,
                }

            transitions[state][action]["next_states"][next_state] += 1
            transitions[state][action]["total"] += 1
            if outcome.lower() == "win":
                transitions[state][action]["wins"] += 1

    for state in transitions:
        for action in transitions[state]:
            trans = transitions[state][action]
            trans["next_states"] = dict(trans["next_states"])
            trans["win_rate"] = (
                trans["wins"] / trans["total"] if trans["total"] > 0 else 0.0
            )

    transitions = dict(transitions)

    if unique_states is not None:
        terminal_count = 0
        for state_id in unique_states:
            if state_id not in transitions:
                transitions[state_id] = {"__terminal__": True}
                terminal_count += 1
        logger.info(f"  Marked {terminal_count} terminal states")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "kg_simple_transitions.pkl"
    with open(path, "wb") as f:
        pickle.dump(transitions, f)

    logger.info(f"  Transitions saved to {path} ({len(transitions)} states)")
    return transitions


def build_knowledge_graph(
    state_episodes: List[List[int]],
    action_episodes: List[List[str]],
    reward_episodes: List[List[float]],
    outcome_episodes: List[str],
    output_dir: Path,
    verbose: bool = True,
) -> DecisionKnowledgeGraph:
    logger.info("Building Knowledge Graph...")
    kg = DecisionKnowledgeGraph(
        use_context=False,
        context_window=0,
        action_format="cluster+letter",
    )

    kg.build_from_data(
        state_episodes=state_episodes,
        action_episodes=action_episodes,
        reward_episodes=reward_episodes,
        outcome_episodes=outcome_episodes,
        verbose=verbose,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "kg_simple.pkl"
    kg.save(str(output_path))

    stats = kg.get_statistics()
    if verbose:
        logger.info(f"  Total visits: {stats['total_visits']}")
        logger.info(f"  Unique states: {stats['unique_states']}")
        logger.info(f"  Unique actions: {stats['unique_actions']}")

    return kg


def compute_distance_matrix(
    bktree_dir: str,
    output_dir: Path,
    node_to_state: Dict[int, Tuple[int, int]],
):
    logger.info("Computing state distance matrix...")

    reverse_dict: Dict[int, Dict] = {}
    for nid, (p, s) in node_to_state.items():
        reverse_dict[nid] = {"cluster": (p, s), "score": 0.0}

    custom_distance = CustomDistance(threshold=0.5)

    secondary_bktrees: Dict[int, BKTree] = {}
    bktree_path = Path(bktree_dir)
    for f in bktree_path.glob("secondary_bktree_*.json"):
        cid_str = f.stem.replace("secondary_bktree_", "")
        try:
            cid = int(cid_str)
        except ValueError:
            continue
        tree = BKTree(custom_distance.multi_distance, distance_index=1)
        try:

            def deserialize_node(node_data):
                if node_data is None:
                    return None
                node = ClusterNode(node_data["state"], node_data["cluster_id"])
                for dist_key, child_data in node_data.get("children", {}).items():
                    dist_val = int(dist_key) if dist_key.isdigit() else float(dist_key)
                    child_node = deserialize_node(child_data)
                    if child_node is not None:
                        node.children[dist_val] = child_node
                return node

            with open(f, "r") as fh:
                tree_data = json.load(fh)
            tree.root = deserialize_node(tree_data)
            max_id = get_max_cluster_id(tree)
            tree.next_cluster_id = max_id + 1
            secondary_bktrees[cid] = tree
        except Exception as e:
            logger.warning(f"  Failed to load {f}: {e}")

    logger.info(f"  Loaded {len(secondary_bktrees)} secondary BKTrees")

    npy_dir = output_dir / "npy"
    npy_dir.mkdir(parents=True, exist_ok=True)

    try:
        from src.utils.calculate_utils import calculate_and_save_distance_matrix

        dist_matrix = calculate_and_save_distance_matrix(
            reverse_dict, custom_distance, secondary_bktrees, str(npy_dir)
        )
        logger.info(
            f"  Distance matrix computed: {dist_matrix.shape if dist_matrix is not None else 'N/A'}"
        )
    except Exception as e:
        logger.error(f"  Failed to compute distance matrix: {e}")


def validate_knowledge_graph(kg: DecisionKnowledgeGraph):
    logger.info("Validating Knowledge Graph...")
    test_states = list(kg.unique_states)[:10]
    for state in test_states:
        top_actions = kg.get_top_k_actions(state, k=5, metric="quality_score")
        if not top_actions:
            continue
        logger.info(f"  State {state}:")
        for action, stats in top_actions[:3]:
            logger.info(
                f"    {action}: visits={stats['visits']}, "
                f"win_rate={stats['win_rate'] * 100:.1f}%, "
                f"quality={stats['quality_score']:.1f}"
            )

    states_with_data = 0
    for state in kg.unique_states:
        top_actions = kg.get_top_k_actions(state, k=1)
        if top_actions:
            states_with_data += 1

    coverage = states_with_data / len(kg.unique_states) * 100 if kg.unique_states else 0
    logger.info(
        f"  Coverage: {states_with_data}/{len(kg.unique_states)} ({coverage:.1f}%)"
    )


def save_state_node_txt(state_to_node: Dict[Tuple[int, int], int], output_dir: Path):
    path = output_dir / "state_node.txt"
    sorted_items = sorted(state_to_node.items(), key=lambda x: x[1])
    with open(path, "w", encoding="utf-8") as f:
        for (p, s), nid in sorted_items:
            f.write(f"({p}, {s})\t{nid}\t0.0\n")
    logger.info(f"  Saved state_node.txt ({len(sorted_items)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Build ETG from collected replay data")
    parser.add_argument("--input", required=True, help="Collected data .pkl path")
    parser.add_argument(
        "--bktree-dir",
        required=True,
        help="Directory containing BKTree JSON files from ReplayCollector",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: cache/knowledge_graph/{name}_augmented/)",
    )
    parser.add_argument(
        "--context-windows", type=int, nargs="+", default=[0], help="Context windows"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate KG after building",
    )
    args = parser.parse_args()

    data = load_collected_data(args.input)
    episodes = data["episodes"]

    if args.output_dir is None:
        source = data.get("source_action_log", "unknown")
        map_id = "augmented"
        for part in source.replace("\\", "/").split("/"):
            if "MarineMicro" in part or "marine" in part.lower():
                map_id = part
                break
        args.output_dir = str(
            ROOT_DIR / "cache" / "knowledge_graph" / f"{map_id}_augmented"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    state_to_node, node_to_state = build_cluster_mapping(episodes, args.bktree_dir)
    save_state_node_txt(state_to_node, output_dir)

    state_episodes, action_episodes, reward_episodes, outcome_episodes = (
        build_episodes_arrays(episodes, state_to_node)
    )

    for context_window in args.context_windows:
        transitions = build_transitions(
            state_episodes=state_episodes,
            action_episodes=action_episodes,
            outcome_episodes=outcome_episodes,
            output_dir=output_dir,
            unique_states=set(state_to_node.values()),
        )

        kg = build_knowledge_graph(
            state_episodes=state_episodes,
            action_episodes=action_episodes,
            reward_episodes=reward_episodes,
            outcome_episodes=outcome_episodes,
            output_dir=output_dir,
        )

        if args.validate:
            validate_knowledge_graph(kg)

    compute_distance_matrix(args.bktree_dir, output_dir, node_to_state)

    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Episodes: {len(episodes)}")
    logger.info(f"Unique states: {len(state_to_node)}")
    logger.info(f"Transitions: {len(transitions)}")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
