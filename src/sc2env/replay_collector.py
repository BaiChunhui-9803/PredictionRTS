#!/usr/bin/env python
"""
ReplayCollector: 批量回放 action 序列，收集 norm_state 数据并增量构建 BKTree。

继承 SmartAgent，复用其 BKTree 和动作执行能力，
在 step() 中只做：获取 norm_state → 聚类 → 执行 action → 记录帧。

不执行 Q-learning、不写文件、不依赖 KG/transitions/beam search。
"""

import os
import sys
import json
import csv
import pickle
import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from pysc2.lib import actions as sc2_actions

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.sc2env.agent import SmartAgent
from src.sc2env.config import get_map_config
from src.structure.BKTree_sc2 import ClusterNode

_MAP_CONFIG, _MAP, _ENV_CONFIG, _ALG_CONFIG, _PATH_CONFIG = get_map_config("sce-1")

logger = logging.getLogger(__name__)


class ReplayCollector(SmartAgent):
    def __init__(
        self,
        bridge=None,
        action_log_path: str = "",
        replay_count: int = 3,
        batch_start: int = 0,
        batch_end: Optional[int] = None,
        output_dir: Optional[str] = None,
        primary_threshold: float = 1.0,
        secondary_threshold: float = 0.5,
    ):
        self.bridge = bridge
        self._action_log_path = action_log_path
        self._replay_count = replay_count
        self._batch_start = batch_start
        self._batch_end = batch_end
        self._base_output_dir = output_dir or "output/collected_data"
        self._primary_threshold = primary_threshold
        self._secondary_threshold = secondary_threshold

        self._all_sequences: List[List[str]] = []
        self._current_seq_idx: int = batch_start
        self._current_run_idx: int = 0
        self._replay_actions: List[str] = []
        self._replay_idx: int = 0
        self._done: bool = False

        self._collected_episodes: List[Dict[str, Any]] = []
        self._current_frames: List[Dict[str, Any]] = []
        self._completed_episodes: int = 0
        self._total_episodes: int = 0
        self._seq_stats: Dict[int, Dict[str, list]] = defaultdict(
            lambda: {"results": [], "scores": [], "frame_counts": []}
        )

        super().__init__()

        self._all_sequences = self._load_action_log(action_log_path)
        effective_end = (
            batch_end if batch_end is not None else len(self._all_sequences) - 1
        )
        effective_end = min(effective_end, len(self._all_sequences) - 1)
        self._batch_end = effective_end
        self._total_episodes = (effective_end - batch_start + 1) * replay_count

        subdir_name = f"ep{batch_start}-{effective_end}_r{replay_count}_p{primary_threshold:g}_s{secondary_threshold:g}"
        self._output_dir = str(Path(self._base_output_dir) / subdir_name)

        self._load_current_sequence()

        print(
            f"[ReplayCollector] sequences={batch_start}~{effective_end}, "
            f"replay_count={replay_count}, total_episodes={self._total_episodes}, "
            f"threshold=({primary_threshold}, {secondary_threshold})"
        )

    def _load_action_log(self, path: str) -> List[List[str]]:
        if not path or not os.path.exists(path):
            print(f"[ReplayCollector] Warning: action_log not found: {path}")
            return []
        sequences = []
        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or not row[0].strip():
                    continue
                raw = row[0].strip()
                if len(raw) < 2:
                    continue
                actions = [raw[i : i + 2] for i in range(0, len(raw), 2)]
                sequences.append(actions)
        print(f"[ReplayCollector] Loaded {len(sequences)} action sequences from {path}")
        return sequences

    def _load_current_sequence(self):
        if self._current_seq_idx <= self._batch_end and self._current_seq_idx < len(
            self._all_sequences
        ):
            self._replay_actions = list(self._all_sequences[self._current_seq_idx])
            self._replay_idx = 0
        else:
            self._replay_actions = []
            self._replay_idx = 0

    def get_state_cluster(self, norm_state):
        if self.primary_bktree.root is None:
            self.primary_bktree.root = ClusterNode(norm_state, 1)
            self.secondary_bktree[1].root = ClusterNode(norm_state, 1)
            return (1, 1)
        else:
            new_cluster_id = self.classify_new_state(
                norm_state, self.primary_bktree, threshold=self._primary_threshold
            )
            if self.secondary_bktree[new_cluster_id].root is None:
                self.secondary_bktree[new_cluster_id].root = ClusterNode(norm_state, 1)
                return (new_cluster_id, 1)
            else:
                new_sub_cluster_id = self.classify_new_state(
                    norm_state,
                    self.secondary_bktree[new_cluster_id],
                    threshold=self._secondary_threshold,
                )
                return (new_cluster_id, new_sub_cluster_id)

    def step(self, obs, env):
        super(SmartAgent, self).step(obs, env)

        if obs.last():
            return sc2_actions.RAW_FUNCTIONS.no_op()

        if obs.first():
            if not self._initial_spawned:
                my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
                enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
                self._initial_units_my = [(u.x, u.y) for u in my_units]
                self._initial_units_enemy = [(u.x, u.y) for u in enemy_units]
                self._initial_spawned = True

        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])

        if self._done:
            return sc2_actions.RAW_FUNCTIONS.no_op()

        if not self._termination_signaled:
            if not enemy_units and my_units:
                self.end_game_state = "Win"
                self.end_game_flag = True
                self._termination_signaled = True
                env.f_result = "win"
            elif not my_units and enemy_units:
                self.end_game_state = "Loss"
                self.end_game_flag = True
                self._termination_signaled = True
                env.f_result = "loss"

        if (my_units or enemy_units) and not self._done:
            norm_state = self.get_norm_state(obs)
            state_cluster = self.get_state_cluster(norm_state)

            action_code = None
            if self._replay_idx < len(self._replay_actions):
                action_code = self._replay_actions[self._replay_idx]
                self._replay_idx += 1

            action_name = self._resolve_action(action_code)

            self._pending_cluster = None
            if action_name and len(action_code or "") == 2 and action_code[0].isdigit():
                cluster_idx = int(action_code[0])
                if 0 <= cluster_idx < len(self.clusters):
                    self._pending_cluster = self.clusters[cluster_idx]

            result = None
            if self._pending_cluster and hasattr(self, self._pending_cluster):
                self.cluster_result = getattr(self, self._pending_cluster)(obs)
            if action_name and hasattr(self, action_name):
                result = getattr(self, action_name)(obs)

            hp_my = int(sum(u.health for u in my_units)) if my_units else 0
            hp_enemy = int(sum(u.health for u in enemy_units)) if enemy_units else 0

            self._current_frames.append(
                {
                    "norm_state": norm_state,
                    "state_cluster": (
                        int(state_cluster[0]),
                        int(state_cluster[1]),
                    ),
                    "action_code": action_code,
                    "hp_my": hp_my,
                    "hp_enemy": hp_enemy,
                    "game_loop": int(obs.observation.game_loop[0]),
                }
            )

            if result is not None:
                return result

        return sc2_actions.RAW_FUNCTIONS.no_op()

    def _resolve_action(self, raw_action: Optional[str]) -> Optional[str]:
        if raw_action is None or len(raw_action) < 2:
            return None
        letter = raw_action[1].lower()
        action_idx = ord(letter) - ord("a")
        if 0 <= action_idx < len(self.actions):
            return self.actions[action_idx]
        return None

    def new_game(self):
        if self._done:
            return

        if self._current_frames:
            result = self.end_game_state or "Dogfall"
            last_frame = self._current_frames[-1] if self._current_frames else {}
            score = last_frame.get("hp_my", 0) - last_frame.get("hp_enemy", 0)
            episode = {
                "source_idx": self._current_seq_idx,
                "replay_idx": self._current_run_idx,
                "result": result,
                "score": float(score),
                "num_frames": len(self._current_frames),
                "frames": list(self._current_frames),
            }
            self._collected_episodes.append(episode)
            self._current_frames = []
            self._completed_episodes += 1

            stats = self._seq_stats[self._current_seq_idx]
            stats["results"].append(result)
            stats["scores"].append(score)
            stats["frame_counts"].append(len(episode["frames"]))

        self.end_game_frames = _ENV_CONFIG["_MAX_STEP"] * _ENV_CONFIG["_STEP_MUL"]
        self.end_game_state = "Dogfall"
        self.end_game_flag = False
        self._termination_signaled = False
        self.action_queue.clear()

        self._current_run_idx += 1
        if self._current_run_idx < self._replay_count:
            self._replay_idx = 0
        else:
            self._current_run_idx = 0
            self._current_seq_idx += 1
            if self._current_seq_idx <= self._batch_end:
                self._load_current_sequence()
            else:
                self._done = True
                self._save_all()
                return

        self._report_progress()

    def _end_episode(self, obs):
        pass

    def _report_progress(self):
        done = self._completed_episodes
        total = self._total_episodes
        pct = done / total * 100 if total else 0

        if self.bridge:
            try:
                self.bridge.update_status(
                    frame=done,
                    batch_replay_progress=f"{done}/{total} ({pct:.1f}%)",
                    batch_replay_current_seq=self._current_seq_idx,
                )
            except Exception:
                pass
            if done % 100 == 0 or done == total:
                summary = self._compute_summary_stats()
                try:
                    self.bridge.put_event(
                        {
                            "level": "info",
                            "source": "game",
                            "message": f"进度: {done}/{total} ({pct:.1f}%)",
                            "batch_stats": summary,
                        }
                    )
                except Exception:
                    pass
        else:
            print(f"[ReplayCollector] 进度: {done}/{total} ({pct:.1f}%)")

    def _compute_summary_stats(self) -> Dict[str, Any]:
        import numpy as np

        all_results: List[str] = []
        all_scores: List[float] = []
        for stats in self._seq_stats.values():
            all_results.extend(stats["results"])
            all_scores.extend(stats["scores"])
        if not all_results:
            return {}
        win_count = sum(1 for r in all_results if r == "Win")
        return {
            "completed_sequences": len(self._seq_stats),
            "total_episodes": len(all_results),
            "win_rate": round(win_count / len(all_results), 4),
            "mean_score": round(float(np.mean(all_scores)), 2),
            "std_score": round(float(np.std(all_scores)), 2),
        }

    def _save_all(self):
        output_dir = Path(self._output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pkl_path = output_dir / f"collected_data_{ts}.pkl"
        data = {
            "source_action_log": self._action_log_path,
            "batch_start": self._batch_start,
            "batch_end": self._batch_end,
            "replay_count": self._replay_count,
            "primary_threshold": self._primary_threshold,
            "secondary_threshold": self._secondary_threshold,
            "total_episodes": self._completed_episodes,
            "episodes": self._collected_episodes,
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[ReplayCollector] Saved collected data to {pkl_path}")

        self._save_bktree_to_dir(output_dir)

        summary = self._compute_summary_stats()
        summary["primary_threshold"] = self._primary_threshold
        summary["secondary_threshold"] = self._secondary_threshold
        stats_path = output_dir / f"batch_stats_{ts}.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[ReplayCollector] Saved batch stats to {stats_path}")

        if self.bridge:
            try:
                self.bridge.put_event(
                    {
                        "level": "success",
                        "source": "game",
                        "message": f"数据增强完成: {self._completed_episodes} episodes, BKTree 已保存",
                        "batch_stats": summary,
                    }
                )
                self.bridge.update_status(batch_replay_done=True)
                self.bridge.send_control("pause")
            except Exception:
                pass
        else:
            print(
                f"[ReplayCollector] 全部完成: {self._completed_episodes} episodes, "
                f"Win rate: {summary.get('win_rate', 0):.1%}"
            )

    def _save_bktree_to_dir(self, output_dir: Path):
        def serialize_node(node):
            if node is None:
                return None
            node_info = {
                "state": node.state,
                "cluster_id": node.cluster_id,
                "children": {},
            }
            for dist, child in node.children.items():
                node_info["children"][str(dist)] = serialize_node(child)
            return node_info

        if self.primary_bktree.root is not None:
            primary_path = output_dir / "primary_bktree.json"
            with open(primary_path, "w") as f:
                json.dump(serialize_node(self.primary_bktree.root), f)
            print(f"[ReplayCollector] Saved primary BKTree to {primary_path}")

        for cluster_id, bktree in self.secondary_bktree.items():
            if bktree.root is not None:
                sec_path = output_dir / f"secondary_bktree_{cluster_id}.json"
                with open(sec_path, "w") as f:
                    json.dump(serialize_node(bktree.root), f)

        print(f"[ReplayCollector] Saved {len(self.secondary_bktree)} secondary BKTrees")
