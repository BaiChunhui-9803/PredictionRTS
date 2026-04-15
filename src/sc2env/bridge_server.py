"""
Bridge Server — FastAPI 桥接服务

在独立进程中运行，通过 GameBridge (multiprocessing.Queue) 与 SC2 游戏进程通信，
对外提供 REST API 供 Streamlit Web 前端调用。

端点:
    GET  /game/status           → 游戏运行状态
    GET  /game/observation      → 当前游戏观测
    GET  /game/logs             → 合并日志 (game事件 + API请求)
    POST /game/predict          → KG beam search 推理
    POST /game/rollout          → 多步链式推演
    POST /game/autopilot        → 开关自动决策
    GET  /game/autopilot/status → 查询自动决策状态
    POST /game/action           → 发送动作指令
    POST /game/fallback         → 设置默认回退策略
    POST /game/control          → 控制游戏 (pause/resume/stop/step)
    POST /game/start            → 启动游戏子进程
    GET  /game/process          → 游戏子进程状态
    POST /game/stop-process     → 终止游戏子进程 (保留API)
    GET  /game/events           → SSE 游戏事件流
    GET  /game/actions          → 可用动作列表
    GET  /game/beam_params      → 获取/更新 beam search 参数
"""

from __future__ import annotations

import asyncio
import collections
import multiprocessing
import os
import sys
import json
import logging
import pickle
import time
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

import threading

from src.sc2env.bridge import GameBridge
from src import ROOT_DIR


def _game_worker(bridge, map_key, run_name, fallback_action):
    from absl import flags as absl_flags

    if not absl_flags.FLAGS.is_parsed():
        absl_flags.FLAGS(["run_live_game.py"])
    from src.sc2env.run_game import run_game

    run_game(
        map_key=map_key,
        run_name=run_name,
        bridge=bridge,
        agent_type="kg_guided",
        fallback_action=fallback_action,
    )


logger = logging.getLogger(__name__)

KG_DIR = ROOT_DIR / "cache" / "knowledge_graph"

_instance: Optional["BridgeServer"] = None

_MAX_LOG_BUFFER = 500


class PredictRequest(BaseModel):
    state: Optional[int] = None
    beam_width: int = 3
    max_steps: int = 3
    min_visits: int = 1
    min_cum_prob: float = 0.01
    score_mode: str = "quality"
    max_state_revisits: int = 2
    discount_factor: float = 0.9


class RolloutRequest(BaseModel):
    start_state: Optional[int] = None
    score_mode: str = "quality"
    action_strategy: str = "best_beam"
    next_state_mode: str = "sample"
    beam_width: int = 3
    lookahead_steps: int = 5
    max_rollout_steps: int = 30
    min_visits: int = 1
    min_cum_prob: float = 0.01
    max_state_revisits: int = 2
    discount_factor: float = 0.9
    rollout_mode: str = "single_step"
    enable_backup: bool = False
    epsilon: float = 0.1
    rng_seed: Optional[int] = None


class AutopilotRequest(BaseModel):
    enabled: bool
    mode: str = "single_step"
    score_mode: str = "quality"
    action_strategy: str = "best_beam"
    next_state_mode: str = "sample"
    beam_width: int = 3
    lookahead_steps: int = 5
    max_rollout_steps: int = 30
    min_visits: int = 1
    min_cum_prob: float = 0.01
    max_state_revisits: int = 2
    discount_factor: float = 0.9
    enable_backup: bool = False
    epsilon: float = 0.1


class ActionRequest(BaseModel):
    action: str


class ControlRequest(BaseModel):
    command: str


class FallbackRequest(BaseModel):
    action: str


class LogEntry:
    __slots__ = ("ts", "level", "source", "message", "seq")

    def __init__(self, level: str, source: str, message: str):
        self.ts = datetime.now().strftime("%H:%M:%S")
        self.level = level
        self.source = source
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "level": self.level,
            "source": self.source,
            "message": self.message,
            "seq": self.seq,
        }


@dataclass
class EpisodeRecord:
    id: int
    result: str = ""
    score: float = 0.0
    markov_flow: List[Tuple[int, str]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    plans: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""
    mode: str = ""
    steps: int = 0
    match_mode: str = ""


def _parse_state_id(obs: Dict[str, Any]) -> Optional[int]:
    state_cluster = obs.get("state_cluster_str", "0")
    try:
        if isinstance(state_cluster, str):
            state_cluster = eval(state_cluster)
        if isinstance(state_cluster, (list, tuple)) and len(state_cluster) >= 2:
            key = (int(state_cluster[0]), int(state_cluster[1]))
            if _instance and _instance._state_id_map:
                nid = _instance._state_id_map.get(key)
                if nid is not None:
                    return nid
                norm_state = obs.get("norm_state")
                if norm_state and _instance._nid_norm_states:
                    from src.structure.custom_distance_sc2 import DistributionDistance

                    best_nid = None
                    best_dist = float("inf")
                    best_hp = float("inf")
                    for nid, stored in _instance._nid_norm_states.items():
                        try:
                            dd = DistributionDistance(norm_state, stored)
                            d, h = dd()
                            if d < best_dist or (d == best_dist and h < best_hp):
                                best_dist = d
                                best_hp = h
                                best_nid = nid
                        except Exception:
                            pass
                    if best_nid is not None:
                        _instance.add_log(
                            "warn",
                            "state_mapping",
                            f"({key[0]},{key[1]})→S{best_nid} dist={best_dist:.3f} hp={best_hp:.3f}",
                        )
                        return best_nid
                _instance.add_log(
                    "error",
                    "state_mapping",
                    f"({key[0]},{key[1]}) 无映射, 跳过该帧",
                )
                return None
            return int(state_cluster[0])
        elif isinstance(state_cluster, (list, tuple)) and len(state_cluster) == 1:
            return int(state_cluster[0])
        elif isinstance(state_cluster, (int, float)):
            return int(state_cluster)
    except Exception:
        pass
    return None


def _enrich_event(event: Dict[str, Any]) -> Dict[str, Any]:
    if event.get("level") == "debug" and "state_cluster" in event:
        p, s = event["state_cluster"]
        nid = None
        if _instance and _instance._state_id_map:
            nid = _instance._state_id_map.get((int(p), int(s)))
        msg = event.get("message", "")
        if nid is not None:
            event["message"] = f"nid={nid} {msg}"
        else:
            event["message"] = f"nid=? {msg}"
        del event["state_cluster"]
    return event


def _states_match(
    actual: int, expected: int, dist_matrix=None, threshold: float = 0.2
) -> bool:
    if actual == expected:
        return True
    if dist_matrix is not None:
        try:
            d = float(dist_matrix[actual, expected])
            if not np.isnan(d) and d < threshold:
                return True
        except (IndexError, TypeError, KeyError):
            pass
    return False


class BridgeServer:
    def __init__(self, bridge: GameBridge):
        self.bridge = bridge
        self.kg = None
        self.transitions = None
        self.kg_loaded = False
        self.kg_file: Optional[str] = None
        self._state_id_map: Dict[Tuple[int, int], int] = {}
        self._local_status: Dict[str, Any] = {
            "running": False,
            "paused": False,
            "frame": 0,
            "episode": 0,
            "map_key": "",
            "mode": "idle",
        }
        self._beam_search_params = {
            "beam_width": 3,
            "max_steps": 3,
            "min_visits": 1,
            "min_cum_prob": 0.01,
            "score_mode": "quality",
            "max_state_revisits": 2,
            "discount_factor": 0.9,
        }
        self._log_buffer: collections.deque = collections.deque(maxlen=_MAX_LOG_BUFFER)
        self._log_seq: int = 0

        self._autopilot_enabled = False
        self._autopilot_mode = "single_step"
        self._autopilot_params: Dict[str, Any] = {}
        self._autopilot_task: Optional[asyncio.Task] = None
        self._autopilot_lock = threading.Lock()
        self._action_plan: List[str] = []
        self._planned_states: List[int] = []
        self._plan_idx: int = 0
        self._autopilot_stats = {
            "total_decisions": 0,
            "total_replans": 0,
            "total_divergences": 0,
            "last_action": None,
            "last_state": None,
            "plan_progress": "0/0",
        }
        self._dist_matrix: Optional[np.ndarray] = None
        self._last_obs_state: Optional[int] = None
        self._nid_norm_states: Dict[int, dict] = {}

        self._episodes: List[EpisodeRecord] = []
        self._episode_counter: int = 0
        self._episode_buffer: List[Dict[str, Any]] = []
        self._current_plans: List[Dict[str, Any]] = []
        self._prev_end_game_flag: bool = False
        self._last_score: float = 0.0

        self._history_store: Dict[int, List[Dict[str, Any]]] = {}
        self._history_meta: Dict[int, Dict[str, Any]] = {}
        self._history_max_episodes: int = 100
        self._plans_by_episode: Dict[int, List[Dict[str, Any]]] = {}
        self._current_episode: int = 0

    def add_log(self, level: str, source: str, message: str) -> None:
        self._log_seq += 1
        entry = LogEntry(level, source, message)
        entry.seq = self._log_seq
        self._log_buffer.append(entry)

    def drain_logs(self, after_seq: int = 0) -> List[Dict[str, Any]]:
        for event in self.bridge.get_events():
            event = _enrich_event(event)
            level = event.get("level", "info")
            source = event.get("source", "game")
            message = event.get("message", str(event))
            if isinstance(message, dict):
                message = json.dumps(message, default=str)
            self.add_log(level, source, message)
            if "得分:" in str(message):
                try:
                    score_str = str(message).split("得分:")[1].strip().split()[0]
                    self._last_score = float(score_str)
                except (ValueError, IndexError):
                    pass

        result = []
        for entry in self._log_buffer:
            if hasattr(entry, "seq") and entry.seq > after_seq:
                result.append(entry.to_dict())
        return result

    def drain_all_logs(self) -> List[Dict[str, Any]]:
        for event in self.bridge.get_events():
            event = _enrich_event(event)
            level = event.get("level", "info")
            source = event.get("source", "game")
            message = event.get("message", str(event))
            if isinstance(message, dict):
                message = json.dumps(message, default=str)
            self.add_log(level, source, message)

        return [entry.to_dict() for entry in self._log_buffer]

    def clear_logs(self) -> None:
        self._log_buffer.clear()

    def _to_native(self, val):
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, dict):
            return {k: self._to_native(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [self._to_native(v) for v in val]
        return val

    def _refresh_status(self) -> Dict[str, Any]:
        for upd in self.bridge.drain_status_updates():
            self._local_status.update(upd)
        result = {k: self._to_native(v) for k, v in self._local_status.items()}
        result["kg_loaded"] = self.kg_loaded
        result["kg_file"] = self.kg_file
        return result

    def load_kg(self, kg_file: str, data_dir: Optional[str] = None) -> bool:
        from src.decision.knowledge_graph import DecisionKnowledgeGraph

        path = KG_DIR / kg_file
        if not path.exists():
            logger.error(f"KG file not found: {path}")
            return False

        try:
            self.kg = DecisionKnowledgeGraph.load(str(path))
            self.kg_file = kg_file
            logger.info(f"Loaded KG from {path}")
        except Exception as e:
            logger.error(f"Failed to load KG: {e}")
            return False

        trans_file = kg_file.replace(".pkl", "_transitions.pkl")
        if not trans_file.endswith(".pkl"):
            trans_file = kg_file + "_transitions.pkl"
        trans_path = KG_DIR / trans_file
        if trans_path.exists():
            try:
                with open(trans_path, "rb") as f:
                    self.transitions = pickle.load(f)
                logger.info(f"Loaded transitions from {trans_path}")
            except Exception as e:
                logger.warning(f"Failed to load transitions: {e}")
                self.transitions = {}
        else:
            logger.warning(f"Transitions file not found: {trans_path}")
            self.transitions = {}

        self._dist_matrix = None
        if data_dir:
            dp = Path(data_dir)
            parts = dp.parts
            if len(parts) >= 2:
                map_id, data_id = parts[-2], parts[-1]
                npy_dir = ROOT_DIR / "cache" / "npy"
                dm_path = npy_dir / f"state_distance_matrix_{map_id}_{data_id}.npy"
                if dm_path.exists():
                    try:
                        self._dist_matrix = np.load(str(dm_path))
                        logger.info(
                            f"Loaded distance matrix from {dm_path} ({self._dist_matrix.shape})"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load distance matrix: {e}")
                else:
                    logger.warning(f"Distance matrix not found: {dm_path}")

        self._state_id_map = {}
        if data_dir:
            sn_path = Path(data_dir) / "graph" / "state_node.txt"
            if not sn_path.exists():
                sn_path = ROOT_DIR / data_dir / "graph" / "state_node.txt"
            if sn_path.exists():
                try:
                    for line in sn_path.read_text(encoding="utf-8").splitlines():
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            try:
                                key = eval(parts[0])
                                if isinstance(key, tuple) and len(key) == 2:
                                    self._state_id_map[(int(key[0]), int(key[1]))] = (
                                        int(parts[1])
                                    )
                            except Exception:
                                pass
                    logger.info(
                        f"Loaded state_id_map: {len(self._state_id_map)} entries from {sn_path}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load state_node.txt: {e}")
            else:
                logger.warning(f"state_node.txt not found: {sn_path}")

        self._nid_norm_states = {}
        if data_dir and self._state_id_map:
            self._load_known_states(data_dir)

        self.kg_loaded = True
        return True

    def _load_known_states(self, data_dir: str) -> None:
        import json as _json

        bktree_dir = Path(data_dir) / "bktree"
        if not bktree_dir.exists():
            logger.warning(f"bktree dir not found: {bktree_dir}")
            return

        def _collect_nodes(node):
            nodes = []
            if node is not None:
                nodes.append(node)
                for child in node.get("children", {}).values():
                    nodes.extend(_collect_nodes(child))
            return nodes

        secondary_states = {}
        sec_files = list(bktree_dir.glob("secondary_bktree_*.json"))
        for sf in sec_files:
            cid_str = sf.stem.replace("secondary_bktree_", "")
            try:
                with open(str(sf), "r") as f:
                    root = _json.load(f)
                p_id = int(cid_str)
                for node in _collect_nodes(root):
                    s_id = node.get("cluster_id")
                    st = node.get("state")
                    if s_id is not None and st is not None:
                        secondary_states[(p_id, s_id)] = st
            except Exception:
                pass

        primary_file = bktree_dir / "primary_bktree.json"
        if primary_file.exists():
            try:
                with open(str(primary_file), "r") as f:
                    root = _json.load(f)
                for node in _collect_nodes(root):
                    p_id = node.get("cluster_id")
                    st = node.get("state")
                    if p_id is not None and st is not None:
                        for (pk, sk), nid in self._state_id_map.items():
                            if pk == p_id and (pk, sk) in secondary_states:
                                self._nid_norm_states[nid] = secondary_states[(pk, sk)]
            except Exception as e:
                logger.warning(f"Failed to load known states: {e}")

        logger.info(
            f"Loaded {len(self._nid_norm_states)} nid norm_states for distance matching"
        )

    def set_beam_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if k in self._beam_search_params:
                self._beam_search_params[k] = v

    def predict(self, state_id: int, **override_params) -> Dict[str, Any]:
        if not self.kg_loaded or self.kg is None:
            return {"error": "KG not loaded"}

        params = dict(self._beam_search_params)
        params.update(override_params)

        from src.decision.kg_beam_search import find_optimal_action

        action, info = find_optimal_action(
            self.kg,
            self.transitions,
            state_id,
            beam_width=params["beam_width"],
            max_steps=params["max_steps"],
            min_visits=params["min_visits"],
            min_cum_prob=params["min_cum_prob"],
            score_mode=params["score_mode"],
            max_state_revisits=params["max_state_revisits"],
            discount_factor=params["discount_factor"],
        )

        return {
            "action": action,
            "expected_cumulative_reward": info.get("expected_cumulative_reward", 0),
            "expected_win_rate": info.get("expected_win_rate", 0),
            "best_beam_cum_prob": info.get("best_beam_cum_prob", 0),
            "best_beam_length": info.get("best_beam_length", 0),
            "reason": info.get("reason"),
        }

    def _get_plan_from_beam(
        self, state_id: int, lookahead_steps: int
    ) -> Tuple[Optional[str], List[str], List[int], List[Dict], List[Dict]]:
        from src.decision.kg_beam_search import find_optimal_action, get_beam_paths

        action, info = find_optimal_action(
            self.kg,
            self.transitions,
            state_id,
            beam_width=self._autopilot_params.get("beam_width", 3),
            max_steps=lookahead_steps,
            min_visits=self._autopilot_params.get("min_visits", 1),
            min_cum_prob=self._autopilot_params.get("min_cum_prob", 0.01),
            score_mode=self._autopilot_params.get("score_mode", "quality"),
            max_state_revisits=self._autopilot_params.get("max_state_revisits", 2),
            discount_factor=self._autopilot_params.get("discount_factor", 0.9),
        )

        all_results = info.get("all_results", [])
        beam_dicts = []
        for r in all_results:
            beam_dicts.append(
                {
                    "step": getattr(r, "step", 0),
                    "state": getattr(r, "state", 0),
                    "action": getattr(r, "action", ""),
                    "beam_id": getattr(r, "beam_id", 0),
                    "parent_idx": getattr(r, "parent_idx", None),
                    "cumulative_probability": getattr(r, "cumulative_probability", 0),
                    "quality_score": getattr(r, "quality_score", 0),
                    "win_rate": getattr(r, "win_rate", 0),
                    "avg_step_reward": getattr(r, "avg_step_reward", 0),
                    "avg_future_reward": getattr(r, "avg_future_reward", 0),
                }
            )

        paths = get_beam_paths(all_results) if all_results else []
        paths.sort(key=lambda p: p[-1].cumulative_probability, reverse=True)
        chosen_idx = 0
        beam_paths = []
        for rank, path in enumerate(paths):
            is_chosen = rank == 0
            if is_chosen and path and len(path) > 1 and path[0].action:
                chosen_idx = rank
            path_steps = []
            for node in path:
                path_steps.append(
                    {
                        "state": node.state,
                        "action": node.action or "",
                        "cum_prob": node.cumulative_probability,
                        "win_rate": node.win_rate,
                    }
                )
            beam_paths.append(
                {
                    "rank": rank + 1,
                    "chosen": is_chosen,
                    "steps": path_steps,
                    "cum_prob": path[-1].cumulative_probability if path else 0,
                }
            )

        actions = [action] if action else []
        states = [state_id]

        if all_results and action:
            best_path_nodes = paths[0] if paths else None
            if best_path_nodes and len(best_path_nodes) > 1:
                actions = []
                states = []
                for node in best_path_nodes:
                    states.append(node.state)
                    if node.action:
                        actions.append(node.action)
                if states and states[0] != state_id:
                    states.insert(0, state_id)
                chosen_idx = 0

        first_action = actions[0] if actions else None
        return first_action, actions, states, beam_dicts, beam_paths

    def _pick_best_path(self, all_results):
        from src.decision.kg_beam_search import get_beam_paths

        try:
            paths = get_beam_paths(all_results)
            if not paths:
                return None
            paths.sort(key=lambda p: p[-1].cumulative_probability, reverse=True)
            return paths[0]
        except Exception:
            return None

    def _autopilot_decide(
        self, state_id: int
    ) -> Tuple[Optional[str], str, Optional[Dict]]:
        if not self.kg_loaded or self.kg is None:
            self.add_log("warn", "autopilot", "KG 未加载，跳过决策")
            return None, "no_action", None

        mode = self._autopilot_mode
        params = self._autopilot_params
        lookahead = params.get("lookahead_steps", 5)

        if mode == "single_step":
            try:
                first_action, actions, states, beam_dicts, beam_paths = (
                    self._get_plan_from_beam(
                        state_id, self._beam_search_params.get("max_steps", 3)
                    )
                )
            except Exception as e:
                self.add_log("error", "autopilot", f"推理异常: {e}")
                return None, "fallback", None
            action = first_action
            self._autopilot_stats["total_decisions"] += 1
            self._action_plan = []
            self._planned_states = []
            self._plan_idx = 0
            self._autopilot_stats["plan_progress"] = "0/0"
            plan_snap = (
                {
                    "state_id": state_id,
                    "action_plan": [action] if action else [],
                    "planned_states": [state_id],
                    "beam_results": beam_dicts,
                    "beam_paths": beam_paths,
                    "mode": "single_step",
                }
                if action
                else None
            )
            return action, "kg_plan" if action else "no_action", plan_snap

        else:
            is_diverge = False
            if self._plan_idx < len(self._action_plan):
                expected = (
                    self._planned_states[self._plan_idx]
                    if self._plan_idx < len(self._planned_states)
                    else None
                )
                if expected is not None and not _states_match(
                    state_id, expected, self._dist_matrix, 0.2
                ):
                    self._autopilot_stats["total_divergences"] += 1
                    self.add_log(
                        "warn",
                        "autopilot",
                        f"状态偏离: 预期 S{expected}, 实际 S{state_id}, 重新规划",
                    )
                    self._action_plan = []
                    self._planned_states = []
                    self._plan_idx = 0
                    is_diverge = True

            plan_snap = None
            if self._plan_idx >= len(self._action_plan):
                self._autopilot_stats["total_replans"] += 1
                try:
                    first_action, actions, states, beam_dicts, beam_paths = (
                        self._get_plan_from_beam(state_id, lookahead)
                    )
                    if not actions:
                        return None, "no_action", None
                    self._action_plan = actions
                    self._planned_states = states
                    self._plan_idx = 0
                    plan_snap = {
                        "state_id": state_id,
                        "action_plan": actions,
                        "planned_states": states,
                        "beam_results": beam_dicts,
                        "beam_paths": beam_paths,
                        "mode": "multi_step",
                        "trigger": "diverge" if is_diverge else "exhausted",
                    }
                    self.add_log(
                        "info",
                        "autopilot",
                        f"新规划: {len(actions)} 步, 起点 S{state_id}"
                        + (" (偏离触发)" if is_diverge else ""),
                    )
                except Exception as e:
                    self.add_log("error", "autopilot", f"规划异常: {e}")
                    return None, "fallback", None

            action = self._action_plan[self._plan_idx]
            event_type = "kg_plan" if self._plan_idx == 0 else "kg_follow"
            self._plan_idx += 1
            self._autopilot_stats["total_decisions"] += 1
            self._autopilot_stats["plan_progress"] = (
                f"{self._plan_idx}/{len(self._action_plan)}"
            )
            return action, event_type, plan_snap

    def get_autopilot_status(self) -> Dict[str, Any]:
        return {
            "enabled": self._autopilot_enabled,
            "mode": self._autopilot_mode,
            "plan_progress": self._autopilot_stats["plan_progress"],
            "current_action": (
                self._action_plan[self._plan_idx]
                if self._plan_idx < len(self._action_plan)
                else None
            ),
            "stats": dict(self._autopilot_stats),
        }

    def start_autopilot(self, params: Dict[str, Any]) -> None:
        self._autopilot_params = params
        self._autopilot_mode = params.get("mode", "single_step")
        self._autopilot_enabled = True
        self._action_plan = []
        self._planned_states = []
        self._plan_idx = 0
        self._autopilot_stats = {
            "total_decisions": 0,
            "total_replans": 0,
            "total_divergences": 0,
            "last_action": None,
            "last_state": None,
            "plan_progress": "0/0",
        }
        self._last_obs_state = None
        self._episode_buffer = []
        self._current_plans = []
        self._prev_end_game_flag = False
        self._last_score = 0.0
        self.add_log(
            "success", "autopilot", f"自动决策已启动 (mode={self._autopilot_mode})"
        )

    def stop_autopilot(self) -> None:
        self._autopilot_enabled = False
        self._action_plan = []
        self._planned_states = []
        self._plan_idx = 0
        if self._episode_buffer:
            self._commit_buffer(result="Interrupted")
        self.add_log("info", "autopilot", "自动决策已停止")

    def run_rollout_sync(
        self, start_state: int, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        from src.decision.chain_rollout import chain_rollout

        result = chain_rollout(
            self.kg,
            self.transitions,
            start_state,
            score_mode=params.get("score_mode", "quality"),
            action_strategy=params.get("action_strategy", "best_beam"),
            next_state_mode=params.get("next_state_mode", "sample"),
            beam_width=params.get("beam_width", 3),
            lookahead_steps=params.get("lookahead_steps", 5),
            max_rollout_steps=params.get("max_rollout_steps", 30),
            min_visits=params.get("min_visits", 1),
            min_cum_prob=params.get("min_cum_prob", 0.01),
            max_state_revisits=params.get("max_state_revisits", 2),
            discount_factor=params.get("discount_factor", 0.9),
            rollout_mode=params.get("rollout_mode", "single_step"),
            enable_backup=params.get("enable_backup", False),
            epsilon=params.get("epsilon", 0.1),
            rng_seed=params.get("rng_seed"),
            dist_matrix=self._dist_matrix,
        )

        nodes = {}
        for nid_str, node in result.nodes.items():
            nodes[nid_str] = {
                "state": node.state,
                "action": node.action,
                "quality_score": node.quality_score,
                "win_rate": node.win_rate,
                "avg_future_reward": node.avg_future_reward,
                "avg_step_reward": node.avg_step_reward,
                "visits": node.visits,
                "transition_prob": node.transition_prob,
                "cumulative_probability": node.cumulative_probability,
                "is_terminal": node.is_terminal,
                "is_on_chosen_path": node.is_on_chosen_path,
            }

        steps = []
        path_ids = result.chosen_path_ids
        for i in range(len(path_ids) - 1):
            prev = nodes.get(str(path_ids[i]), {})
            curr = nodes.get(str(path_ids[i + 1]), {})
            steps.append(
                {
                    "step": i + 1,
                    "state": prev.get("state"),
                    "action": curr.get("action"),
                    "next_state": curr.get("state"),
                    "win_rate": curr.get("win_rate"),
                    "quality_score": curr.get("quality_score"),
                }
            )

        first_action = steps[0]["action"] if steps else None

        return {
            "summary": {
                "total_steps": len(steps),
                "termination_reason": result.termination_reason,
                "first_action": first_action,
                "rollout_mode": result.rollout_mode,
                "total_re_searches": result.total_re_searches,
                "total_backup_switches": result.total_backup_switches,
            },
            "steps": steps,
            "full_nodes": nodes,
        }

    def _commit_buffer(self, result: str = "", score: float = 0.0) -> None:
        if not self._episode_buffer:
            return
        self._episode_counter += 1
        ep = EpisodeRecord(
            id=self._episode_counter,
            result=result,
            score=score,
            markov_flow=[(e["state_id"], e["action"]) for e in self._episode_buffer],
            events=[
                {
                    "state_id": e["state_id"],
                    "action": e["action"],
                    "event_type": e["event_type"],
                    "mode": e["mode"],
                    "match_mode": e["match_mode"],
                    "plan_id": e["plan_id"],
                }
                for e in self._episode_buffer
            ],
            plans=[],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            mode=self._autopilot_mode,
            steps=len(self._episode_buffer),
            match_mode=self._autopilot_params.get("match_mode", ""),
        )
        self._episodes.append(ep)
        if len(self._episodes) > 50:
            self._episodes = self._episodes[-50:]
        self._episode_buffer = []
        self._prev_end_game_flag = False
        self.add_log("info", "episode", f"回合结束: {result}, 得分: {ep.score}")

    def _detect_episode_end(
        self, events: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        for evt in events:
            evt_type = evt.get("type", "")
            if evt_type == "episode_end":
                return evt
            message = evt.get("message", "")
            if isinstance(message, dict):
                message = json.dumps(message, default=str)
            if "得分:" in str(message):
                try:
                    score_str = str(message).split("得分:")[1].strip().split()[0]
                    score = float(score_str)
                    return {"type": "score_detected", "score": score}
                except (ValueError, IndexError):
                    pass
        return None

    def _drain_history(self) -> None:
        for ep_data in self.bridge.get_histories():
            ep_id = ep_data["episode_id"]
            self._history_store[ep_id] = ep_data["frames"]
            self._history_meta[ep_id] = {
                "result": ep_data.get("result", "Unknown"),
                "score": ep_data.get("score", 0.0),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": self._autopilot_mode,
                "frame_count": len(ep_data["frames"]),
                "plans": list(self._current_plans),
            }
            self._current_plans = []
        while len(self._history_store) > self._history_max_episodes:
            oldest = min(self._history_store.keys())
            del self._history_store[oldest]
            if oldest in self._history_meta:
                del self._history_meta[oldest]

    def history_stats(self) -> Dict[str, int]:
        total_frames = sum(len(f) for f in self._history_store.values())
        return {
            "history_episodes": len(self._history_store),
            "history_frames": total_frames,
            "history_capacity": self._history_max_episodes,
        }

    def ack_history(self, episode_ids: List[int]) -> None:
        for eid in episode_ids:
            self._history_store.pop(eid, None)
            self._history_meta.pop(eid, None)

    def event_generator(self):
        try:
            while True:
                events = self.bridge.get_events()
                for event in events:
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                if not events:
                    yield f": heartbeat\n\n"
                time.sleep(0.2)
        except GeneratorExit:
            pass


async def _autopilot_loop():
    inst = _instance
    if inst is None:
        return
    try:
        while inst._autopilot_enabled:
            inst._drain_history()
            raw_events = inst.bridge.get_events()
            if raw_events:
                for re in raw_events:
                    re = _enrich_event(re)
                    level = re.get("level", "info")
                    source = re.get("source", "game")
                    message = re.get("message", str(re))
                    if isinstance(message, dict):
                        message = json.dumps(message, default=str)
                    inst.add_log(level, source, message)
                end_info = inst._detect_episode_end(raw_events)
                if end_info is not None:
                    if end_info.get("type") == "score_detected":
                        score = end_info.get("score", 0.0)
                        inst._last_score = score
                        if inst._episodes:
                            inst._episodes[-1].score = score
                    elif inst._episode_buffer:
                        result = end_info.get("result", "Unknown")
                        inst._commit_buffer(result=result)
                    await asyncio.sleep(0.01)
                    continue

            obs = inst.bridge.get_observation(timeout=0.5)
            if obs is None:
                await asyncio.sleep(0.1)
                continue

            status = inst._refresh_status()
            if not status.get("running", False) or status.get("paused", False):
                await asyncio.sleep(0.5)
                continue

            end_flag = obs.get("end_game_flag", False)
            if end_flag and not inst._prev_end_game_flag:
                result = obs.get("end_game_state", "Unknown")
                inst._commit_buffer(result=result)
                inst._last_obs_state = None
                inst._prev_end_game_flag = True
                await asyncio.sleep(0.01)
                continue

            if end_flag:
                await asyncio.sleep(0.05)
                continue

            inst._prev_end_game_flag = False

            state_id = _parse_state_id(obs)
            if state_id is None:
                await asyncio.sleep(0.1)
                continue

            if inst._last_obs_state == state_id:
                await asyncio.sleep(0.05)
                continue

            inst._last_obs_state = state_id
            loop = asyncio.get_event_loop()
            action, event_type, plan_snap = await loop.run_in_executor(
                None, inst._autopilot_decide, state_id
            )

            if action is not None:
                inst.bridge.put_action(action)
                inst._autopilot_stats["last_action"] = action
                inst._autopilot_stats["last_state"] = state_id
                inst.add_log("info", "autopilot", f"S{state_id} → {action}")

                plan_id = None
                if plan_snap is not None:
                    inst._current_plans.append(plan_snap)
                    plan_id = len(inst._current_plans) - 1
                inst._episode_buffer.append(
                    {
                        "state_id": state_id,
                        "action": action,
                        "event_type": event_type,
                        "mode": inst._autopilot_mode,
                        "match_mode": inst._autopilot_params.get("match_mode", ""),
                        "plan_id": plan_id,
                    }
                )

            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        inst.add_log("error", "autopilot", f"异常退出: {str(e)}")
        inst._autopilot_enabled = False
    finally:
        if inst._episode_buffer:
            inst._commit_buffer(result="Interrupted")


def create_app(bridge: GameBridge) -> FastAPI:
    global _instance
    _instance = BridgeServer(bridge)

    app = FastAPI(title="PredictionRTS Bridge Server", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        path = request.url.path
        if path in (
            "/game/status",
            "/game/observation",
            "/game/logs",
            "/game/events",
            "/game/actions",
            "/game/beam_params",
            "/game/autopilot/status",
            "/game/episodes",
            "/game/episodes/clear",
        ):
            return await call_next(request)

        start = time.time()
        response = await call_next(request)
        elapsed_ms = round((time.time() - start) * 1000, 1)
        method = request.method
        status = response.status_code

        if status >= 500:
            level = "error"
        elif status >= 400:
            level = "warn"
        else:
            level = "info"
        _instance.add_log(level, "api", f"{method} {path} {status} ({elapsed_ms}ms)")

        return response

    @app.get("/game/status")
    async def get_status():
        status = _instance._refresh_status()
        status.update(_instance.history_stats())
        return status

    @app.get("/game/observation")
    async def get_observation():
        obs = _instance.bridge.get_observation(timeout=1.0)
        if obs is None:
            return {
                "error": "no_observation",
                "message": "No observation available. Game may not be running.",
            }
        return _instance._to_native(obs)

    @app.get("/game/logs")
    async def get_logs(after_seq: int = Query(0)):
        logs = _instance.drain_logs(after_seq=after_seq)
        return {"logs": logs, "latest_seq": _instance._log_seq}

    @app.post("/game/logs/clear")
    async def clear_logs():
        _instance.clear_logs()
        return {"status": "cleared"}

    @app.post("/game/predict")
    async def predict(req: PredictRequest):
        if not _instance.kg_loaded:
            raise HTTPException(
                status_code=400, detail="KG not loaded. Call POST /game/load_kg first."
            )

        obs = _instance.bridge.get_observation(timeout=0.1)
        state_id = req.state

        if state_id is None and obs is not None:
            state_id = _parse_state_id(obs)

        if state_id is None:
            raise HTTPException(status_code=400, detail="Cannot determine state ID.")

        result = _instance.predict(
            state_id,
            beam_width=req.beam_width,
            max_steps=req.max_steps,
            min_visits=req.min_visits,
            min_cum_prob=req.min_cum_prob,
            score_mode=req.score_mode,
            max_state_revisits=req.max_state_revisits,
            discount_factor=req.discount_factor,
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    @app.post("/game/rollout")
    async def rollout(req: RolloutRequest):
        if not _instance.kg_loaded:
            raise HTTPException(status_code=400, detail="KG not loaded.")

        start_state = req.start_state
        if start_state is None:
            obs = _instance.bridge.get_observation(timeout=0.5)
            if obs is None:
                raise HTTPException(status_code=400, detail="No observation available.")
            start_state = _parse_state_id(obs)
            if start_state is None:
                raise HTTPException(
                    status_code=400, detail="Cannot determine state ID."
                )

        params = req.model_dump(exclude={"start_state"})
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, _instance.run_rollout_sync, start_state, params
        )
        return _instance._to_native(result)

    @app.post("/game/autopilot")
    async def autopilot(req: AutopilotRequest):
        if not _instance.kg_loaded:
            raise HTTPException(status_code=400, detail="KG not loaded.")

        if req.enabled:
            if _instance._autopilot_enabled:
                return {"status": "already_running"}
            _instance.start_autopilot(req.model_dump(exclude={"enabled"}))
            _instance._autopilot_task = asyncio.create_task(_autopilot_loop())
            return {"status": "started", "mode": req.mode}
        else:
            if _instance._autopilot_task and not _instance._autopilot_task.done():
                _instance._autopilot_task.cancel()
                _instance._autopilot_task = None
            _instance.stop_autopilot()
            return {"status": "stopped"}

    @app.get("/game/autopilot/status")
    async def autopilot_status():
        return _instance.get_autopilot_status()

    @app.get("/game/episodes")
    async def get_episodes(
        page: int = Query(1, ge=1),
        per_page: int = Query(10, ge=1, le=9999),
        sort: str = Query("id_desc"),
        search: Optional[str] = Query(None),
    ):
        all_eps = []
        for ep in _instance._episodes:
            all_eps.append(
                {
                    "id": ep.id,
                    "result": ep.result,
                    "score": ep.score,
                    "markov_flow": ep.markov_flow,
                    "events": ep.events,
                    "plans": ep.plans,
                    "timestamp": ep.timestamp,
                    "mode": ep.mode,
                    "steps": ep.steps,
                    "match_mode": ep.match_mode,
                    "source": "autopilot",
                }
            )
        for ep_id in sorted(_instance._history_store.keys()):
            frames = _instance._history_store[ep_id]
            meta = _instance._history_meta.get(ep_id, {})
            markov_flow = []
            events_list = []
            for fr in frames:
                nid = fr.get("nid")
                if nid is None:
                    nid = fr.get("state_cluster", (0, 0))
                markov_flow.append([nid, fr.get("action_code", fr.get("action", ""))])
                events_list.append(
                    {
                        "state_id": nid,
                        "action": fr.get("action", ""),
                        "event_type": fr.get("action_source", ""),
                        "my_count": fr.get("my_count", 0),
                        "enemy_count": fr.get("enemy_count", 0),
                        "hp_my": fr.get("hp_my", 0),
                        "hp_enemy": fr.get("hp_enemy", 0),
                        "game_loop": fr.get("game_loop", 0),
                        "end_game_flag": fr.get("end_game_flag", False),
                    }
                )
            all_eps.append(
                {
                    "id": 10000 + ep_id,
                    "result": meta.get("result", "Unknown"),
                    "score": meta.get("score", 0.0),
                    "markov_flow": markov_flow,
                    "events": events_list,
                    "plans": meta.get("plans", []),
                    "timestamp": meta.get("timestamp", ""),
                    "mode": meta.get("mode", ""),
                    "steps": len(frames),
                    "match_mode": "",
                    "source": "agent_buffer",
                }
            )
        if search:
            search_lower = search.lower()
            all_eps = [
                ep
                for ep in all_eps
                if search_lower in ep["result"].lower()
                or search_lower in ep["mode"].lower()
            ]
        sort_key = {
            "id_desc": lambda e: -e["id"],
            "id_asc": lambda e: e["id"],
            "score_desc": lambda e: -e["score"],
            "score_asc": lambda e: e["score"],
        }.get(sort, lambda e: -e["id"])
        all_eps.sort(key=sort_key)
        total = len(all_eps)
        start = (page - 1) * per_page
        end = start + per_page
        page_items = all_eps[start:end]
        return {
            "episodes": page_items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, (total + per_page - 1) // per_page),
        }

    @app.post("/game/episodes/ack")
    async def ack_episodes(request: Request):
        body = await request.json()
        ids = body.get("ids", [])
        agent_ids = [eid - 10000 for eid in ids if eid >= 10000]
        if agent_ids:
            _instance.ack_history(agent_ids)
        return {"ok": True, "acked": len(agent_ids)}

    @app.post("/game/episodes/clear")
    async def clear_episodes():
        _instance._episodes.clear()
        _instance._episode_buffer.clear()
        _instance._episode_counter = 0
        _instance._current_plans.clear()
        _instance._history_store.clear()
        _instance._history_meta.clear()
        _instance._plans_by_episode.clear()
        _instance.add_log("info", "system", "对局记录已清空")
        return {"ok": True}

    @app.post("/game/action")
    async def send_action(req: ActionRequest):
        _instance.bridge.put_action(req.action)
        return {"status": "queued", "action": req.action}

    @app.post("/game/fallback")
    async def set_fallback(req: FallbackRequest):
        _instance.bridge.put_action(f"__fallback:{req.action}")
        return {"status": "fallback_set", "action": req.action}

    @app.post("/game/control")
    async def control(req: ControlRequest):
        if req.command not in (
            "start",
            "pause",
            "resume",
            "stop",
            "step",
            "run_episode",
        ):
            raise HTTPException(
                status_code=400, detail=f"Invalid command: {req.command}"
            )
        if req.command == "run_episode":
            _instance.bridge.set_run_episode()
            _instance.add_log("info", "control", "单局运行: 将在下一局结束后暂停")
        _instance.bridge.send_control(req.command)
        return {"status": "sent", "command": req.command}

    @app.get("/game/events")
    async def game_events():
        return StreamingResponse(
            _instance.event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/game/load_kg")
    async def load_kg(kg_file: str = Query(...)):
        success = _instance.load_kg(kg_file)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to load KG file.")
        return {"status": "loaded", "kg_file": kg_file}

    @app.get("/game/actions")
    async def list_actions():
        from src.sc2env.kg_guided_agent import ACTION_NAME_MAP, FALLBACK_ACTIONS

        return {
            "action_map": ACTION_NAME_MAP,
            "available_actions": FALLBACK_ACTIONS,
        }

    @app.get("/game/beam_params")
    async def get_beam_params():
        return _instance._beam_search_params

    @app.post("/game/beam_params")
    async def set_beam_params(req: PredictRequest):
        _instance.set_beam_params(
            beam_width=req.beam_width,
            max_steps=req.max_steps,
            min_visits=req.min_visits,
            min_cum_prob=req.min_cum_prob,
            score_mode=req.score_mode,
            max_state_revisits=req.max_state_revisits,
            discount_factor=req.discount_factor,
        )
        return {"status": "updated", "params": _instance._beam_search_params}

    _FILTER_CFG_FILE = Path(
        os.environ.get(
            "LIVE_FILTER_CFG",
            os.path.join(
                str(Path(__file__).resolve().parent.parent.parent),
                ".live_filter_cfg.json",
            ),
        )
    )
    _FILTER_DEFAULTS = {
        "levels": ["info", "success", "warn", "error"],
        "sources": ["game", "api", "autopilot"],
        "types": ["info", "action", "fallback", "result", "episode", "control"],
    }

    @app.get("/game/filter-config")
    async def get_filter_config():
        if _FILTER_CFG_FILE.is_file():
            try:
                return json.loads(_FILTER_CFG_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return _FILTER_DEFAULTS

    @app.post("/game/filter-config")
    async def save_filter_config(req: Dict[str, Any] = None):
        cfg = req if req else {}
        for key in _FILTER_DEFAULTS:
            if key not in cfg or not isinstance(cfg[key], list):
                cfg[key] = _FILTER_DEFAULTS[key]
        try:
            _FILTER_CFG_FILE.write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
        return {"status": "saved", "config": cfg}

    @app.post("/game/shutdown")
    async def shutdown():
        if _instance._episode_buffer:
            _instance._commit_buffer(result="Shutdown")
        _instance.bridge.request_stop()
        threading.Thread(target=lambda: os._exit(0), daemon=True).start()
        return {"status": "shutting_down"}

    @app.post("/game/window_pos")
    async def set_window_pos(req: dict):
        try:
            x = int(req.get("x", 50))
            y = int(req.get("y", 50))
            w = int(req.get("w", 640))
            h = int(req.get("h", 480))
        except (ValueError, TypeError):
            return {"ok": False, "error": "invalid params"}
        try:
            import ctypes
            import threading

            def _do_move():
                import time

                time.sleep(0.3)
                hwnd = ctypes.windll.user32.FindWindowW(None, "StarCraft II")
                if hwnd:
                    ctypes.windll.user32.SetWindowPos(hwnd, 0, x, y, w, h, 0x0040)

            threading.Thread(target=_do_move, daemon=True).start()
            _instance.add_log("info", "system", f"窗口移动: ({x},{y}) {w}x{h}")
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    return app


def run_server(
    bridge: GameBridge,
    host: str = "0.0.0.0",
    port: int = 8000,
    kg_file: Optional[str] = None,
    data_dir: Optional[str] = None,
):
    import uvicorn

    app = create_app(bridge)
    if kg_file and _instance:
        _instance.load_kg(kg_file, data_dir=data_dir)
    uvicorn.run(app, host=host, port=port, log_level="info")
