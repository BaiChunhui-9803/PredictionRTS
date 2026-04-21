"""
KGGuidedAgent — 基于 Experience Transition Graph 引导的实时决策 Agent

继承 SmartAgent，保留全部 action 执行方法，重写 step() 决策逻辑：
    1. 每帧在本地执行状态聚类 + beam search 规划（无需跨进程通信）
    2. 回放模式下从本地列表消费预设动作
    3. 无有效决策时回退到用户配置的默认策略
    4. 每 N 局批量推送对局记录到 bridge_server 供 Web 显示
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.sc2env.agent import SmartAgent
from src.sc2env.bridge import GameBridge
from src.sc2env.config import get_map_config

_MAP_CONFIG, _MAP, _ENV_CONFIG, _ALG_CONFIG, _PATH_CONFIG = get_map_config("sce-1")

ACTION_NAME_MAP: Dict[str, str] = {
    "k_means_000": "k_means_000",
    "k_means_025": "k_means_025",
    "k_means_050": "k_means_050",
    "k_means_075": "k_means_075",
    "k_means_100": "k_means_100",
    "action_ATK_nearest": "action_ATK_nearest",
    "action_ATK_clu_nearest": "action_ATK_clu_nearest",
    "action_ATK_nearest_weakest": "action_ATK_nearest_weakest",
    "action_ATK_clu_nearest_weakest": "action_ATK_clu_nearest_weakest",
    "action_ATK_threatening": "action_ATK_threatening",
    "action_DEF_clu_nearest": "action_DEF_clu_nearest",
    "action_MIX_gather": "action_MIX_gather",
    "action_MIX_lure": "action_MIX_lure",
    "action_MIX_sacrifice_lure": "action_MIX_sacrifice_lure",
    "do_randomly": "do_randomly",
    "do_nothing": "do_nothing",
}

FALLBACK_ACTIONS = list(SmartAgent.actions)


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


class KGGuidedAgent(SmartAgent):
    def __init__(
        self,
        bridge: GameBridge,
        fallback_action: str = "action_ATK_nearest_weakest",
        initial_bktree_data: Optional[dict] = None,
        state_id_map: Optional[Dict[Tuple[int, int], int]] = None,
        kg=None,
        transitions=None,
        dist_matrix=None,
        mode: str = "multi_step",
        beam_params: Optional[Dict[str, Any]] = None,
        replay_actions: Optional[List[str]] = None,
        replay_runs: int = 1,
        action_strategy: str = "best_beam",
    ):
        super(KGGuidedAgent, self).__init__()
        self.bridge = bridge
        self._fallback_action = fallback_action
        self._prev_state_cluster: Optional[Tuple[int, int]] = None
        self._last_action_executed: str = ""
        self._action_history: List[Dict[str, Any]] = []
        self._pending_cluster: Optional[str] = None
        self._bktree_loaded = False
        self._state_id_map = state_id_map or {}
        self._ep_history: List[Dict[str, Any]] = []
        self._ep_counter: int = 0
        self._prev_end_game_flag: bool = False

        self.kg = kg
        self.transitions = transitions
        self._dist_matrix = dist_matrix
        self._mode = mode
        self._beam_params = beam_params or {}
        self._action_strategy = action_strategy
        self._replay_actions: List[str] = list(replay_actions) if replay_actions else []
        self._replay_idx: int = 0
        self._replay_done: bool = False
        self._replay_per_ep: int = len(self._replay_actions)
        self._replay_frame_count: int = 0
        self._replay_runs_remaining: int = max(1, replay_runs) if replay_actions else 0

        self._action_plan: List[str] = []
        self._planned_states: List[int] = []
        self._plan_idx: int = 0
        self._last_plan_snap: Optional[Dict] = None

        self._ep_batch: List[Dict[str, Any]] = []
        self._ep_push_batch_size: int = 5
        self._frame_count: int = 0
        self._status_push_interval: int = 50

        if initial_bktree_data is not None:
            self._load_bktree_from_data(initial_bktree_data)
            self._bktree_loaded = True

        print(
            f"[KGGuidedAgent] mode={self._mode}, state_id_map={len(self._state_id_map)}, "
            f"kg={'loaded' if self.kg else 'None'}, transitions={'loaded' if self.transitions else 'None'}, "
            f"dist_matrix={'loaded' if self._dist_matrix is not None else 'None'}"
        )

    def _load_bktree_from_data(self, data: dict) -> None:
        from src.structure.BKTree_sc2 import ClusterNode, BKTree, get_max_cluster_id

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

        if "primary" in data and data["primary"] is not None:
            primary_root = deserialize_node(data["primary"])
            if primary_root is not None:
                self.primary_bktree = BKTree(
                    self.custom_distance_manager.multi_distance, distance_index=0
                )
                self.primary_bktree.root = primary_root
                max_id = get_max_cluster_id(self.primary_bktree)
                if max_id >= self.primary_bktree.next_cluster_id:
                    self.primary_bktree.next_cluster_id = max_id + 1

        if "secondary" in data:
            for cluster_id, sec_data in data["secondary"].items():
                sec_root = deserialize_node(sec_data)
                if sec_root is not None:
                    tree = BKTree(
                        self.custom_distance_manager.multi_distance, distance_index=1
                    )
                    tree.root = sec_root
                    max_id = get_max_cluster_id(tree)
                    if max_id >= tree.next_cluster_id:
                        tree.next_cluster_id = max_id + 1
                    self.secondary_bktree[int(cluster_id)] = tree

    def new_game(self):
        super().new_game()
        if not hasattr(self, "_mode"):
            return
        self._replay_frame_count = 0
        self._prev_end_game_flag = False
        if self._mode == "replay" and self._replay_actions:
            if self._ep_history:
                self._ep_counter += 1
                self._ep_batch.append(
                    {
                        "episode_id": self._ep_counter,
                        "frames": list(self._ep_history),
                        "result": self.end_game_state or "Dogfall",
                        "score": 0,
                    }
                )
                self._ep_history = []
            self._flush_ep_batch()

            self._replay_runs_remaining -= 1
            if self._replay_runs_remaining > 0:
                self._replay_idx = 0
                self._replay_done = False
            else:
                self._replay_idx = len(self._replay_actions)

        if hasattr(self, "ctx") and self.ctx:
            self.ctx.episode_count += 1
            ep = self.ctx.episode_count
            self.bridge.update_status(
                episode=ep,
                agent_mode=self._mode,
                agent_params={
                    "mode": self._mode,
                    "beam_width": self._beam_params.get("beam_width", 3),
                    "max_steps": self._beam_params.get("lookahead_steps", 5),
                    "min_visits": self._beam_params.get("min_visits", 1),
                    "min_cum_prob": self._beam_params.get("min_cum_prob", 0.01),
                    "score_mode": self._beam_params.get("score_mode", "quality"),
                    "max_state_revisits": self._beam_params.get(
                        "max_state_revisits", 2
                    ),
                    "discount_factor": self._beam_params.get("discount_factor", 0.9),
                    "action_strategy": self._action_strategy,
                    "epsilon": self._beam_params.get("epsilon", 0.1),
                    "enable_backup": self._beam_params.get("enable_backup", False),
                    "backup_score_threshold": self._beam_params.get(
                        "backup_score_threshold", 0.3
                    ),
                    "backup_distance_threshold": self._beam_params.get(
                        "backup_distance_threshold", 0.2
                    ),
                    "replay_runs": self._replay_runs_remaining
                    + (1 if self._replay_idx < len(self._replay_actions) else 0),
                    "replay_per_ep": self._replay_per_ep,
                },
            )
            self.bridge.put_event(
                {
                    "level": "info",
                    "source": "game",
                    "message": f"Episode #{ep} 开始",
                }
            )

    def _resolve_action(self, raw_action: str) -> Optional[str]:
        self._pending_cluster = None
        if len(raw_action) == 2 and raw_action[0].isdigit() and raw_action[1].isalpha():
            cluster_idx = int(raw_action[0])
            action_idx = ord(raw_action[1]) - ord("a")
            if 0 <= cluster_idx < len(self.clusters):
                self._pending_cluster = self.clusters[cluster_idx]
            if 0 <= action_idx < len(self.actions):
                combat_action = self.actions[action_idx]
                if hasattr(self, combat_action):
                    return combat_action
            return None
        mapped = ACTION_NAME_MAP.get(raw_action, raw_action)
        if hasattr(self, mapped):
            return mapped
        return None

    def _query_readonly(self, norm_state):
        p_id, p_dist = self.primary_bktree.query_nearest(norm_state)
        if p_id is None:
            p_id, p_dist = 1, 0.0
        sec_tree = self.secondary_bktree.get(p_id)
        if sec_tree is None or sec_tree.root is None:
            return (p_id, 1)
        s_id, s_dist = sec_tree.query_nearest(norm_state)
        return (p_id, s_id if s_id is not None else 1)

    def get_state_cluster(self, norm_state):
        if self._bktree_loaded and self.primary_bktree.root is not None:
            return self._query_readonly(norm_state)
        return super(KGGuidedAgent, self).get_state_cluster(norm_state)

    def _record_action(self, state_cluster: Any, action_name: str, source: str) -> None:
        entry = {
            "game_loop": 0,
            "state_cluster": str(state_cluster),
            "action": action_name,
            "source": source,
        }
        if len(self._action_history) > 0:
            entry["game_loop"] = self._action_history[-1].get("game_loop", 0) + 1
        self._action_history.append(entry)
        if len(self._action_history) > 2000:
            self._action_history = self._action_history[-1000:]

    def _get_next_replay_action(self) -> Optional[str]:
        if self._replay_idx < len(self._replay_actions):
            action = self._replay_actions[self._replay_idx]
            self._replay_idx += 1
            return action
        return None

    def _get_plan_from_beam(
        self, state_id: int, lookahead_steps: int
    ) -> Tuple[Optional[str], List[str], List[int], List[Dict], List[Dict]]:
        from src.decision.kg_beam_search import plan_action

        if self.kg is None or self.transitions is None:
            return None, [], [], [], []

        plan = plan_action(
            self.kg,
            self.transitions,
            state_id,
            beam_width=self._beam_params.get("beam_width", 3),
            max_steps=lookahead_steps,
            min_visits=self._beam_params.get("min_visits", 1),
            min_cum_prob=self._beam_params.get("min_cum_prob", 0.01),
            score_mode=self._beam_params.get("score_mode", "quality"),
            max_state_revisits=self._beam_params.get("max_state_revisits", 2),
            discount_factor=self._beam_params.get("discount_factor", 0.9),
            action_strategy=self._action_strategy,
        )

        if plan.recommended_action is None:
            return None, [], [], [], []

        beam_dicts = []
        for r in plan.beam_results:
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

        beam_paths = []
        for rank, path in enumerate(plan.beam_paths):
            is_chosen = rank == plan.best_path_index
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

        return (
            plan.recommended_action,
            plan.action_plan,
            plan.planned_states,
            beam_dicts,
            beam_paths,
        )

    def _local_decide(self, state_id: int) -> Tuple[Optional[str], str, Optional[Dict]]:
        if self.kg is None or self.transitions is None:
            return None, "fallback", None

        if self._mode == "single_step":
            self._action_plan = []
            self._planned_states = []
            self._plan_idx = 0

        lookahead = self._beam_params.get("lookahead_steps", 5)
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
                is_diverge = True
                self._action_plan = []
                self._planned_states = []
                self._plan_idx = 0

        if self._plan_idx >= len(self._action_plan):
            try:
                first_action, actions, states, beam_dicts, beam_paths = (
                    self._get_plan_from_beam(state_id, lookahead)
                )
                if not actions:
                    return None, "fallback", None
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
                self._last_plan_snap = plan_snap
            except Exception:
                return None, "fallback", None

        action = self._action_plan[self._plan_idx]
        if self._plan_idx == 0:
            event_type = "diverge" if is_diverge else "kg_plan"
        else:
            event_type = "kg_follow"
        self._plan_idx += 1
        return action, event_type, self._last_plan_snap

    def _push_status(self, obs, state_cluster_str, my_units, enemy_units):
        if self._frame_count % self._status_push_interval == 0:
            self.bridge.update_status(
                frame=obs.observation.game_loop[0],
                my_count=len(my_units),
                enemy_count=len(enemy_units),
                state_cluster=state_cluster_str,
                my_total_hp=int(sum(u.health for u in my_units)),
                enemy_total_hp=int(sum(u.health for u in enemy_units)),
            )
        self._frame_count += 1

    def _flush_ep_batch(self):
        if not self._ep_batch:
            return
        for ep in self._ep_batch:
            try:
                self.bridge.put_history(ep)
            except Exception:
                pass
        self._ep_batch = []

    def step(self, obs, env):
        from pysc2.lib import actions as sc2_actions

        super(SmartAgent, self).step(obs, env)

        if obs.last():
            result_event = {
                "type": "episode_end",
                "result": self.end_game_state,
                "frames": int(self.end_game_frames)
                if hasattr(self.end_game_frames, "__int__")
                else self.end_game_frames,
                "episode": getattr(self.ctx, "episode_count", 0) if self.ctx else 0,
            }
            self.bridge.put_event(result_event)
            level = (
                "success"
                if self.end_game_state == "Win"
                else ("error" if self.end_game_state == "Loss" else "warn")
            )
            self.bridge.put_event(
                {
                    "level": level,
                    "source": "game",
                    "message": f"Episode #{result_event['episode']} 结束: {self.end_game_state} (frame={result_event['frames']})",
                }
            )
            self.bridge.update_status(result=self.end_game_state)
            if self._ep_history:
                self._ep_counter += 1
                self._ep_batch.append(
                    {
                        "episode_id": self._ep_counter,
                        "frames": list(self._ep_history),
                        "result": self.end_game_state or "Dogfall",
                        "score": 0,
                    }
                )
                self._ep_history = []
                self._replay_frame_count = 0
            self._flush_ep_batch()
            return sc2_actions.RAW_FUNCTIONS.no_op()

        if obs.first():
            self._termination_signaled = False
            if not self._initial_spawned:
                unit_list_my = self.get_my_units_by_type(obs, _MAP["unit_type"])
                unit_list_enemy = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
                self._initial_units_my = [(u.x, u.y) for u in unit_list_my]
                self._initial_units_enemy = [(u.x, u.y) for u in unit_list_enemy]
                self._initial_spawned = True
            self._replay_frame_count = 0

            unit_list_my = self.get_my_units_by_type(obs, _MAP["unit_type"])
            unit_list_enemy = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
            self.score_attack_max = sum([item["health"] for item in unit_list_enemy])
            self.score_defense_max = sum([item["health"] for item in unit_list_my])
            self.score_cumulative_attack_last = sum(
                [item["health"] for item in unit_list_enemy]
            )
            self.score_cumulative_defense_last = sum(
                [item["health"] for item in unit_list_my]
            )

        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])

        if not self._termination_signaled and len(enemy_units) == 0:
            self.end_game_state = "Win"
            self.end_game_flag = True
            self._termination_signaled = True
            env.f_result = "win"
            if self.end_game_frames > obs.observation.game_loop:
                self.end_game_frames = obs.observation.game_loop

        if not self._termination_signaled and len(my_units) == 0:
            self.end_game_state = "Loss"
            self.end_game_flag = True
            self._termination_signaled = True
            env.f_result = "loss"
            if self.end_game_frames > obs.observation.game_loop:
                self.end_game_frames = obs.observation.game_loop

        map_resolution = _ENV_CONFIG["_MAP_RESOLUTION"]
        my_sorted = sorted(my_units, key=lambda u: u.tag)
        enemy_sorted = sorted(enemy_units, key=lambda u: u.tag)
        state_norm = {
            "red_army": [
                (
                    u.x / (map_resolution / 2) - 1.0,
                    1.0 - u.y / (map_resolution / 2),
                    u.health / 45.0,
                )
                for u in my_sorted
            ],
            "blue_army": [
                (
                    u.x / (map_resolution / 2) - 1.0,
                    1.0 - u.y / (map_resolution / 2),
                    u.health / 45.0,
                )
                for u in enemy_sorted
            ],
        }
        state_cluster = self.get_state_cluster(state_norm)
        self._prev_state_cluster = state_cluster

        self._push_status(obs, str(state_cluster), my_units, enemy_units)

        p, s = int(state_cluster[0]), int(state_cluster[1])
        nid = self._state_id_map.get((p, s))
        action_code = "4c"
        action_to_execute = None
        action_source = "fallback"
        plan_snap = None

        hp_my = int(sum(u.health for u in my_units))
        hp_enemy = int(sum(u.health for u in enemy_units))

        if self._mode == "replay":
            replay_action = self._get_next_replay_action()
            if replay_action is not None:
                resolved = self._resolve_action(replay_action)
                if resolved is not None:
                    action_to_execute = resolved
                    action_source = "replay"
                    action_code = replay_action
                else:
                    action_to_execute = self._resolve_action(self._fallback_action)
                    action_source = "replay_fallback"
            else:
                if not self._replay_done:
                    self._replay_done = True
                    self._flush_ep_batch()
                    self.bridge.put_event(
                        {
                            "level": "info",
                            "source": "game",
                            "message": f"回放完成: 共执行 {self._replay_idx}/{len(self._replay_actions)} 步",
                        }
                    )
                    self.bridge.update_status(replay_done=True)
                    try:
                        self.bridge.send_control("pause")
                    except Exception:
                        pass
                return sc2_actions.RAW_FUNCTIONS.no_op()

        elif self._mode != "replay" and nid is not None:
            action_code_raw, evt_type, plan_snap = self._local_decide(nid)
            if action_code_raw is not None:
                resolved = self._resolve_action(action_code_raw)
                if resolved is not None:
                    action_to_execute = resolved
                    action_source = evt_type
                    action_code = action_code_raw
                else:
                    action_to_execute = self._resolve_action(self._fallback_action)
                    action_source = "fallback"
            else:
                action_to_execute = self._resolve_action(self._fallback_action)
                action_source = "fallback"

        if action_to_execute is None:
            action_to_execute = self._resolve_action(self._fallback_action)
            if action_to_execute is None:
                action_to_execute = "action_ATK_nearest_weakest"
            action_source = "fallback"

        if action_source == "fallback" and action_to_execute in self.actions:
            a_idx = self.actions.index(action_to_execute)
            action_code = "4" + chr(ord("a") + a_idx)

        self._last_action_executed = action_to_execute
        self._record_action(state_cluster, action_to_execute, action_source)

        if (my_units or enemy_units) and not (
            self._mode == "replay"
            and self._replay_per_ep > 0
            and self._replay_frame_count >= self._replay_per_ep
        ):
            self._ep_history.append(
                {
                    "state_cluster": (p, s),
                    "nid": nid,
                    "action": action_to_execute,
                    "action_code": action_code,
                    "action_source": action_source,
                    "my_count": len(my_units),
                    "enemy_count": len(enemy_units),
                    "hp_my": hp_my,
                    "hp_enemy": hp_enemy,
                    "game_loop": int(obs.observation.game_loop[0]),
                    "end_game_flag": self.end_game_flag,
                    "plan": plan_snap,
                    "my_units_pos": [
                        {"x": float(u.x), "y": float(u.y), "hp": float(u.health)}
                        for u in my_units
                    ],
                    "enemy_units_pos": [
                        {"x": float(u.x), "y": float(u.y), "hp": float(u.health)}
                        for u in enemy_units
                    ],
                }
            )
            self._replay_frame_count += 1

        end_flag = self.end_game_flag
        if end_flag and not self._prev_end_game_flag:
            if self._ep_history:
                self._ep_counter += 1
                self._ep_batch.append(
                    {
                        "episode_id": self._ep_counter,
                        "frames": list(self._ep_history),
                        "result": self.end_game_state,
                        "score": float(hp_my - hp_enemy),
                    }
                )
                self._ep_history = []
                self._replay_frame_count = 0
                if len(self._ep_batch) >= self._ep_push_batch_size:
                    self._flush_ep_batch()
        self._prev_end_game_flag = end_flag

        if self._pending_cluster and hasattr(self, self._pending_cluster):
            self.cluster_result = getattr(self, self._pending_cluster)(obs)

        if hasattr(self, action_to_execute):
            result = getattr(self, action_to_execute)(obs)
            if result is not None:
                return result

        return sc2_actions.RAW_FUNCTIONS.no_op()

    def set_fallback_action(self, action_name: str) -> bool:
        resolved = self._resolve_action(action_name)
        if resolved is not None:
            self._fallback_action = resolved
            return True
        return False

    def get_action_history(self) -> List[Dict[str, Any]]:
        return list(self._action_history)
