"""
KGGuidedAgent — 基于 Knowledge Graph 引导的实时决策 Agent

继承 SmartAgent，保留全部 action 执行方法，重写 step() 决策逻辑：
    1. 每帧将游戏状态通过 GameBridge 推送给 FastAPI 服务
    2. 从 Bridge 读取外部动作指令（KG 推荐 / 手动选择）
    3. 无外部指令时回退到用户配置的默认策略

[预留接口] ACTION_NAME_MAP: KG action 名称 → SmartAgent 方法名 映射表
    当前使用 SmartAgent.actions / SmartAgent.clusters 作为可执行 action 集合，
    但 KG 中存储的 action 名称可能与 SmartAgent 方法名不完全一致，
    后续需要根据实际 KG 数据验证并调整此映射。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math

from src.sc2env.agent import SmartAgent, Agent
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


class KGGuidedAgent(SmartAgent):
    def __init__(
        self,
        bridge: GameBridge,
        fallback_action: str = "action_ATK_nearest_weakest",
        initial_bktree_data: Optional[dict] = None,
        state_id_map: Optional[Dict[Tuple[int, int], int]] = None,
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

        if initial_bktree_data is not None:
            self._load_bktree_from_data(initial_bktree_data)
            self._bktree_loaded = True

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

    def _build_observation_dict(
        self, obs, state_cluster=None, norm_state=None, my_units=None, enemy_units=None
    ) -> Dict[str, Any]:
        if my_units is None:
            my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        if enemy_units is None:
            enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])

        my_units_info = [
            {
                "tag": u.tag,
                "x": u.x,
                "y": u.y,
                "health": u.health,
                "health_ratio": u.health_ratio,
                "weapon_cooldown": u.weapon_cooldown,
            }
            for u in my_units
        ]
        enemy_units_info = [
            {
                "tag": u.tag,
                "x": u.x,
                "y": u.y,
                "health": u.health,
                "health_ratio": u.health_ratio,
                "weapon_cooldown": u.weapon_cooldown,
            }
            for u in enemy_units
        ]

        if state_cluster is None:
            if norm_state is None:
                norm_state = self.get_norm_state(obs)
            state_cluster = self.get_state_cluster(norm_state)

        obs_dict = {
            "game_loop": obs.observation.game_loop[0],
            "state_cluster": list(state_cluster)
            if isinstance(state_cluster, tuple)
            else state_cluster,
            "state_cluster_str": str(state_cluster),
            "my_units": my_units_info,
            "enemy_units": enemy_units_info,
            "my_count": len(my_units),
            "enemy_count": len(enemy_units),
            "my_total_hp": sum(u.health for u in my_units),
            "enemy_total_hp": sum(u.health for u in enemy_units),
            "last_action": self._last_action_executed,
            "end_game_state": self.end_game_state,
            "end_game_flag": self.end_game_flag,
            "episode": getattr(self.ctx, "episode_count", 0) if self.ctx else 0,
        }
        if norm_state is not None:
            obs_dict["norm_state"] = norm_state
        return obs_dict

    def _query_readonly(self, norm_state):
        p_id, p_dist = self.primary_bktree.query_nearest(norm_state)
        if p_id is None:
            p_id, p_dist = 1, 0.0
            self.bridge.put_event(
                {
                    "level": "error",
                    "source": "bktree",
                    "message": "BKTree: 主簇查询失败, 使用默认 P1",
                }
            )
        sec_tree = self.secondary_bktree.get(p_id)
        if sec_tree is None or sec_tree.root is None:
            self.bridge.put_event(
                {
                    "level": "error",
                    "source": "bktree",
                    "message": f"BKTree: 主簇 P{p_id} 无对应子树, 使用默认 S1",
                }
            )
            return (p_id, 1)
        s_id, s_dist = sec_tree.query_nearest(norm_state)
        if p_dist > 1.0 or s_dist > 0.5:
            self.bridge.put_event(
                {
                    "level": "warn",
                    "source": "bktree",
                    "message": (
                        f"BKTree 近似匹配: P{p_id}(dist={p_dist:.3f},thr=1.0) "
                        f"S{s_id}(dist={s_dist:.3f},thr=0.5)"
                    ),
                }
            )
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
            self.new_game()
            self.end_game_frames = _ENV_CONFIG["_MAX_STEP"] * _ENV_CONFIG["_STEP_MUL"]
            if self.end_game_flag != True:
                self.end_game_state = "Dogfall"
                self.end_game_flag = False
            return sc2_actions.RAW_FUNCTIONS.no_op()

        if obs.first():
            self._termination_signaled = False
            if not self._initial_spawned:
                unit_list_my = self.get_my_units_by_type(obs, _MAP["unit_type"])
                unit_list_enemy = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
                self._initial_units_my = [(u.x, u.y) for u in unit_list_my]
                self._initial_units_enemy = [(u.x, u.y) for u in unit_list_enemy]
                self._initial_spawned = True
            if self.ctx:
                self.ctx.episode_count += 1
                ep = self.ctx.episode_count
                self.bridge.update_status(episode=ep)
                self.bridge.put_event(
                    {
                        "level": "info",
                        "source": "game",
                        "message": f"Episode #{ep} 开始",
                    }
                )

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

        if (
            not self._termination_signaled
            and len(enemy_units) == 0
            and obs.observation["score_cumulative"][5]
            == obs.observation["score_cumulative"][3]
        ):
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

        self.bridge.put_event(
            {
                "level": "debug",
                "source": "cluster",
                "state_cluster": (int(state_cluster[0]), int(state_cluster[1])),
                "message": (
                    f"cluster=({state_cluster[0]},{state_cluster[1]}) "
                    f"my={len(my_units)} enemy={len(enemy_units)} "
                    f"hp={sum(u.health for u in my_units)}/{sum(u.health for u in enemy_units)}"
                ),
            }
        )

        obs_dict = self._build_observation_dict(
            obs,
            state_cluster=state_cluster,
            norm_state=state_norm,
            my_units=my_units,
            enemy_units=enemy_units,
        )
        self.bridge.put_observation(obs_dict)
        self.bridge.update_status(
            frame=obs.observation.game_loop[0],
            my_count=len(my_units),
            enemy_count=len(enemy_units),
            state_cluster=obs_dict["state_cluster_str"],
        )

        replay_action = self.bridge.get_replay_action()
        if replay_action is not None:
            resolved = self._resolve_action(replay_action)
            if resolved is not None:
                action_to_execute = resolved
                action_source = "replay"
                action_code = replay_action
                plan_snap = None
            else:
                action_to_execute = self._resolve_action(self._fallback_action)
                action_source = "replay_fallback"
                action_code = "4c"
                plan_snap = None
        else:
            external_action = self.bridge.get_action(timeout=0.05)
            action_to_execute = None
            action_source = "fallback"
            action_code = "4c"
            plan_snap = None

            if external_action is not None:
                if isinstance(external_action, dict):
                    action_code_raw = external_action.get("action_code")
                    plan_snap = external_action.get("plan_snap")
                    evt_type = external_action.get("event_type")
                    if action_code_raw is not None:
                        resolved = self._resolve_action(action_code_raw)
                        if resolved is not None:
                            action_to_execute = resolved
                            action_source = evt_type if evt_type else "external"
                            action_code = action_code_raw
                        else:
                            action_to_execute = self._resolve_action(
                                self._fallback_action
                            )
                            action_source = f"fallback_unknown({action_code_raw})"
                    else:
                        action_to_execute = self._resolve_action(self._fallback_action)
                        if action_to_execute is None:
                            action_to_execute = "action_ATK_nearest_weakest"
                        action_source = "fallback"
                else:
                    resolved = self._resolve_action(external_action)
                    if resolved is not None:
                        action_to_execute = resolved
                        action_source = "external"
                        action_code = external_action
                    else:
                        action_to_execute = self._resolve_action(self._fallback_action)
                        action_source = f"fallback_unknown({external_action})"

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

        p, s = int(state_cluster[0]), int(state_cluster[1])
        nid = self._state_id_map.get((p, s))
        hp_my = int(sum(u.health for u in my_units))
        hp_enemy = int(sum(u.health for u in enemy_units))
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
                    {"x": float(u.x), "y": float(u.y), "hp": u.health} for u in my_units
                ],
                "enemy_units_pos": [
                    {"x": float(u.x), "y": float(u.y), "hp": u.health}
                    for u in enemy_units
                ],
            }
        )

        end_flag = self.end_game_flag
        if end_flag and not self._prev_end_game_flag:
            self._ep_counter += 1
            self.bridge.put_history(
                {
                    "episode_id": self._ep_counter,
                    "frames": list(self._ep_history),
                    "result": self.end_game_state,
                    "score": float(hp_my - hp_enemy),
                }
            )
            self._ep_history = []
        self._prev_end_game_flag = end_flag

        if action_source == "fallback":
            self.bridge.put_event(
                {
                    "level": "warn",
                    "source": "game",
                    "message": f"回退策略: {action_to_execute}",
                }
            )
        else:
            self.bridge.put_event(
                {
                    "level": "info",
                    "source": "game",
                    "message": f"执行动作: {action_to_execute} ({action_source})",
                }
            )

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
