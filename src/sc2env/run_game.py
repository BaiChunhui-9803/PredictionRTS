import time
import os
import shutil
from datetime import datetime
from absl import flags, app

from pysc2.env import sc2_env, run_loop, environment
from pysc2.lib import actions, features, units
from s2clientprotocol import debug_pb2 as sc_debug
from s2clientprotocol import common_pb2 as sc_common

from typing import Optional

from src.sc2env.config import get_map_config
from src.sc2env.agent import SmartAgent, Agent
from src.sc2env.utils import GameContext, init_game
from src.sc2env.bridge import GameBridge

_MAP_CONFIG, _MAP, _ENV_CONFIG, _ALG_CONFIG, _PATH_CONFIG = get_map_config("sce-1")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

flags.DEFINE_string(
    "run_name", None, "Name for this run. Defaults to map_key_YYYYMMDD_HHMMSS"
)
FLAGS = flags.FLAGS


def kill_all_units(env, obs):
    unit_tags = [u.tag for u in obs.raw_units]
    debug_command = [
        sc_debug.DebugCommand(kill_unit=sc_debug.DebugKillUnit(tag=unit_tags))
    ]
    env._controllers[0].debug(debug_command)


def spawn_units(env, agent):
    unit_type_id = _MAP["unit_type_id"]
    debug_commands = []
    for pos in agent._initial_units_my:
        debug_commands.append(
            sc_debug.DebugCommand(
                create_unit=sc_debug.DebugCreateUnit(
                    unit_type=unit_type_id,
                    owner=1,
                    pos=sc_common.Point2D(x=pos[0], y=pos[1]),
                    quantity=1,
                )
            )
        )
    for pos in agent._initial_units_enemy:
        debug_commands.append(
            sc_debug.DebugCommand(
                create_unit=sc_debug.DebugCreateUnit(
                    unit_type=unit_type_id,
                    owner=2,
                    pos=sc_common.Point2D(x=pos[0], y=pos[1]),
                    quantity=1,
                )
            )
        )
    env._controllers[0].debug(debug_commands)


def _move_sc2_window(x=50, y=50, w=640, h=480, timeout=10):
    import ctypes
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        hwnd = ctypes.windll.user32.FindWindowW(None, "StarCraft II")
        if hwnd:
            ctypes.windll.user32.SetWindowPos(hwnd, 0, x, y, w, h, 0x0040)
            return
        time.sleep(0.3)


def run_loop_custom(
    agents,
    env,
    reset_frames=0,
    max_frames=0,
    max_episodes=0,
    bridge: Optional[GameBridge] = None,
):
    """A run loop to have agents and an environment interact.

    When *bridge* is provided, supports:
      - pause/resume via bridge control signals
      - graceful stop via bridge stop signal
    """
    total_frames = 0
    env.total_episodes = 0
    start_time = time.time()
    global_test_flag = False

    env.f_start = True

    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
        agent.setup(obs_spec, act_spec)

    if bridge:
        bridge.update_status(running=True, paused=False, mode="playing")
        bridge.put_event({"level": "success", "source": "game", "message": "游戏启动"})

    try:
        while not max_episodes or env.total_episodes < max_episodes:
            if bridge and bridge.should_stop():
                bridge.update_status(running=False, mode="stopped")
                bridge.put_event(
                    {"level": "info", "source": "game", "message": "游戏停止"}
                )
                break

            if bridge:
                ctrl = bridge.check_control()
                if ctrl == "step":
                    bridge.send_control("pause")
                    bridge.put_event(
                        {"level": "info", "source": "game", "message": "步进 1 帧"}
                    )
                elif ctrl == "pause":
                    bridge.put_event(
                        {"level": "info", "source": "game", "message": "游戏暂停"}
                    )
                    _resume = bridge.wait_until_resumed()
                    if _resume == "step":
                        bridge.send_control("pause")
                        bridge.put_event(
                            {"level": "info", "source": "game", "message": "步进 1 帧"}
                        )
                    else:
                        bridge.put_event(
                            {"level": "info", "source": "game", "message": "游戏恢复"}
                        )
                        continue
                elif ctrl == "stop":
                    bridge.update_status(running=False, mode="stopped")
                    bridge.put_event(
                        {"level": "info", "source": "game", "message": "游戏停止"}
                    )
                    break

            env.total_episodes += 1
            env.f_result = None
            timesteps = env.reset()
            for a in agents:
                a.reset()
            while True:
                if bridge and bridge.should_stop():
                    bridge.update_status(running=False, mode="stopped")
                    bridge.put_event(
                        {"level": "info", "source": "game", "message": "游戏停止"}
                    )
                    return

                if bridge:
                    ctrl = bridge.check_control()
                    if ctrl == "step":
                        bridge.send_control("pause")
                        bridge.put_event(
                            {"level": "info", "source": "game", "message": "步进 1 帧"}
                        )
                    elif ctrl == "pause":
                        bridge.put_event(
                            {"level": "info", "source": "game", "message": "游戏暂停"}
                        )
                        _resume = bridge.wait_until_resumed()
                        if _resume == "step":
                            bridge.send_control("pause")
                            bridge.put_event(
                                {
                                    "level": "info",
                                    "source": "game",
                                    "message": "步进 1 帧",
                                }
                            )
                        else:
                            bridge.put_event(
                                {
                                    "level": "info",
                                    "source": "game",
                                    "message": "游戏恢复",
                                }
                            )
                            continue
                    elif ctrl == "stop":
                        bridge.update_status(running=False, mode="stopped")
                        bridge.put_event(
                            {"level": "info", "source": "game", "message": "游戏停止"}
                        )
                        return

                total_frames += 1

                timesteps[0].set_test_flag(global_test_flag)

                agent_actions = [
                    agent.step(timestep, env)
                    for agent, timestep in zip(agents, timesteps)
                ]

                if env.f_result == "win" or env.f_result == "loss":
                    result_str = str(env.f_result)
                    obs_now = timesteps[0]
                    my_units = agents[0].get_my_units_by_type(
                        obs_now, _MAP["unit_type"]
                    )
                    enemy_units = agents[0].get_enemy_units_by_type(
                        obs_now, _MAP["unit_type"]
                    )
                    my_hp = sum(u["health"] for u in my_units)
                    enemy_hp = sum(u["health"] for u in enemy_units)
                    score_val = my_hp - enemy_hp
                    try:
                        if bridge:
                            if bridge.check_run_episode():
                                bridge.put_event(
                                    {
                                        "level": "info",
                                        "source": "game",
                                        "message": f"单局结束: {result_str.upper()}, 已暂停",
                                    }
                                )
                                bridge.send_control("pause")
                                _resume = bridge.wait_until_resumed()
                                if _resume == "stop":
                                    bridge.update_status(running=False, mode="stopped")
                                    return
                                continue
                            agents[0].new_game()
                            agents[0].end_game_frames = (
                                _ENV_CONFIG["_MAX_STEP"] * _ENV_CONFIG["_STEP_MUL"]
                            )
                            agents[0].end_game_state = "Dogfall"
                            agents[0].end_game_flag = False
                            agents[0]._termination_signaled = False
                        else:
                            agents[0]._end_episode(timesteps[0])
                            agents[0]._termination_signaled = False
                        kill_all_units(env, timesteps[0].observation)
                        env.f_result = None
                        total_frames = 0
                        spawn_units(env, agents[0])
                        timesteps = env.step(agent_actions, 2)
                        env.total_episodes += 1
                        if bridge:
                            bridge.put_event(
                                {
                                    "level": "success"
                                    if result_str == "win"
                                    else "error",
                                    "source": "game",
                                    "message": f"判定 {result_str.upper()}, 得分: {score_val:+d} (我方{my_hp} vs 敌方{enemy_hp})",
                                }
                            )
                        continue
                    except Exception as e:
                        if bridge:
                            bridge.put_event(
                                {
                                    "level": "error",
                                    "source": "game",
                                    "message": f"kill+spawn 异常: {e}",
                                }
                            )
                if timesteps[0].last():
                    global_test_flag = not global_test_flag
                    break
                if reset_frames > 0 and total_frames > reset_frames:
                    if bridge:
                        if bridge.check_run_episode():
                            bridge.put_event(
                                {
                                    "level": "info",
                                    "source": "game",
                                    "message": "单局结束: 帧数超时, 已暂停",
                                }
                            )
                            bridge.send_control("pause")
                            _resume = bridge.wait_until_resumed()
                            if _resume == "stop":
                                bridge.update_status(running=False, mode="stopped")
                                return
                            continue
                        agents[0].new_game()
                        agents[0].end_game_frames = (
                            _ENV_CONFIG["_MAX_STEP"] * _ENV_CONFIG["_STEP_MUL"]
                        )
                        agents[0]._termination_signaled = False
                    else:
                        agents[0]._end_episode(timesteps[0])
                        agents[0]._termination_signaled = False
                    kill_all_units(env, timesteps[0].observation)
                    env.f_result = None
                    total_frames = 0
                    spawn_units(env, agents[0])
                    timesteps = env.step(agent_actions, 2)
                    env.total_episodes += 1
                    continue
                if max_frames and total_frames >= max_frames:
                    if bridge:
                        bridge.update_status(running=False, mode="stopped")
                    return
                timesteps = env.step(agent_actions)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        if bridge:
            bridge.update_status(running=False, mode="stopped")
    print(
        "Took %.3f seconds for %s steps: %.3f fps"
        % (elapsed_time, total_frames, total_frames / max(elapsed_time, 1e-6))
    )


def save_run(run_name):
    src = _PATH_CONFIG["_DATA_TRANSIT_PATH"]
    dst = os.path.join(_PATH_CONFIG["_RUNS_PATH"], run_name)
    if os.path.exists(dst):
        dst = dst + f"_dup_{datetime.now().strftime('%H%M%S')}"
    shutil.copytree(src, dst)
    print(f"Run saved to: {os.path.abspath(dst)}")


def run_game(
    map_key,
    run_name,
    bridge: Optional[GameBridge] = None,
    agent_type: str = "smart",
    fallback_action: str = "action_ATK_nearest_weakest",
    window_loc: Optional[tuple] = None,
    data_dir: Optional[str] = None,
):
    steps = _ENV_CONFIG["_MAX_STEP"]
    step_mul = _ENV_CONFIG["_STEP_MUL"]

    agent1 = None
    if agent_type == "kg_guided" and bridge is not None:
        from src.sc2env.kg_guided_agent import KGGuidedAgent

        bktree_data = None
        primary_bktree_path = _PATH_CONFIG.get("_GAME_PRIMARY_BKTREE_PATH", "")
        if primary_bktree_path and os.path.exists(primary_bktree_path):
            try:
                import json

                bktree_data = {"primary": None, "secondary": {}}
                with open(primary_bktree_path, "r") as f:
                    bktree_data["primary"] = json.load(f)
                prefix = _PATH_CONFIG.get("_GAME_SECONDARY_BKTREE_PREFIX", "")
                if prefix:
                    import glob as _glob

                    for sec_file in _glob.glob(f"{prefix}_*.json"):
                        cid_str = sec_file.rsplit("_", 1)[-1].replace(".json", "")
                        try:
                            with open(sec_file, "r") as sf:
                                bktree_data["secondary"][cid_str] = json.load(sf)
                        except Exception:
                            pass
            except Exception as e:
                print(f"Warning: Failed to load BKTree data: {e}")

        state_id_map = {}
        if data_dir:
            _sn_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                data_dir,
                "graph",
                "state_node.txt",
            )
            if not os.path.exists(_sn_path):
                _sn_path = os.path.join(data_dir, "graph", "state_node.txt")
            if os.path.exists(_sn_path):
                try:
                    with open(_sn_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split("\t")
                            if len(parts) >= 2:
                                import ast

                                ps = ast.literal_eval(parts[0])
                                nid = int(parts[1])
                                state_id_map[(int(ps[0]), int(ps[1]))] = nid
                except Exception as e:
                    print(f"Warning: Failed to load state_node.txt: {e}")

        agent1 = KGGuidedAgent(
            bridge=bridge,
            fallback_action=fallback_action,
            initial_bktree_data=bktree_data,
            state_id_map=state_id_map,
        )
    else:
        agent1 = SmartAgent()

    try:
        with sc2_env.SC2Env(
            map_name=_MAP["map_name"],
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_hard),
            ],
            agent_interface_format=features.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=_ENV_CONFIG["_MAP_RESOLUTION"],
            ),
            score_index=-1,
            disable_fog=True,
            step_mul=step_mul,
            game_steps_per_episode=steps * step_mul,
            # realtime=True
        ) as env:
            ctx = GameContext()
            init_game(ctx, _PATH_CONFIG)
            if window_loc:
                import threading

                threading.Thread(
                    target=_move_sc2_window, args=window_loc, daemon=True
                ).start()
            agent2 = Agent()
            agent1.ctx = ctx
            run_loop_custom(
                [agent1, agent2],
                env,
                reset_frames=_ENV_CONFIG["_RESET_FRAMES"],
                max_episodes=_ENV_CONFIG["_MAX_EPISODE"],
                bridge=bridge,
            )
            if bridge is None:
                save_run(run_name)
    except KeyboardInterrupt:
        if bridge is None:
            save_run(run_name)
        pass


def main(unused_argv):
    map_key = "sce-1"
    run_name = FLAGS.run_name
    if not run_name:
        run_name = f"{map_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Run name: {run_name}")
    run_game(map_key, run_name)


if __name__ == "__main__":
    app.run(main)
