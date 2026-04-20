#!/usr/bin/env python
"""
Live Game Launcher — 启动实时对局系统

Usage:
    # 一键启动 (游戏进程 + API 服务)
    python scripts/run_live_game.py --map_key sce-1 --kg_file xxx.pkl

    # 分别启动
    python scripts/run_live_game.py --mode game --map_key sce-1
    python scripts/run_live_game.py --mode api --port 8000

    # 查看帮助
    python scripts/run_live_game.py --help
"""

import sys
import argparse
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ROOT_DIR


def _run_game_process(bridge, args):
    from absl import flags as absl_flags

    if not absl_flags.FLAGS.is_parsed():
        absl_flags.FLAGS(["run_live_game.py"])

    from src.sc2env.run_game import run_game

    window_loc = None
    if args.window_x is not None and args.window_y is not None:
        window_loc = (args.window_x, args.window_y, args.window_w, args.window_h)

    beam_params = {}
    if args.beam_width is not None:
        beam_params["beam_width"] = args.beam_width
    if args.lookahead_steps is not None:
        beam_params["lookahead_steps"] = args.lookahead_steps
    if args.score_mode is not None:
        beam_params["score_mode"] = args.score_mode
    if args.min_visits is not None:
        beam_params["min_visits"] = args.min_visits
    if args.max_state_revisits is not None:
        beam_params["max_state_revisits"] = args.max_state_revisits
    if args.min_cum_prob is not None:
        beam_params["min_cum_prob"] = args.min_cum_prob
    if args.discount_factor is not None:
        beam_params["discount_factor"] = args.discount_factor

    run_game(
        map_key=args.map_key,
        run_name=args.run_name,
        bridge=bridge,
        agent_type="kg_guided",
        fallback_action=args.fallback_action,
        window_loc=window_loc,
        data_dir=args.data_dir,
        autopilot_mode=args.autopilot_mode,
        beam_params=beam_params,
        replay_actions=args.replay_actions.split(",") if args.replay_actions else None,
        replay_runs=args.replay_runs,
        kg_file=args.kg_file,
    )


def _run_api_process(bridge, args):
    from src.sc2env.bridge_server import run_server

    run_server(
        bridge,
        host=args.host,
        port=args.port,
        kg_file=args.kg_file,
        data_dir=args.data_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="PredictionRTS Live Game System")
    parser.add_argument(
        "--mode",
        choices=["all", "game", "api"],
        default="all",
        help="all=game+api, game=SC2 only, api=server only",
    )
    parser.add_argument("--map_key", default="sce-1", help="Map config key")
    parser.add_argument(
        "--run_name", default=None, help="Run name (auto-generated if None)"
    )
    parser.add_argument(
        "--kg_file", default=None, help="KG pickle filename in cache/knowledge_graph/"
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Training data directory (e.g. data/MarineMicro_MvsM_4/6), contains graph/state_node.txt",
    )
    parser.add_argument(
        "--fallback_action",
        default="action_ATK_nearest_weakest",
        help="Default fallback action when no KG recommendation",
    )
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument(
        "--window_x", type=int, default=None, help="SC2 window X position"
    )
    parser.add_argument(
        "--window_y", type=int, default=None, help="SC2 window Y position"
    )
    parser.add_argument("--window_w", type=int, default=640, help="SC2 window width")
    parser.add_argument("--window_h", type=int, default=480, help="SC2 window height")
    parser.add_argument(
        "--autopilot_mode",
        default="multi_step",
        choices=["single_step", "multi_step", "replay"],
        help="Autopilot mode",
    )
    parser.add_argument("--beam_width", type=int, default=None, help="Beam width")
    parser.add_argument(
        "--lookahead_steps", type=int, default=None, help="Lookahead steps"
    )
    parser.add_argument("--score_mode", default=None, help="Score mode")
    parser.add_argument("--min_visits", type=int, default=None, help="Min visits")
    parser.add_argument(
        "--max_state_revisits", type=int, default=None, help="Max state revisits"
    )
    parser.add_argument(
        "--min_cum_prob", type=float, default=None, help="Min cumulative probability"
    )
    parser.add_argument(
        "--discount_factor", type=float, default=None, help="Discount factor"
    )
    parser.add_argument(
        "--replay_actions", default=None, help="Comma-separated replay action codes"
    )
    parser.add_argument(
        "--replay_runs", type=int, default=1, help="Number of replay runs"
    )
    args = parser.parse_args()

    if args.mode in ("game", "all") and args.kg_file is None:
        print(
            "Warning: --kg_file not specified. KG predictions will not work until loaded via API."
        )
        print("         You can load it later via: POST /game/load_kg?kg_file=xxx.pkl")

    if args.mode == "all":
        bridge = mp.get_context("spawn").Manager().Queue() if False else None
        from src.sc2env.bridge import GameBridge

        bridge = GameBridge()

        game_proc = mp.Process(
            target=_run_game_process,
            args=(bridge, args),
            name="sc2_game",
            daemon=True,
        )
        api_proc = mp.Process(
            target=_run_api_process,
            args=(bridge, args),
            name="bridge_api",
            daemon=True,
        )

        game_proc.start()
        api_proc.start()

        print(f"Game process PID: {game_proc.pid}")
        print(f"API server PID: {api_proc.pid}")
        print(f"API endpoint: http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop all processes.")

        try:
            game_proc.join()
        except KeyboardInterrupt:
            print("\nStopping...")
            bridge.request_stop()
            game_proc.join(timeout=5)
            api_proc.join(timeout=5)
            print("All processes stopped.")

    elif args.mode == "game":
        from src.sc2env.bridge import GameBridge

        bridge = GameBridge()
        _run_game_process(bridge, args)

    elif args.mode == "api":
        from src.sc2env.bridge import GameBridge

        bridge = GameBridge()
        print("API mode: no game process. Connect via existing bridge or standalone.")
        print(f"API endpoint: http://{args.host}:{args.port}")
        _run_api_process(bridge, args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
