#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Finetune Trainer — 状态微调机训练流程

三阶段:
  A: 从试验记录取 top-K 参数组合重跑，收集高质量 episode 决策流作为基准
  B: 基于状态置信度优先探索（优先对同一高质量 episode 中更多不同状态进行多次探索）
  C: 评估收敛（avg replacement_score > 0.6 且已探索比例 > 80%）

Usage:
    python scripts/finetune_trainer.py --top_k 5 --explore_budget 500
"""

import sys
import os
import json
import time
import socket
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import requests
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ROOT_DIR
from src.decision.finetune_model import FinetuneModel

_DEFAULT_CONFIG = ROOT_DIR / "configs" / "learner_config.yaml"
_RESULTS_DIR = ROOT_DIR / "output" / "learner_results"
_RUNS_DIR = _RESULTS_DIR / "runs"
_TRIALS_DIR = _RESULTS_DIR / "trials"
_FINETUNE_DIR = _RESULTS_DIR / "finetune_runs"
_FINETUNE_SAMPLES_DIR = _RESULTS_DIR / "finetune_samples"


def _find_free_port(exclude=None):
    exclude = set(exclude or [])
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            if port not in exclude:
                return port
    raise RuntimeError("Cannot find free port")


def _wait_for_server(port, timeout=30):
    url = f"http://127.0.0.1:{port}/game/status"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def _set_beam_params(port, params):
    url = f"http://127.0.0.1:{port}/game/beam_params"
    try:
        r = requests.post(url, json=params, timeout=10)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _pause_game(port):
    try:
        requests.post(
            f"http://127.0.0.1:{port}/game/control",
            json={"command": "pause"},
            timeout=5,
        )
    except Exception:
        pass


def _resume_game(port):
    try:
        requests.post(
            f"http://127.0.0.1:{port}/game/control",
            json={"command": "resume"},
            timeout=5,
        )
    except Exception:
        pass


def _wait_for_file_progress(trial_dir, target, timeout_minutes=60, poll_interval=3):
    deadline = time.time() + timeout_minutes * 60
    progress_file = trial_dir / "progress.json"
    ep_file = trial_dir / "episodes.jsonl"
    last_logged = 0

    while time.time() < deadline:
        try:
            if progress_file.exists():
                data = json.loads(progress_file.read_text(encoding="utf-8"))
                done = data.get("completed", 0)
                if done >= target:
                    print(f"  trial done: {done} episodes")
                    return True
                if done >= last_logged + 10:
                    print(f"  progress: {done}/{target}")
                    last_logged = done
        except (json.JSONDecodeError, OSError):
            pass
        time.sleep(poll_interval)

    print(f"  [ERROR] timeout ({timeout_minutes} min)")
    return False


def _load_study_trials():
    study_db = str(_RESULTS_DIR / "study.db")
    if not Path(study_db).exists():
        return []
    try:
        study = optuna.load_study(
            study_name="beam_search", storage=f"sqlite:///{study_db}"
        )
        trials = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
                trials.append(
                    {
                        "number": t.number,
                        "value": t.value,
                        "params": dict(t.params),
                        "user_attrs": dict(t.user_attrs),
                    }
                )
        trials.sort(key=lambda x: x["value"], reverse=True)
        return trials
    except Exception as e:
        print(f"[ERROR] failed to load study: {e}")
        return []


def _extract_decision_flow(trial_dir: Path) -> List[List[dict]]:
    ep_file = trial_dir / "episodes.jsonl"
    if not ep_file.exists():
        return []

    flows = []
    with open(str(ep_file), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
                frames = ep.get("frames", [])
                flow = []
                for frame in frames:
                    nid = frame.get("nid")
                    action_code = frame.get("action_code")
                    hp_my = frame.get("hp_my", 0)
                    hp_enemy = frame.get("hp_enemy", 0)
                    if action_code is None:
                        continue
                    if nid is None:
                        sc = frame.get("state_cluster")
                        if sc is not None:
                            nid = hash(tuple(sc))
                    if nid is not None:
                        flow.append(
                            {
                                "state_id": nid,
                                "action_code": action_code,
                                "hp_my": hp_my,
                                "hp_enemy": hp_enemy,
                            }
                        )
                if flow:
                    flows.append(flow)
            except json.JSONDecodeError:
                continue
    return flows


def _compute_flow_reward(flow: List[dict]) -> float:
    if len(flow) < 2:
        return 0.0
    total_delta = 0.0
    for i in range(len(flow)):
        delta_my = 0.0
        delta_enemy = 0.0
        if i > 0:
            delta_my = flow[i]["hp_my"] - flow[i - 1]["hp_my"]
            delta_enemy = flow[i]["hp_enemy"] - flow[i - 1]["hp_enemy"]
        total_delta += delta_my - delta_enemy
    return total_delta


def _filter_high_quality_flows(
    flows: List[List[dict]],
    win_rate_threshold: float = 0.5,
) -> List[List[dict]]:
    if not flows:
        return []
    rewards = [_compute_flow_reward(f) for f in flows]
    median_reward = float(np.median(rewards)) if rewards else 0.0
    return [f for f, r in zip(flows, rewards) if r >= median_reward]


def _select_exploration_targets(
    model: FinetuneModel,
    base_flows: List[List[dict]],
    covered_states: set,
) -> Dict[int, str]:
    candidates = []
    for flow in base_flows:
        for step in flow:
            sid = step["state_id"]
            if sid in covered_states:
                continue
            best_action = None
            best_q = -float("inf")
            for ac, est in model.q_table.get(sid, {}).items():
                if est.avg_reward > best_q:
                    best_q = est.avg_reward
                    best_action = ac
            if best_action is None:
                best_action = step["action_code"]
            score = model.replacement_score(sid, best_action)
            candidates.append((sid, best_action, score))

    if not candidates:
        return {}

    candidates.sort(key=lambda x: x[2])
    targets = {}
    seen_states = set()
    for sid, action, _ in candidates:
        if sid not in seen_states:
            targets[sid] = action
            seen_states.add(sid)
            if len(targets) >= 5:
                break
    return targets


class FinetuneTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.top_k = config.get("top_k", 5)
        self.episodes_per_config = config.get("episodes_per_config", 50)
        self.win_rate_threshold = config.get("win_rate_threshold", 0.5)
        self.explore_budget = config.get("explore_budget", 500)
        self.convergence_threshold = config.get("convergence_threshold", 0.6)
        self.sigma = config.get("sigma", 0.5)
        self.target_visits = config.get("target_visits", 10)

        self.model = FinetuneModel(
            sigma=self.sigma,
            target_visits=self.target_visits,
        )
        self._port = None
        self._proc = None
        self._explore_count = 0
        self._finetune_id = 0
        self._sample_dir: Optional[Path] = None
        self._kg_file = config.get("kg_file")
        self._data_dir = config.get("data_dir")

    def run(self):
        study_trials = _load_study_trials()
        if len(study_trials) < self.top_k:
            print(
                f"[ERROR] only {len(study_trials)} completed trials, need {self.top_k}"
            )
            return

        _FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
        _FINETUNE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        _TRIALS_DIR.mkdir(parents=True, exist_ok=True)

        existing = list(_FINETUNE_DIR.glob("finetune_*_run.json"))
        self._finetune_id = len(existing) + 1

        self._sample_dir = _FINETUNE_SAMPLES_DIR / f"finetune_{self._finetune_id:04d}"
        self._sample_dir.mkdir(parents=True, exist_ok=True)

        top_trials = study_trials[: self.top_k]
        print(f"\n{'=' * 60}")
        print(f"Finetune Training #{self._finetune_id:04d}")
        print(f"  top-{self.top_k} trials: {[t['number'] for t in top_trials]}")
        print(f"  episodes/config: {self.episodes_per_config}")
        print(f"  explore budget: {self.explore_budget}")
        print(f"  sigma: {self.sigma}, target_visits: {self.target_visits}")
        print(f"{'=' * 60}")

        run_record = {
            "finetune_id": self._finetune_id,
            "config": self.config,
            "top_trials": [t["number"] for t in top_trials],
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "running",
        }
        run_path = _FINETUNE_DIR / f"finetune_{self._finetune_id:04d}_run.json"
        with open(str(run_path), "w", encoding="utf-8") as f:
            json.dump(run_record, f, ensure_ascii=False, indent=2)

        try:
            base_flows = self._phase_a(top_trials)

            if not base_flows:
                print("[WARN] no high-quality episodes collected, aborting")
                run_record["status"] = "no_data"
                self._save_run_record(run_record, run_path)
                return

            print(f"\nPhase A complete: {len(base_flows)} high-quality flows collected")

            self._phase_b(base_flows)
            self._phase_c()

            model_path = _RESULTS_DIR / "finetune_model.pkl"
            self.model.save(str(model_path))
            print(f"\nModel saved to {model_path}")

            stats = self.model.get_overall_stats()
            run_record["status"] = "completed"
            run_record["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            run_record["model_stats"] = stats
            run_record["model_path"] = str(model_path)
            self._save_run_record(run_record, run_path)

            print(f"\n{'=' * 60}")
            print("Finetune Training Complete")
            print(
                f"  explored states: {stats['explored_states']}/{stats['total_states']}"
            )
            print(f"  avg replacement_score: {stats['avg_replacement_score']}")
            print(f"  explored ratio: {stats['explored_ratio']:.2%}")
            print(f"{'=' * 60}")

        except KeyboardInterrupt:
            print("\ninterrupted, saving progress...")
            model_path = _RESULTS_DIR / "finetune_model.pkl"
            self.model.save(str(model_path))
            run_record["status"] = "interrupted"
            run_record["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            run_record["model_stats"] = self.model.get_overall_stats()
            self._save_run_record(run_record, run_path)
        finally:
            self._shutdown()

    def _phase_a(self, top_trials: list) -> List[List[dict]]:
        print(f"\n--- Phase A: Collecting base episodes ---")
        self._startup()

        all_flows = []
        for idx, trial_info in enumerate(top_trials):
            trial_number = trial_info["number"]
            params = trial_info["params"]
            print(f"\n  [{idx + 1}/{self.top_k}] Rerunning Trial #{trial_number}")

            trial_dir = _TRIALS_DIR / f"trial_{trial_number:04d}"
            if trial_dir.exists():
                existing_flows = _extract_decision_flow(trial_dir)
                if existing_flows:
                    print(f"    using existing data: {len(existing_flows)} flows")
                    all_flows.extend(existing_flows)
                    continue

            trial_dir.mkdir(parents=True, exist_ok=True)
            for fname in ("episodes.jsonl", "progress.json"):
                fp = trial_dir / fname
                if fp.exists():
                    fp.unlink()

            send_params = dict(params)
            send_params["local_result_dir"] = str(trial_dir)
            send_params["target_episodes"] = self.episodes_per_config
            send_params["trial_number"] = trial_number

            sent = False
            for _attempt in range(5):
                if _set_beam_params(self._port, send_params):
                    sent = True
                    break
                print(f"    [WARN] set_beam_params failed, retrying...")
                time.sleep(5)
            if not sent:
                print(
                    f"    [ERROR] failed to set beam params for Trial #{trial_number}"
                )
                continue

            _resume_game(self._port)

            time.sleep(1)
            completed = _wait_for_file_progress(
                trial_dir, self.episodes_per_config, timeout_minutes=60
            )
            _pause_game(self._port)

            if completed:
                flows = _extract_decision_flow(trial_dir)
                all_flows.extend(flows)
                print(f"    collected {len(flows)} flows")
            else:
                print(f"    [WARN] trial #{trial_number} timed out")

        high_quality = _filter_high_quality_flows(all_flows, self.win_rate_threshold)
        print(f"\n  Total flows: {len(all_flows)}")
        print(f"  High-quality: {len(high_quality)}")

        self._update_model_from_flows(high_quality)

        for trial_info in top_trials:
            tn = trial_info["number"]
            remark = f"微调训练 #{self._finetune_id:04d} Phase A (top-K)"
            self._write_trial_run_json(tn, trial_info["params"], "completed", remark)

        base_ep_path = self._sample_dir / "base_episodes.jsonl"
        with open(str(base_ep_path), "w", encoding="utf-8") as f:
            for flow in high_quality:
                f.write(json.dumps(flow, ensure_ascii=False) + "\n")

        return high_quality

    def _phase_b(self, base_flows: List[List[dict]]):
        print(f"\n--- Phase B: Exploration ---")
        covered_states = set()
        for flow in base_flows:
            for step in flow:
                sid = step["state_id"]
                if sid in self.model.q_table:
                    covered_states.add(sid)

        explore_flows = []
        while self._explore_count < self.explore_budget:
            targets = _select_exploration_targets(
                self.model, base_flows, covered_states
            )
            if not targets:
                print("  no more targets to explore")
                break

            print(
                f"\n  Explore [{self._explore_count}/{self.explore_budget}]: "
                f"{len(targets)} targets"
            )

            trial_number = 10000 + self._finetune_id * 10000 + self._explore_count
            trial_dir = _TRIALS_DIR / f"trial_{trial_number:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            for fname in ("episodes.jsonl", "progress.json"):
                fp = trial_dir / fname
                if fp.exists():
                    fp.unlink()

            base_params = self._pick_base_params(base_flows)

            send_params = dict(base_params)
            send_params["local_result_dir"] = str(trial_dir)
            send_params["target_episodes"] = 1
            send_params["trial_number"] = trial_number
            send_params["exploration_targets"] = targets

            sent = False
            for _attempt in range(5):
                if _set_beam_params(self._port, send_params):
                    sent = True
                    break
                time.sleep(5)
            if not sent:
                print(f"    [ERROR] failed to set beam params")
                break

            _resume_game(self._port)
            time.sleep(1)
            completed = _wait_for_file_progress(trial_dir, 1, timeout_minutes=10)
            _pause_game(self._port)

            if completed:
                flows = _extract_decision_flow(trial_dir)
                explore_flows.extend(flows)
                self._update_model_from_flows(flows)

                for sid in targets:
                    covered_states.add(sid)

                self._explore_count += 1
                print(
                    f"    collected {len(flows)} flows, total explored: {self._explore_count}"
                )

                remark = f"微调训练 #{self._finetune_id:04d} Phase B 探索 [{self._explore_count}/{self.explore_budget}]"
                self._write_trial_run_json(
                    trial_number, send_params, "completed", remark
                )

                if self._explore_count % 10 == 0:
                    self._save_q_table_snapshot()
                    progress = {
                        "explore_count": self._explore_count,
                        "covered_states": len(covered_states),
                        "model_stats": self.model.get_overall_stats(),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    progress_path = self._sample_dir / "progress.json"
                    with open(str(progress_path), "w", encoding="utf-8") as f:
                        json.dump(progress, f, ensure_ascii=False, indent=2)
            else:
                print(f"    [WARN] exploration episode timed out")

        explore_ep_path = self._sample_dir / "explore_episodes.jsonl"
        with open(str(explore_ep_path), "w", encoding="utf-8") as f:
            for flow in explore_flows:
                f.write(json.dumps(flow, ensure_ascii=False) + "\n")

        self._save_q_table_snapshot()

    def _phase_c(self):
        print(f"\n--- Phase C: Convergence check ---")
        stats = self.model.get_overall_stats()
        avg_rs = stats["avg_replacement_score"]
        explored_ratio = stats["explored_ratio"]
        converged = avg_rs >= self.convergence_threshold and explored_ratio >= 0.8

        print(
            f"  avg replacement_score: {avg_rs:.4f} (threshold: {self.convergence_threshold})"
        )
        print(f"  explored ratio: {explored_ratio:.2%} (threshold: 80%)")
        print(f"  converged: {converged}")

        return converged

    def _update_model_from_flows(self, flows: List[List[dict]]):
        for flow in flows:
            reward = _compute_flow_reward(flow)
            for step in flow:
                sid = step["state_id"]
                ac = step["action_code"]
                if sid is not None and ac:
                    self.model.update(sid, ac, reward)

    def _pick_base_params(self, base_flows: List[List[dict]]) -> dict:
        cfg_path = _DEFAULT_CONFIG
        if cfg_path.exists():
            try:
                with open(str(cfg_path), "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                space = cfg.get("search_space", {})
                params = {
                    "beam_width": (space.get("beam_width", [3, 10]))[0],
                    "lookahead_steps": (space.get("lookahead_steps", [3, 15]))[0],
                    "score_mode": (space.get("score_mode", ["quality"]))[0],
                    "action_strategy": "best_beam",
                    "mode": "multi_step",
                    "min_visits": (space.get("min_visits", [1, 10]))[0],
                    "max_state_revisits": (space.get("max_state_revisits", [1, 5]))[0],
                    "min_cum_prob": (space.get("min_cum_prob", [0.001, 0.1]))[0],
                    "discount_factor": (space.get("discount_factor", [0.5, 1.0]))[0],
                    "enable_backup": False,
                    "backup_score_threshold": 0.3,
                    "backup_distance_threshold": 0.2,
                    "epsilon": 0.1,
                    "masked_actions": [],
                }
                return params
            except Exception:
                pass
        return {
            "beam_width": 3,
            "lookahead_steps": 5,
            "score_mode": "quality",
            "action_strategy": "best_beam",
            "mode": "multi_step",
            "min_visits": 1,
            "max_state_revisits": 2,
            "min_cum_prob": 0.01,
            "discount_factor": 0.9,
            "enable_backup": False,
            "backup_score_threshold": 0.3,
            "backup_distance_threshold": 0.2,
            "epsilon": 0.1,
            "masked_actions": [],
        }

    def _save_q_table_snapshot(self):
        snapshot = {}
        for sid, actions in self.model.q_table.items():
            snapshot[str(sid)] = {}
            for ac, est in actions.items():
                snapshot[str(sid)][ac] = {
                    "visits": est.visits,
                    "avg_reward": round(est.avg_reward, 4),
                    "confidence": round(est.confidence, 4),
                    "action_rank": round(est.action_rank, 4),
                }
        path = self._sample_dir / "q_table_snapshot.json"
        with open(str(path), "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

    def _save_run_record(self, record, path):
        with open(str(path), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    def _write_trial_run_json(
        self,
        trial_number: int,
        params: dict,
        status: str,
        remark: str = "",
    ):
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        run_path = _RUNS_DIR / f"trial_{trial_number:04d}_run.json"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "trial": trial_number,
            "port": self._port,
            "target_episodes": params.get("target_episodes", 0),
            "params": {k: v for k, v in params.items() if not k.startswith("local_")},
            "start_time": now,
            "status": status,
            "batch": None,
            "source_trial": None,
            "remark": remark,
        }
        if status in ("completed", "timeout"):
            record["end_time"] = now
        with open(str(run_path), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    def _startup(self):
        if self._port and self._proc and self._proc.poll() is None:
            return

        port = _find_free_port(exclude={8000, 8501, 8502})
        self._port = port

        cfg_path = _DEFAULT_CONFIG
        game_cfg = {}
        if cfg_path.exists():
            try:
                with open(str(cfg_path), "r", encoding="utf-8") as f:
                    full_cfg = yaml.safe_load(f)
                game_cfg = full_cfg.get("game", {})
            except Exception:
                pass

        cmd = [
            sys.executable,
            str(ROOT_DIR / "scripts" / "run_live_game.py"),
            "--mode",
            "all",
            "--port",
            str(port),
            "--map_key",
            game_cfg.get("map_key", "sce-1"),
            "--max_episodes",
            "0",
            "--autopilot_mode",
            game_cfg.get("autopilot_mode", "multi_step"),
        ]
        _kg_file = self._kg_file or game_cfg.get("kg_file")
        _data_dir = self._data_dir or game_cfg.get("data_dir")
        if _kg_file:
            cmd.extend(["--kg_file", _kg_file])
        if _data_dir:
            cmd.extend(["--data_dir", _data_dir])
        if game_cfg.get("fallback_action"):
            cmd.extend(["--fallback_action", game_cfg["fallback_action"]])

        log_path = (
            self._sample_dir / "trainer.log"
            if self._sample_dir
            else _TRIALS_DIR / "trainer.log"
        )
        log_file = open(str(log_path), "w", encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW
        if sys.platform == "win32":
            flags |= subprocess.CREATE_NEW_PROCESS_GROUP
        self._proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            cwd=str(ROOT_DIR),
            creationflags=flags,
        )

        print(f"  SC2 process started (PID={self._proc.pid}, port={port})")
        if not _wait_for_server(port, timeout=30):
            self._proc.terminate()
            raise RuntimeError("server startup timeout")

        if _kg_file:
            try:
                requests.post(
                    f"http://127.0.0.1:{port}/game/load_kg",
                    json={
                        "kg_file": _kg_file,
                        "data_dir": _data_dir,
                    },
                    timeout=30,
                )
                print(f"  KG loaded: {_kg_file}")
            except requests.RequestException as e:
                print(f"  [WARN] KG load failed: {e}")

    def _shutdown(self):
        if self._port:
            try:
                requests.post(f"http://127.0.0.1:{self._port}/game/shutdown", timeout=5)
            except Exception:
                pass
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="PredictionRTS Finetune Trainer")
    parser.add_argument(
        "--top_k", type=int, default=5, help="top-K trials for base episodes"
    )
    parser.add_argument("--episodes_per_config", type=int, default=50)
    parser.add_argument("--explore_budget", type=int, default=500)
    parser.add_argument("--sigma", type=float, default=0.5, help="k-NN distance decay")
    parser.add_argument(
        "--target_visits", type=int, default=10, help="confidence target"
    )
    parser.add_argument("--convergence_threshold", type=float, default=0.6)
    parser.add_argument("--win_rate_threshold", type=float, default=0.5)
    parser.add_argument("--kg_file", type=str, default=None, help="KG pickle file")
    parser.add_argument("--data_dir", type=str, default=None, help="Training data dir")
    args = parser.parse_args()

    config = {
        "top_k": args.top_k,
        "episodes_per_config": args.episodes_per_config,
        "explore_budget": args.explore_budget,
        "sigma": args.sigma,
        "target_visits": args.target_visits,
        "convergence_threshold": args.convergence_threshold,
        "win_rate_threshold": args.win_rate_threshold,
        "kg_file": args.kg_file,
        "data_dir": args.data_dir,
    }

    trainer = FinetuneTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
