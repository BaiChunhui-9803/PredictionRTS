import os
import re
import sys
import shutil
import subprocess
import time
import requests as _requests
from pathlib import Path
from typing import Optional, Dict

import pandas as pd

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna

from src import ROOT_DIR

_RESULTS_DIR = ROOT_DIR / "output" / "learner_results"
_RUNS_DIR = _RESULTS_DIR / "runs"
_TRIALS_DIR = _RESULTS_DIR / "trials"
_SUMMARY_PATH = _RESULTS_DIR / "study_summary.json"
_STUDY_DB = _RESULTS_DIR / "study.db"

_ACTION_STRATEGY_LABELS = {
    "best_beam": "Best Beam",
    "best_subtree_quality": "Best Subtree Quality",
    "best_subtree_winrate": "Best Subtree WinRate",
    "highest_transition_prob": "Highest Trans. Prob",
    "random_beam": "Random Beam",
    "epsilon_greedy": "Epsilon-Greedy",
}


def _load_summary():
    if _SUMMARY_PATH.exists():
        with open(str(_SUMMARY_PATH), "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_resource(ttl=60, show_spinner=False)
def _load_study():
    if _STUDY_DB.exists():
        try:
            storage = f"sqlite:///{_STUDY_DB}"
            study = optuna.load_study(study_name="beam_search", storage=storage)
            return study
        except Exception:
            pass
    return None


@st.cache_data(ttl=10, show_spinner=False)
def _get_running_trial_number():
    if not _PID_FILE.exists():
        return None
    if not _is_learner_alive():
        return None
    try:
        import sqlite3 as _sqlite

        db = _sqlite.connect(str(_STUDY_DB))
        cur = db.cursor()
        cur.execute(
            "SELECT t.number FROM trials t WHERE t.state = 'RUNNING' ORDER BY t.number DESC LIMIT 1"
        )
        row = cur.fetchone()
        db.close()
        return row[0] if row else None
    except Exception:
        return None


def _get_running_trial(study):
    if not study:
        return None
    num = _get_running_trial_number()
    if num is None:
        return None
    for t in study.trials:
        if t.number == num:
            return t
    return None


_PID_FILE = _RESULTS_DIR / ".learner_pid"


@st.cache_data(ttl=15, show_spinner=False)
def _is_learner_alive():
    if not _PID_FILE.exists():
        return False
    try:
        pid = int(_PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return False
    try:
        r = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            timeout=3,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return str(pid) in r.stdout.decode(errors="replace")
    except Exception:
        return False


def _kill_learner_process():
    if not _PID_FILE.exists():
        return
    try:
        pid = int(_PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    _is_learner_alive.clear()


def _kill_port_process(port: int):
    try:
        r = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        output = r.stdout.decode(errors="replace")
        for line in output.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    if pid.isdigit():
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", pid],
                            capture_output=True,
                            creationflags=subprocess.CREATE_NO_WINDOW
                            if sys.platform == "win32"
                            else 0,
                        )
    except Exception:
        pass


def _delete_trial(trial_number: int):
    run_json = _RUNS_DIR / f"trial_{trial_number:04d}_run.json"
    run_log = _RUNS_DIR / f"trial_{trial_number:04d}.log"
    run_json.unlink(missing_ok=True)
    run_log.unlink(missing_ok=True)

    trial_dir = _TRIALS_DIR / f"trial_{trial_number:04d}"
    if trial_dir.is_dir():
        shutil.rmtree(trial_dir, ignore_errors=True)

    study = _load_study()
    if not study:
        return
    for t in study.trials:
        if t.number == trial_number:
            try:
                study._storage.delete_trial(t._trial_id)
            except Exception:
                try:
                    study._storage.set_trial_state(
                        t._trial_id, optuna.trial.TrialState.FAIL
                    )
                except Exception:
                    pass
            break


_BATCH_RULES = [
    (0, 499, 1),
    (500, 1787, 2),
]


def _migrate_batches():
    if not _STUDY_DB.exists():
        return
    import sqlite3 as _sqlite

    try:
        db = _sqlite.connect(str(_STUDY_DB))
        cur = db.cursor()

        cur.execute(
            "SELECT value_json FROM study_user_attributes WHERE key = 'batch_migrated'"
        )
        row = cur.fetchone()
        if row and row[0] == '"true"':
            db.close()
            return

        cur.execute("SELECT study_id FROM studies WHERE study_name = 'beam_search'")
        srow = cur.fetchone()
        if not srow:
            db.close()
            return
        study_id = srow[0]

        cur.execute(
            "SELECT t.trial_id, t.number FROM trials t WHERE t.state IN ('COMPLETE', 'FAIL')"
        )
        trials = cur.fetchall()

        for trial_id, number in trials:
            cur.execute(
                "SELECT 1 FROM trial_user_attributes WHERE trial_id = ? AND key = 'batch'",
                (trial_id,),
            )
            if cur.fetchone():
                continue

            batch = 0
            for lo, hi, b in _BATCH_RULES:
                if lo <= number <= hi:
                    batch = b
                    break
            if batch > 0:
                cur.execute(
                    "INSERT OR IGNORE INTO trial_user_attributes (trial_id, key, value_json) VALUES (?, ?, ?)",
                    (trial_id, "batch", str(batch)),
                )

        cur.execute(
            "INSERT OR REPLACE INTO study_user_attributes (study_id, key, value_json) VALUES (?, ?, ?)",
            (study_id, "batch_migrated", '"true"'),
        )
        db.commit()
        db.close()

        if _RUNS_DIR.exists():
            for fp in _RUNS_DIR.glob("trial_*_run.json"):
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    if "batch" not in data or data["batch"] is None:
                        tn = data.get("trial")
                        if isinstance(tn, int):
                            batch = 0
                            for lo, hi, b in _BATCH_RULES:
                                if lo <= tn <= hi:
                                    batch = b
                                    break
                            if batch > 0:
                                data["batch"] = batch
                                with open(str(fp), "w", encoding="utf-8") as f:
                                    json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
    except Exception:
        pass


@st.cache_data(ttl=120, show_spinner=False)
def _get_batch_info():
    if not _STUDY_DB.exists():
        return {}
    import sqlite3 as _sqlite

    try:
        db = _sqlite.connect(str(_STUDY_DB))
        cur = db.cursor()
        cur.execute("""
            SELECT tua.value_json, COUNT(*), MIN(t.number), MAX(t.number)
            FROM trial_user_attributes tua
            JOIN trials t ON tua.trial_id = t.trial_id
            WHERE tua.key = 'batch' AND t.state IN ('COMPLETE', 'FAIL')
            GROUP BY tua.value_json
            ORDER BY tua.value_json
        """)
        rows = cur.fetchall()
        db.close()

        result = {}
        for val_json, count, min_num, max_num in rows:
            batch = int(val_json)
            result[batch] = {"count": count, "min_trial": min_num, "max_trial": max_num}
        return result
    except Exception:
        return {}


def _delete_batch(batch_num: int):
    import sqlite3 as _sqlite

    db = _sqlite.connect(str(_STUDY_DB))
    cur = db.cursor()

    cur.execute(
        """
        SELECT t.number FROM trial_user_attributes tua
        JOIN trials t ON tua.trial_id = t.trial_id
        WHERE tua.key = 'batch' AND tua.value_json = ?
    """,
        (str(batch_num),),
    )
    trial_numbers = [row[0] for row in cur.fetchall()]

    cur.execute(
        """
        SELECT t.trial_id FROM trial_user_attributes tua
        JOIN trials t ON tua.trial_id = t.trial_id
        WHERE tua.key = 'batch' AND tua.value_json = ?
    """,
        (str(batch_num),),
    )
    trial_ids = [row[0] for row in cur.fetchall()]

    for tid in trial_ids:
        for table in (
            "trial_user_attributes",
            "trial_system_attributes",
            "trial_params",
            "trial_values",
            "trial_intermediate_values",
            "trial_heartbeats",
        ):
            cur.execute(f"DELETE FROM {table} WHERE trial_id = ?", (tid,))
        cur.execute("DELETE FROM trials WHERE trial_id = ?", (tid,))

    db.commit()
    db.close()

    for tn in trial_numbers:
        run_json = _RUNS_DIR / f"trial_{tn:04d}_run.json"
        run_log = _RUNS_DIR / f"trial_{tn:04d}.log"
        run_json.unlink(missing_ok=True)
        run_log.unlink(missing_ok=True)
        trial_dir = _TRIALS_DIR / f"trial_{tn:04d}"
        if trial_dir.is_dir():
            shutil.rmtree(trial_dir, ignore_errors=True)

    _load_study.clear()
    _compute_importance.clear()
    _load_trials_from_db.clear()
    _load_runs.clear()
    _get_batch_info.clear()


def _clear_all_cache():
    for fn in (
        _load_study,
        _compute_importance,
        _load_trials_from_db,
        _load_runs,
        _get_batch_info,
    ):
        try:
            fn.clear()
        except Exception:
            pass


def _start_rerun(trial_numbers):
    if _is_learner_alive():
        st.error("参数寻优正在运行中，请先停止。")
        return

    episodes = st.session_state.get("learner_episodes", 100)

    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "parameter_learner.py"),
        "--rerun",
        str(trial_numbers[0]),
        "--episodes",
        str(episodes),
        "--resume",
    ]
    kg_file = st.session_state.get("_learner_kg_file", "")
    kg_data_dir = st.session_state.get("_learner_kg_data_dir", "")
    if kg_file:
        cmd.extend(["--kg_file", kg_file])
    if kg_data_dir:
        cmd.extend(["--data_dir", kg_data_dir])

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _TRIALS_DIR / "learner.log"
    log_file = open(str(log_path), "w", encoding="utf-8")
    st.session_state.learner_log_file = log_file
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    p = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        cwd=str(ROOT_DIR),
        creationflags=flags,
    )
    st.session_state.learner_proc = p
    _PID_FILE.write_text(str(p.pid))
    st.toast(f"重跑已启动 (PID: {p.pid})", icon="🚀")
    time.sleep(1)
    st.rerun()


_CONFIG_PATH = ROOT_DIR / "configs" / "learner_config.yaml"

_DEFAULT_SEARCH_SPACE = {
    "mode": ["single_step", "multi_step"],
    "beam_width": [1, 10],
    "lookahead_steps": [1, 15],
    "score_mode": ["quality", "future_reward", "win_rate"],
    "action_strategy": [
        "best_beam",
        "best_subtree_quality",
        "best_subtree_winrate",
        "highest_transition_prob",
        "random_beam",
        "epsilon_greedy",
    ],
    "min_visits": [1, 10],
    "max_state_revisits": [1, 5],
    "min_cum_prob": [0.001, 0.1],
    "discount_factor": [0.5, 1.0],
    "enable_backup": [True, False],
    "epsilon": [0.01, 0.5],
    "backup_score_threshold": [0.0, 1.0],
    "backup_distance_threshold": [0.0, 1.0],
}

_INT_PARAMS = {"beam_width", "lookahead_steps", "min_visits", "max_state_revisits"}
_FLOAT_PARAMS = {
    "min_cum_prob",
    "discount_factor",
    "epsilon",
    "backup_score_threshold",
    "backup_distance_threshold",
}
_CATEGORY_PARAMS = {"score_mode", "action_strategy", "mode"}


def _load_config_space() -> dict:
    if _CONFIG_PATH.exists():
        try:
            with open(str(_CONFIG_PATH), "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("search_space", {})
        except Exception:
            pass
    return {}


def _save_config_space(space: dict):
    cfg = {}
    if _CONFIG_PATH.exists():
        try:
            with open(str(_CONFIG_PATH), "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            pass
    cfg["search_space"] = space
    with open(str(_CONFIG_PATH), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _save_obj_config(w_win, w_score, alpha, cap):
    cfg = {}
    if _CONFIG_PATH.exists():
        try:
            with open(str(_CONFIG_PATH), "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            pass
    cfg.setdefault("objective", {})
    cfg["objective"]["win_rate_weight"] = w_win
    cfg["objective"]["avg_score_weight"] = w_score
    cfg["objective"]["stability_alpha"] = alpha
    cfg["objective"]["stability_cap"] = cap
    with open(str(_CONFIG_PATH), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _render_space_editor():
    space = _load_config_space()
    if not space:
        space = dict(_DEFAULT_SEARCH_SPACE)

    changed = False

    with st.container(border=True):
        st.markdown("**搜索空间**")

        for key in _INT_PARAMS:
            if key in space and isinstance(space[key], list) and len(space[key]) == 2:
                lo, hi = space[key]
                c1, c2 = st.columns(2)
                with c1:
                    new_lo = st.number_input(
                        f"{key} 最小", min_value=0, value=int(lo), key=f"sp_{key}_lo"
                    )
                with c2:
                    new_hi = st.number_input(
                        f"{key} 最大", min_value=0, value=int(hi), key=f"sp_{key}_hi"
                    )
                if new_lo != int(lo) or new_hi != int(hi):
                    space[key] = [int(new_lo), int(new_hi)]
                    changed = True

        for key in _FLOAT_PARAMS:
            if key in space and isinstance(space[key], list) and len(space[key]) == 2:
                lo, hi = space[key]
                c1, c2 = st.columns(2)
                with c1:
                    new_lo = st.number_input(
                        f"{key} 最小",
                        value=float(lo),
                        step=0.001,
                        format="%g",
                        key=f"sp_{key}_lo",
                    )
                with c2:
                    new_hi = st.number_input(
                        f"{key} 最大",
                        value=float(hi),
                        step=0.001,
                        format="%g",
                        key=f"sp_{key}_hi",
                    )
                if new_lo != float(lo) or new_hi != float(hi):
                    space[key] = [float(new_lo), float(new_hi)]
                    changed = True

        for key in _CATEGORY_PARAMS:
            if key in space and isinstance(space[key], list):
                val_str = ", ".join(str(v) for v in space[key])
                new_str = st.text_input(f"{key}", value=val_str, key=f"sp_{key}_cat")
                new_list = [v.strip() for v in new_str.split(",") if v.strip()]
                if new_list != space[key]:
                    space[key] = new_list
                    changed = True

        col_save, col_reset = st.columns(2)
        with col_save:
            if st.button(
                "保存修改", use_container_width=True, key="sp_save", type="primary"
            ):
                _save_config_space(space)
                st.toast("搜索空间已保存", icon="✅")
        with col_reset:
            if st.button("恢复默认", use_container_width=True, key="sp_reset"):
                _save_config_space(dict(_DEFAULT_SEARCH_SPACE))
                st.toast("已恢复默认搜索空间", icon="🔄")
                st.rerun()


def _show_log(filename):
    log_path = _RESULTS_DIR / filename
    if not log_path.exists():
        st.info("日志文件不存在。")
        return
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
        lines = content.strip().split("\n")
        show_lines = "\n".join(lines[-50:]) if len(lines) > 50 else content
        st.text_area(
            "日志（最近 50 行）", show_lines, height=300, label_visibility="collapsed"
        )
    except Exception as e:
        st.error(f"读取日志失败: {e}")


def _render_learner_sidebar(kg_entry: Optional[Dict] = None):
    kg_file = kg_entry.get("file", "") if kg_entry else ""
    kg_name = kg_entry.get("name", "") if kg_entry else ""
    kg_data_dir = kg_entry.get("data_dir", "") if kg_entry else ""

    st.session_state["_learner_kg_file"] = kg_file
    st.session_state["_learner_kg_data_dir"] = kg_data_dir

    if kg_file:
        st.caption(f"KG: {kg_name}")

    if "learner_proc" in st.session_state:
        proc = st.session_state.learner_proc
        if proc and proc.poll() is not None:
            del st.session_state.learner_proc

    study = _load_study()
    running_trial = _get_running_trial(study)

    st.divider()

    if running_trial:
        st.warning(f"正在运行 Trial #{running_trial.number}")
        st.progress(0.5)
        if st.button(
            "停止并清理",
            use_container_width=True,
            key="learner_stop",
            type="secondary",
        ):
            _kill_learner_process()
            time.sleep(1)
            _PID_FILE.unlink(missing_ok=True)
            st.toast("已停止并清理", icon="🛑")
            st.rerun()
        if st.button("刷新进度", use_container_width=True, key="learner_refresh"):
            st.rerun()
        if st.button("查看日志", use_container_width=True, key="learner_show_log"):
            _show_log("trials/learner.log")
        return

    st.number_input(
        "每轮对局数", min_value=10, max_value=1000, value=100, key="learner_episodes"
    )

    total_default = 50
    if study:
        completed_count = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        total_default = max(50, completed_count + 10)
    st.number_input(
        "总试验轮数",
        min_value=1,
        max_value=5000,
        value=total_default,
        key="learner_trials",
    )

    st.divider()
    st.markdown("**目标函数**")

    _obj_cfg = {}
    if _CONFIG_PATH.exists():
        try:
            with open(str(_CONFIG_PATH), "r", encoding="utf-8") as f:
                _full_cfg = yaml.safe_load(f) or {}
            _obj_cfg = _full_cfg.get("objective", {})
        except Exception:
            pass

    w_win = st.slider(
        "胜率权重 (w_win)",
        0.1,
        1.0,
        _obj_cfg.get("win_rate_weight", 0.8),
        step=0.05,
        key="learner_w_win",
    )
    w_score = st.slider(
        "得分权重 (w_score)",
        0.0,
        0.9,
        _obj_cfg.get("avg_score_weight", 0.2),
        step=0.05,
        key="learner_w_score",
    )
    alpha = st.slider(
        "稳定性惩罚强度 (alpha)",
        0.0,
        1.0,
        _obj_cfg.get("stability_alpha", 0.5),
        step=0.05,
        key="learner_alpha",
        help="0=不惩罚稳定性，1=极不稳定时得分项完全归零",
    )
    cap = st.number_input(
        "稳定性归一化上限 (cap)",
        min_value=0.1,
        max_value=10.0,
        value=_obj_cfg.get("stability_cap", 5.0),
        step=0.1,
        format="%g",
        key="learner_cap",
        help="stability 归一化参考值，超过此值按 cap 计算",
    )
    st.caption(
        "公式: `win_rate×w_win + norm_score×w_score×max(1−α×min(stability/cap,1), 0)`"
    )

    st.divider()

    if st.button(
        "查看/编辑搜索空间", use_container_width=True, key="learner_toggle_space"
    ):
        st.session_state._show_space_editor = not st.session_state.get(
            "_show_space_editor", False
        )

    if st.session_state.get("_show_space_editor", False):
        _render_space_editor()

    st.caption(f"数据目录: `{_RESULTS_DIR}`")

    start_clicked = st.button(
        "一键启动参数寻优",
        type="primary",
        use_container_width=True,
        key="learner_start",
    )

    if start_clicked:
        episodes = st.session_state.get("learner_episodes", 100)
        trials = st.session_state.get("learner_trials", 50)

        cmd = [
            sys.executable,
            str(ROOT_DIR / "scripts" / "parameter_learner.py"),
            "--episodes",
            str(episodes),
            "--trials",
            str(trials),
        ]
        if kg_file:
            cmd.extend(["--kg_file", kg_file])
        if kg_data_dir:
            cmd.extend(["--data_dir", kg_data_dir])

        if study:
            cmd.append("--resume")

        _save_obj_config(w_win, w_score, alpha, cap)

        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        _TRIALS_DIR.mkdir(parents=True, exist_ok=True)
        if "learner_log_file" in st.session_state:
            try:
                st.session_state.learner_log_file.close()
            except Exception:
                pass
        log_path = _TRIALS_DIR / "learner.log"
        log_file = open(str(log_path), "w", encoding="utf-8")
        st.session_state.learner_log_file = log_file
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        p = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            cwd=str(ROOT_DIR),
            creationflags=flags,
        )
        st.session_state.learner_proc = p
        st.toast(f"参数寻优已启动 (PID: {p.pid})", icon="🚀")
        time.sleep(1)
        st.rerun()


def _plot_objective_history(study):
    if not study or len(study.trials) < 2:
        st.info("试验数据不足，至少需要 2 轮完成的试验。")
        return

    fig = go.Figure()

    trial_numbers = []
    values = []
    best_so_far = []
    best_val = -float("inf")

    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
            trial_numbers.append(t.number)
            values.append(t.value)
            best_val = max(best_val, t.value)
            best_so_far.append(best_val)

    if not trial_numbers:
        st.info("没有完成的试验。")
        return

    fig.add_trace(
        go.Scatter(
            x=trial_numbers,
            y=values,
            mode="markers+lines",
            name="目标值",
            line=dict(color="#636EFA", width=1),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trial_numbers,
            y=best_so_far,
            mode="lines",
            name="最优目标值",
            line=dict(color="#EF553B", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="优化目标值变化",
        xaxis_title="Trial #",
        yaxis_title="Objective",
        height=350,
        margin=dict(l=50, r=30, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(ttl=120, show_spinner=False)
def _compute_importance():
    study = _load_study()
    if not study or len(study.trials) < 5:
        return None
    try:
        return dict(optuna.importance.get_param_importances(study))
    except Exception:
        return None


def _plot_importance(study):
    if not study or len(study.trials) < 5:
        st.info("试验数据不足，至少需要 5 轮完成才能计算参数重要性。")
        return

    importance = _compute_importance()
    if importance is None:
        st.info("无法计算参数重要性（可能缺少足够的参数变化）。")
        return

    if not importance:
        st.info("参数重要性数据为空。")
        return

    params = list(importance.keys())
    values = list(importance.values())
    sorted_pairs = sorted(zip(values, params), reverse=True)
    values = [p[0] for p in sorted_pairs]
    params = [p[1] for p in sorted_pairs]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=params,
            orientation="h",
            marker_color="#636EFA",
        )
    )
    fig.update_layout(
        title="参数重要性",
        xaxis_title="重要性",
        yaxis_title="参数",
        height=680,
        margin=dict(l=150, r=30, t=50, b=40),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _get_completed_trials(study, min_count=5):
    if not study or len(study.trials) < min_count:
        return []
    return [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]


def _classify_params(completed):
    all_keys = []
    for t in completed:
        for k in t.params:
            if k not in all_keys:
                all_keys.append(k)

    numeric_keys = [
        k
        for k in all_keys
        if isinstance(completed[0].params.get(k), (int, float))
        and all(t.params.get(k) is not None for t in completed)
    ]

    categorical_keys = [
        k
        for k in all_keys
        if isinstance(completed[0].params.get(k), str)
        and all(t.params.get(k) is not None for t in completed)
    ]

    return numeric_keys, categorical_keys


_METRIC_KEYS = ["objective", "win_rate", "avg_score", "stability"]
_METRIC_LABELS = {
    "objective": "Objective",
    "win_rate": "Win Rate",
    "avg_score": "Avg Score",
    "stability": "Stability",
}
_METRIC_COLORS = {
    "objective": "#636EFA",
    "win_rate": "#00CC96",
    "avg_score": "#EF553B",
    "stability": "#AB63FA",
}


def _get_trial_metrics(t) -> dict:
    return {
        "objective": float(t.value) if t.value is not None else None,
        "win_rate": t.user_attrs.get("win_rate"),
        "avg_score": t.user_attrs.get("avg_score"),
        "stability": t.user_attrs.get("stability"),
    }


def _plot_numeric_correlation(study):
    completed = _get_completed_trials(study)
    if len(completed) < 5:
        st.info("完成的试验不足 5 轮。")
        return

    numeric_keys, _ = _classify_params(completed)

    if not numeric_keys:
        st.info("无数值型参数。")
        return

    valid_metrics = []
    for mk in _METRIC_KEYS:
        vals = [_get_trial_metrics(t).get(mk) for t in completed]
        if all(v is not None for v in vals):
            valid_metrics.append(mk)

    if not valid_metrics:
        st.info("指标数据不完整。")
        return

    corr_results = {}
    for mk in valid_metrics:
        metric_vals = [_get_trial_metrics(t)[mk] for t in completed]
        data_matrix = []
        for t in completed:
            row = [float(t.params[k]) for k in numeric_keys]
            row.append(float(metric_vals[completed.index(t)]))
            data_matrix.append(row)

        arr = np.array(data_matrix)
        if arr.shape[0] < 2:
            continue
        corr = np.corrcoef(arr, rowvar=False)
        corr_results[mk] = corr[-1, :-1]

    if not corr_results:
        st.info("数据不足以计算相关性。")
        return

    first_mk = valid_metrics[0]
    sorted_pairs = sorted(
        zip(numeric_keys, corr_results[first_mk]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    param_names = [p[0] for p in sorted_pairs]

    fig = go.Figure()
    for mk in valid_metrics:
        corr_vals = [corr_results[mk][numeric_keys.index(p)] for p in param_names]
        fig.add_trace(
            go.Bar(
                name=_METRIC_LABELS[mk],
                x=corr_vals,
                y=param_names,
                orientation="h",
                marker_color=_METRIC_COLORS[mk],
                text=[f"{v:+.3f}" for v in corr_vals],
                textposition="outside",
                textfont=dict(size=9),
            )
        )

    n_groups = len(param_names)
    fig.update_layout(
        title="参数相关性",
        xaxis_title="相关系数",
        xaxis_range=[-1.3, 1.3],
        yaxis_title="参数",
        height=680,
        margin=dict(l=150, r=60, t=50, b=40),
        yaxis=dict(autorange="reversed"),
        barmode="group",
        bargap=0.3,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    st.plotly_chart(fig, use_container_width=True)


def _plot_categorical_effect(study, cat_key):
    completed = _get_completed_trials(study, min_count=3)
    if len(completed) < 3:
        st.info("试验数据不足。")
        return

    cat_labels = {
        "score_mode": "Score Mode",
        "action_strategy": "Action Strategy",
        "mode": "Mode",
        "enable_backup": "Backup",
    }
    label = cat_labels.get(cat_key, cat_key)

    groups = {}
    for t in completed:
        val = t.params.get(cat_key)
        if val is None:
            continue
        display = (
            _ACTION_STRATEGY_LABELS.get(val, val)
            if cat_key == "action_strategy"
            else val
        )
        metrics = _get_trial_metrics(t)
        groups.setdefault(display, []).append(metrics)

    if len(groups) < 2:
        st.info(f"{label} 只有单一取值。")
        return

    valid_metrics = [
        mk
        for mk in _METRIC_KEYS
        if all(g[0].get(mk) is not None for g in groups.values() if g)
    ]
    if not valid_metrics:
        st.info("指标数据不完整。")
        return

    sorted_groups = sorted(
        groups.items(),
        key=lambda x: np.mean([m.get("objective", 0) or 0 for m in x[1]]),
        reverse=True,
    )
    names = [name for name, _ in sorted_groups]

    has_avg_score = "avg_score" in valid_metrics
    if has_avg_score:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.55, 0.45],
        )
    else:
        fig = go.Figure()

    for mk in valid_metrics:
        means = [np.mean([m[mk] for m in grp]) for _, grp in sorted_groups]
        stds = [np.std([m[mk] for m in grp]) for _, grp in sorted_groups]

        text_labels = []
        for v in means:
            if mk == "win_rate":
                text_labels.append(f"{v:.1%}")
            elif mk == "avg_score":
                text_labels.append(f"{v:.1f}")
            else:
                text_labels.append(f"{v:.3f}")

        trace = go.Bar(
            name=_METRIC_LABELS[mk],
            x=names,
            y=means,
            error_y=dict(type="data", array=stds, visible=True),
            marker_color=_METRIC_COLORS[mk],
            text=text_labels,
            textposition="outside",
            textfont=dict(size=9),
            legendgroup=_METRIC_LABELS[mk],
            showlegend=True,
        )
        if has_avg_score:
            row = 2 if mk == "avg_score" else 1
            fig.add_trace(trace, row=row, col=1)
        else:
            fig.add_trace(trace)

    layout_kwargs = dict(
        title=f"{label}",
        height=340,
        margin=dict(l=60, r=30, t=40, b=30),
        xaxis_tickangle=-25,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if has_avg_score:
        fig.update_yaxes(title_text="Obj / WR / Stability", row=1, col=1)
        fig.update_yaxes(title_text="Avg Score", row=2, col=1)
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(tickangle=-25, row=2, col=1)
    fig.update_layout(**layout_kwargs)
    st.plotly_chart(fig, use_container_width=True)


def _plot_parallel_coordinates(study):
    if not study:
        return

    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if len(completed) < 3:
        st.info("试验数据不足。")
        return

    all_keys = []
    for t in completed:
        for k in t.params:
            if k not in all_keys:
                all_keys.append(k)

    numeric_keys = [
        k
        for k in all_keys
        if isinstance(completed[0].params.get(k), (int, float))
        and all(t.params.get(k) is not None for t in completed)
    ]

    categorical_keys = [
        k
        for k in all_keys
        if isinstance(completed[0].params.get(k), str)
        and all(t.params.get(k) is not None for t in completed)
    ]

    cat_display_order = ["mode", "score_mode", "action_strategy", "enable_backup"]
    categorical_keys = [k for k in cat_display_order if k in categorical_keys]

    if not numeric_keys and not categorical_keys:
        st.info("无可显示的参数。")
        return

    dimensions = []

    for key in categorical_keys:
        unique_vals = sorted(set(t.params[key] for t in completed))
        val_to_int = {v: i for i, v in enumerate(unique_vals)}
        display_vals = [
            _ACTION_STRATEGY_LABELS.get(v, v) if key == "action_strategy" else v
            for v in unique_vals
        ]
        dimensions.append(
            dict(
                label=key,
                values=[val_to_int[t.params[key]] for t in completed],
                range=[-0.5, len(unique_vals) - 0.5],
                ticktext=display_vals,
                tickvals=list(range(len(unique_vals))),
            )
        )

    for key in numeric_keys:
        vals = [t.params[key] for t in completed]
        v_min, v_max = min(vals), max(vals)
        padding = (v_max - v_min) * 0.05 if v_max > v_min else 0.5
        dimensions.append(
            dict(
                label=key,
                values=vals,
                range=[v_min - padding, v_max + padding],
            )
        )

    fig = go.Figure(
        go.Parcoords(
            line=dict(
                color=[t.value for t in completed],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Objective"),
            ),
            dimensions=dimensions,
        )
    )
    fig.update_layout(
        height=400,
        margin=dict(l=80, r=50, t=50, b=40),
        font=dict(color="#333", size=14),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_best_params(study, summary):
    best = None
    if study:
        try:
            best = study.best_trial
        except (ValueError, KeyError):
            best = None
    if best:
        st.subheader("最优参数")
        cols = st.columns(3)
        col_idx = 0
        for k, v in best.params.items():
            label = _ACTION_STRATEGY_LABELS.get(v, v) if isinstance(v, str) else v
            with cols[col_idx % 3]:
                st.metric(k, label)
            col_idx += 1
        st.divider()

        wr = best.user_attrs.get("win_rate", 0)
        sc = best.user_attrs.get("avg_score", 0)
        stab = best.user_attrs.get("stability", 0)
        c1, c2, c3 = st.columns(3)
        c1.metric("胜率", f"{wr:.1%}")
        c2.metric("平均得分", f"{sc:.1f}")
        c3.metric("稳定性 (越低越好)", f"{stab:.4f}")
    elif summary:
        st.subheader("最优参数 (摘要)")
        for k, v in summary.get("best_params", {}).items():
            label = _ACTION_STRATEGY_LABELS.get(v, v) if isinstance(v, str) else v
            st.write(f"**{k}**: {label}")


_PAGE_SIZE_TABLE = 100
_PAGE_SIZE_MONITOR = 50

_CATEGORICAL_PARAM_MAP = {
    "score_mode": {0: "quality", 1: "future_reward", 2: "win_rate"},
    "action_strategy": {
        0: "best_beam",
        1: "best_subtree_quality",
        2: "best_subtree_winrate",
        3: "highest_transition_prob",
        4: "random_beam",
        5: "epsilon_greedy",
    },
    "mode": {0: "single_step", 1: "multi_step"},
    "enable_backup": {0: False, 1: True},
}


@st.cache_data(ttl=120, show_spinner=False)
def _load_trials_from_db():
    if not _STUDY_DB.exists():
        return []
    import sqlite3 as _sqlite

    try:
        db = _sqlite.connect(str(_STUDY_DB))
        cur = db.cursor()
        cur.execute("""
            SELECT t.number, tv.value, tp.param_name, tp.param_value,
                   GROUP_CONCAT(tua.key || '||' || tua.value_json, '&&')
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            JOIN trial_params tp ON t.trial_id = tp.trial_id
            LEFT JOIN trial_user_attributes tua ON t.trial_id = tua.trial_id
            WHERE t.state = 'COMPLETE' AND tv.value IS NOT NULL
            GROUP BY t.trial_id
            ORDER BY t.number
        """)
        raw_rows = cur.fetchall()
        db.close()
    except Exception:
        return []

    rows = []
    for r in raw_rows:
        num, val, pname, pval, attrs_str = r
        attrs = {}
        if attrs_str:
            for pair in attrs_str.split("&&"):
                if "||" in pair:
                    k, v = pair.split("||", 1)
                    try:
                        attrs[k] = json.loads(v)
                    except Exception:
                        pass
        rows.append(
            {
                "trial": num,
                "params": {},
                "metrics": {
                    "win_rate": attrs.get("win_rate", 0),
                    "avg_score": attrs.get("avg_score", 0),
                    "stability": attrs.get("stability", 0),
                },
                "objective": val,
                "status": "completed",
                "batch": attrs.get("batch"),
            }
        )

    params_by_trial = {}
    if _STUDY_DB.exists():
        db2 = _sqlite.connect(str(_STUDY_DB))
        cur2 = db2.cursor()
        cur2.execute("""
            SELECT t.number, tp.param_name, tp.param_value
            FROM trials t
            JOIN trial_params tp ON t.trial_id = tp.trial_id
            WHERE t.state = 'COMPLETE'
            ORDER BY t.number
        """)
        for num, pname, pval in cur2.fetchall():
            params_by_trial.setdefault(num, {})[pname] = pval
        db2.close()

    for row in rows:
        row["params"] = params_by_trial.get(row["trial"], {})

    return rows


def _paginate(items, page_key, page_size):
    total = len(items)
    total_pages = max(1, (total + page_size - 1) // page_size)
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    st.session_state[page_key] = min(st.session_state[page_key], total_pages - 1)
    page = st.session_state[page_key]
    goto_key = f"{page_key}_goto"
    if goto_key not in st.session_state:
        st.session_state[goto_key] = page + 1
    start = page * page_size
    end = min(start + page_size, total)
    return page, total_pages, start, end


def _render_pagination(page, total_pages, page_key):
    goto_key = f"{page_key}_goto"

    def _on_prev():
        cur = st.session_state.get(page_key, 0)
        new_page = max(0, cur - 1)
        st.session_state[page_key] = new_page
        st.session_state[goto_key] = new_page + 1

    def _on_next():
        cur = st.session_state.get(page_key, 0)
        new_page = min(total_pages - 1, cur + 1)
        st.session_state[page_key] = new_page
        st.session_state[goto_key] = new_page + 1

    def _on_goto():
        st.session_state[page_key] = st.session_state[goto_key] - 1

    col_prev, col_info, col_next, col_goto = st.columns([1, 2, 1, 1.5])
    with col_prev:
        st.button(
            "◀ 上一页", key=f"{page_key}_prev", disabled=(page <= 0), on_click=_on_prev
        )
    with col_info:
        st.markdown(
            f"<div style='text-align:center;padding-top:4px'>第 {page + 1} / {total_pages} 页</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        st.button(
            "下一页 ▶",
            key=f"{page_key}_next",
            disabled=(page >= total_pages - 1),
            on_click=_on_next,
        )
    with col_goto:
        st.number_input(
            "跳转到页",
            min_value=1,
            max_value=total_pages,
            key=goto_key,
            step=1,
            on_change=_on_goto,
            label_visibility="collapsed",
        )


def _filter_rows(rows, keyword):
    if not keyword.strip():
        return rows
    kw = keyword.strip().lower()
    return [r for r in rows if kw in str(r).lower()]


def _render_trials_table():
    study = _load_study()
    runs = _load_runs()

    all_rows = _load_trials_from_db()
    if not all_rows and runs:
        for run in runs:
            all_rows.append(
                {
                    "trial": run.get("trial", 0),
                    "params": run.get("params", {}),
                    "metrics": run.get("metrics", {}),
                    "objective": run.get("objective"),
                    "status": run.get("status", "?"),
                }
            )

    if not all_rows:
        st.info("暂无试验记录。")
        return

    all_rows.sort(key=lambda x: x.get("trial", 0), reverse=True)

    col_search, col_size = st.columns([2, 1])
    with col_search:
        keyword = st.text_input(
            "Trial 编号", key="table_search", placeholder="输入编号精确查找"
        )
    with col_size:
        _PAGE_SIZE_OPTIONS = [50, 100, 200, 500]
        ps = st.selectbox(
            "每页显示",
            _PAGE_SIZE_OPTIONS,
            index=_PAGE_SIZE_OPTIONS.index(_PAGE_SIZE_TABLE),
            key="table_page_size",
        )

    if keyword.strip().isdigit():
        target = int(keyword.strip())
        filtered = [r for r in all_rows if r.get("trial") == target]
    else:
        filtered = all_rows

    if not filtered:
        st.info("无匹配结果。")
        return

    st.caption(
        f"共 {len(filtered)} 条记录"
        + (f"（已过滤，全部 {len(all_rows)} 条）" if keyword.strip() else "")
    )

    page, total_pages, start, end = _paginate(filtered, "table_page", ps)
    _render_pagination(page, total_pages, "table_page")

    display_rows = []
    for r in filtered[start:end]:
        p = r.get("params", {})
        m = r.get("metrics", {})
        display_rows.append(
            {
                "_trial_num": r.get("trial", 0),
                "批次": r.get("batch", "-") or "-",
                "Trial": f"#{r.get('trial', '?')}",
                "score_mode": p.get("score_mode", ""),
                "beam_width": p.get("beam_width", ""),
                "lookahead": p.get("lookahead_steps", ""),
                "action_strategy": _ACTION_STRATEGY_LABELS.get(
                    p.get("action_strategy", ""), p.get("action_strategy", "")
                ),
                "mode": p.get("mode", ""),
                "min_visits": p.get("min_visits", ""),
                "max_revisits": p.get("max_state_revisits", ""),
                "min_cum_prob": f"{p.get('min_cum_prob', ''):.4f}"
                if isinstance(p.get("min_cum_prob"), float)
                else p.get("min_cum_prob", ""),
                "discount": f"{p.get('discount_factor', ''):.2f}"
                if isinstance(p.get("discount_factor"), float)
                else p.get("discount_factor", ""),
                "backup": "Yes" if p.get("enable_backup") else "No",
                "胜率": f"{m.get('win_rate', 0):.1%}"
                if m.get("win_rate") is not None
                else "-",
                "平均得分": f"{m.get('avg_score', 0):.1f}"
                if m.get("avg_score") is not None
                else "-",
                "稳定性": f"{m.get('stability', 0):.4f}"
                if m.get("stability") is not None
                else "-",
                "目标值": f"{r.get('objective', 0):.4f}"
                if r.get("objective") is not None
                else "-",
            }
        )

    df = pd.DataFrame(display_rows)
    df.insert(0, "选择", False)
    disabled_cols = [c for c in df.columns if c != "选择"]

    edited = st.data_editor(
        df,
        use_container_width=True,
        height=min(len(display_rows) * 35 + 50, 600),
        disabled=disabled_cols,
        hide_index=True,
        key="table_editor",
    )

    selected = edited[edited["选择"] == True]
    selected_trials = []
    if len(selected) > 0:
        for _, row in selected.iterrows():
            trial_str = str(row.get("Trial", ""))
            num_str = trial_str.replace("#", "")
            if num_str.isdigit():
                selected_trials.append(int(num_str))

    if selected_trials:
        st.caption(
            f"已选中 {len(selected_trials)} 条: {', '.join(f'#{t}' for t in selected_trials)}"
        )
        col_rerun, col_del, col_spacer = st.columns([1, 1, 3])
        with col_rerun:
            if st.button("重跑选中", key="table_rerun_selected", type="primary"):
                _start_rerun(selected_trials)
        with col_del:
            if st.button("删除选中", key="table_del_selected"):
                for tn in selected_trials:
                    _delete_trial(tn)
                st.toast(f"已删除 {len(selected_trials)} 条", icon="🗑️")
                _clear_all_cache()
                st.rerun()


@st.cache_data(ttl=30, show_spinner=False)
def _load_runs():
    runs = []
    if not _RUNS_DIR.exists():
        return runs
    for fp in sorted(_RUNS_DIR.glob("trial_*_run.json")):
        try:
            with open(str(fp), "r", encoding="utf-8") as f:
                runs.append(json.load(f))
        except Exception:
            continue
    return runs


def _check_port_alive(port):
    try:
        r = _requests.get(f"http://127.0.0.1:{port}/game/status", timeout=0.5)
        return r.status_code == 200
    except Exception:
        return False


def _read_trial_progress(trial_num, target_episodes):
    progress_file = _TRIALS_DIR / f"trial_{trial_num:04d}" / "progress.json"
    if not progress_file.exists():
        return 0
    try:
        data = json.loads(progress_file.read_text(encoding="utf-8"))
        return data.get("completed", 0)
    except (json.JSONDecodeError, OSError):
        return 0


def _render_run_monitor():
    runs = _load_runs()

    if not runs:
        st.info("暂无运行记录。启动参数寻优后将自动记录。")
        return

    runs.reverse()

    col_search, col_size = st.columns([2, 1])
    with col_search:
        keyword = st.text_input(
            "Trial 编号", key="monitor_search", placeholder="输入编号精确查找"
        )
    with col_size:
        _PAGE_SIZE_OPTIONS = [50, 100, 200, 500]
        ps = st.selectbox(
            "每页显示",
            _PAGE_SIZE_OPTIONS,
            index=_PAGE_SIZE_OPTIONS.index(_PAGE_SIZE_MONITOR),
            key="monitor_page_size",
        )

    if keyword.strip().isdigit():
        target = int(keyword.strip())
        filtered = [r for r in runs if r.get("trial") == target]
    else:
        filtered = runs

    if not filtered:
        st.info("无匹配结果。")
        return

    st.caption(
        f"共 {len(filtered)} 条记录"
        + (f"（已过滤，全部 {len(runs)} 条）" if keyword.strip() else "")
    )

    page, total_pages, start, end = _paginate(filtered, "monitor_page", ps)
    _render_pagination(page, total_pages, "monitor_page")

    header_cols = st.columns([0.5, 1, 1, 1.5, 2.5, 1.5, 1, 1, 0.6, 0.6])
    header_labels = [
        "批次",
        "Trial",
        "端口",
        "时间",
        "参数",
        "状态",
        "指标",
        "备注",
        "重跑",
        "删除",
    ]
    for col, label in zip(header_cols, header_labels):
        col.caption(f"**{label}**")

    for run in filtered[start:end]:
        trial_num = run.get("trial", "?")
        p = run.get("params", {})
        status = run.get("status", "unknown")
        port = run.get("port", 0)
        target_episodes = run.get("target_episodes", 0)
        batch = run.get("batch", "-") or "-"

        if status == "running":
            if isinstance(trial_num, int):
                done = _read_trial_progress(trial_num, target_episodes)
                if done > 0:
                    status_display = "🟢 运行中"
                    progress_info = f" {done}/{target_episodes}"
                else:
                    alive = _check_port_alive(port)
                    status_display = "🟢 运行中" if alive else "🔴 已停止"
                    progress_info = ""
            else:
                status_display = "🟢 运行中"
                progress_info = ""
        else:
            progress_info = ""
            if status == "timeout":
                status_display = "🟡 超时"
            elif status == "completed":
                status_display = "✅ 已完成"
            else:
                status_display = "❓ 未知"

        param_summary = " ".join(
            f"{k}={v}"
            for k, v in p.items()
            if k in ("beam_width", "score_mode", "action_strategy", "lookahead_steps")
        )

        metrics = run.get("metrics", {})
        obj_val = run.get("objective", None)
        if metrics:
            metric_str = f"胜率 {metrics.get('win_rate', 0):.0%}"
            if obj_val is not None:
                metric_str += f" | {obj_val:.4f}"
        else:
            metric_str = "-"

        cols = st.columns([0.5, 1, 1, 1.5, 2.5, 1.5, 1, 1, 0.6, 0.6])
        cols[0].caption(str(batch))
        cols[1].caption(f"#{trial_num}")
        cols[2].caption(str(port))
        cols[3].caption(run.get("start_time", ""))
        cols[4].caption(param_summary)
        if status == "running" and progress_info:
            parts = progress_info.strip().split("/")
            done_val = int(parts[0]) if parts[0].isdigit() else 0
            pct = done_val / target_episodes if target_episodes > 0 else 0
            cols[5].caption(status_display)
            cols[5].progress(min(pct, 1.0))
            cols[5].caption(progress_info)
        else:
            cols[5].caption(status_display)
        cols[6].caption(metric_str)
        source = run.get("source_trial")
        cols[7].caption(f"重跑自 #{source}" if source else "")
        rerun_key = f"_rerun_trial_{trial_num}_{page}"
        if cols[8].button("▶", key=rerun_key):
            if isinstance(trial_num, int):
                _start_rerun([trial_num])
        del_key = f"_del_trial_{trial_num}_{page}"
        if cols[9].button("🗑", key=del_key):
            if isinstance(trial_num, int) and status == "running":
                _kill_port_process(port)
            if isinstance(trial_num, int):
                _delete_trial(trial_num)
            st.toast(f"Trial #{trial_num} 已删除", icon="🗑️")
            st.rerun()

    st.divider()

    learner_log = _TRIALS_DIR / "learner.log"
    if learner_log.exists():
        if st.button("查看 learner.log", key="monitor_learner_log"):
            _show_log("trials/learner.log")

    ep_options = []
    for run in runs:
        tn = run.get("trial")
        if isinstance(tn, int):
            ep_file = _TRIALS_DIR / f"trial_{tn:04d}" / "episodes.jsonl"
            if ep_file.exists():
                ep_options.append(tn)
    if ep_options:
        selected_tn = st.selectbox(
            "查看 Trial Episodes",
            ep_options,
            key="monitor_ep_select",
            format_func=lambda x: f"Trial #{x:04d}",
        )
        if selected_tn is not None:
            ep_path = _TRIALS_DIR / f"trial_{selected_tn:04d}" / "episodes.jsonl"
            try:
                content = ep_path.read_text(encoding="utf-8", errors="replace")
                lines = content.strip().split("\n")
                show_lines = "\n".join(lines[-50:]) if len(lines) > 50 else content
                st.text_area(
                    f"episodes.jsonl (最近 50 行)",
                    show_lines,
                    height=300,
                    label_visibility="collapsed",
                )
            except Exception as e:
                st.error(f"读取失败: {e}")
    if not learner_log.exists() and not ep_options:
        st.caption("暂无日志文件。")


def _render_learner_tab():
    st.markdown("### 基于 Optuna 贝叶斯优化的 Beam Search 参数自动寻优")

    _migrate_batches()

    summary = _load_summary()
    study = _load_study()
    running_trial = _get_running_trial(study)

    if study:
        active_trials = [
            t
            for t in study.trials
            if not (
                t.state == optuna.trial.TrialState.RUNNING and not _is_learner_alive()
            )
        ]
        total = len(active_trials)
        completed = sum(
            1 for t in active_trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        running = sum(
            1 for t in active_trials if t.state == optuna.trial.TrialState.RUNNING
        )
        failed = sum(
            1 for t in active_trials if t.state == optuna.trial.TrialState.FAIL
        )

        if running:
            st.markdown(
                f"**{total} 轮试验 | {completed} 完成 |** :orange[{running} 运行中] **| {failed} 失败**"
            )
        else:
            st.markdown(f"**{total} 轮试验 | {completed} 完成 | {failed} 失败**")
    elif summary:
        st.markdown(
            f"**{summary.get('total_trials', 0)} 轮试验 | {summary.get('completed_trials', 0)} 完成**"
        )
    else:
        st.info("暂无优化数据。在侧边栏配置参数后点击「一键启动参数寻优」。")

    if study and len(study.trials) > 0:
        batch_info = _get_batch_info()

        with st.expander("批次管理", expanded=False):
            if batch_info:
                selected_batches = []
                for batch_num in sorted(batch_info.keys()):
                    info = batch_info[batch_num]
                    checked = st.checkbox(
                        f"批次 {batch_num}: {info['count']} 条 "
                        f"(Trial #{info['min_trial']} - #{info['max_trial']})",
                        key=f"_batch_sel_{batch_num}",
                    )
                    if checked:
                        selected_batches.append(batch_num)

                if selected_batches:
                    st.caption(f"已选中 {len(selected_batches)} 个批次")
                    confirmed = st.checkbox(
                        f"确认删除选中的 {len(selected_batches)} 个批次",
                        key="_batch_delete_confirm",
                    )
                    if st.button(
                        f"删除选中的 {len(selected_batches)} 个批次",
                        disabled=not confirmed,
                        type="primary" if confirmed else "secondary",
                        key="_batch_delete_btn",
                    ):
                        total = 0
                        for bn in selected_batches:
                            info = batch_info[bn]
                            total += info["count"]
                            _delete_batch(bn)
                        st.toast(
                            f"已删除 {len(selected_batches)} 个批次（{total} 条）",
                            icon="🗑️",
                        )
                        st.rerun()
            else:
                st.info("暂无批次信息。")

        col_reset, col_grefresh, col_export = st.columns([3, 1, 1])
        with col_reset:
            if st.button("重置数据库（清除所有试验记录）", key="learner_reset_db"):
                try:
                    optuna.delete_study(
                        study_name="beam_search", storage=f"sqlite:///{_STUDY_DB}"
                    )
                    for f in _RUNS_DIR.glob("trial_*_run.json"):
                        f.unlink(missing_ok=True)
                    for d in _TRIALS_DIR.glob("trial_*"):
                        if d.is_dir():
                            shutil.rmtree(d, ignore_errors=True)
                    if _SUMMARY_PATH.exists():
                        _SUMMARY_PATH.unlink()
                    st.toast("数据库已重置", icon="🗑️")
                    _load_study.clear()
                    _compute_importance.clear()
                    _load_trials_from_db.clear()
                    _load_runs.clear()
                    _get_batch_info.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"重置失败: {e}")
        with col_grefresh:
            if st.button("刷新状态", key="learner_refresh_status"):
                _load_study.clear()
                _load_runs.clear()
                _load_trials_from_db.clear()
                _compute_importance.clear()
                _get_batch_info.clear()
                st.rerun()

    _render_best_params(study, summary)

    _VIEW_OPTIONS = [
        "优化曲线",
        "参数分析",
        "参数关系",
        "试验记录",
        "运行状态监测",
    ]
    _VIEW_KEY = "_learner_active_view"
    if _VIEW_KEY not in st.session_state:
        st.session_state[_VIEW_KEY] = _VIEW_OPTIONS[0]

    active_view = st.radio(
        "选择视图",
        _VIEW_OPTIONS,
        index=_VIEW_OPTIONS.index(st.session_state[_VIEW_KEY]),
        horizontal=True,
        key=_VIEW_KEY + "_radio",
        label_visibility="collapsed",
    )
    st.session_state[_VIEW_KEY] = active_view

    if active_view == "优化曲线":
        _plot_objective_history(study)

    elif active_view == "参数分析":
        col_left, col_right = st.columns([3, 2])
        with col_left:
            left_l, left_r = st.columns(2)
            with left_l:
                _plot_importance(study)
            with left_r:
                _plot_numeric_correlation(study)
        with col_right:
            cat_row1_a, cat_row1_b = st.columns(2)
            with cat_row1_a:
                _plot_categorical_effect(study, "mode")
            with cat_row1_b:
                _plot_categorical_effect(study, "score_mode")
            cat_row2_a, cat_row2_b = st.columns(2)
            with cat_row2_a:
                _plot_categorical_effect(study, "action_strategy")
            with cat_row2_b:
                _plot_categorical_effect(study, "enable_backup")

    elif active_view == "参数关系":
        _plot_parallel_coordinates(study)

    elif active_view == "试验记录":
        _render_trials_table()

    elif active_view == "运行状态监测":
        _render_run_monitor()
