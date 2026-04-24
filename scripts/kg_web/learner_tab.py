import os
import re
import sys
import subprocess
import time
import requests as _requests
from pathlib import Path
from typing import Optional, Dict

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
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


@st.cache_resource(ttl=10, show_spinner=False)
def _load_study():
    if _STUDY_DB.exists():
        try:
            storage = f"sqlite:///{_STUDY_DB}"
            study = optuna.load_study(study_name="beam_search", storage=storage)
            return study
        except Exception:
            pass
    return None


def _get_running_trial(study):
    if not study:
        return None
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            if not _is_learner_alive():
                continue
            return t
    return None


_PID_FILE = _RESULTS_DIR / ".learner_pid"


@st.cache_data(ttl=5, show_spinner=False)
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
        max_value=500,
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
        value=_obj_cfg.get("stability_cap", 2.0),
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


def _plot_importance(study):
    if not study or len(study.trials) < 5:
        st.info("试验数据不足，至少需要 5 轮完成才能计算参数重要性。")
        return

    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception:
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
        height=max(300, len(params) * 35 + 60),
        margin=dict(l=150, r=30, t=50, b=40),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_param_correlation(study):
    if not study or len(study.trials) < 5:
        st.info("试验数据不足，至少需要 5 轮完成才能计算参数相关性。")
        return

    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if len(completed) < 5:
        st.info("完成的试验不足 5 轮。")
        return

    numeric_keys = []
    for t in completed:
        for k, v in t.params.items():
            if isinstance(v, (int, float)) and k not in numeric_keys:
                numeric_keys.append(k)

    if not numeric_keys:
        st.info("无数值型参数。")
        return

    shared_keys = [
        k for k in numeric_keys if all(t.params.get(k) is not None for t in completed)
    ]
    if len(shared_keys) < 2:
        st.info("共享的数值型参数不足 2 个。")
        return

    labels = shared_keys + ["Objective"]
    data_matrix = []
    for t in completed:
        row = [float(t.params[k]) for k in shared_keys]
        row.append(float(t.value))
        data_matrix.append(row)

    arr = np.array(data_matrix)
    if arr.shape[0] < 2:
        st.info("数据不足以计算相关性。")
        return

    corr = np.corrcoef(arr, rowvar=False)
    correlations = corr[-1, :-1]

    sorted_pairs = sorted(
        zip(shared_keys, correlations), key=lambda x: abs(x[1]), reverse=True
    )
    param_names = [p[0] for p in sorted_pairs]
    corr_values = [p[1] for p in sorted_pairs]
    bar_colors = ["#4C72B0" if v >= 0 else "#DD8452" for v in corr_values]
    bar_texts = [f"{v:+.3f}" for v in corr_values]

    fig = go.Figure(
        go.Bar(
            x=corr_values,
            y=param_names,
            orientation="h",
            marker_color=bar_colors,
            text=bar_texts,
            textposition="outside",
            textfont=dict(size=11),
        )
    )
    fig.update_layout(
        title="参数与目标值的相关性（Pearson）",
        xaxis_title="相关系数",
        xaxis_range=[-1, 1],
        yaxis_title="参数",
        height=max(300, len(param_names) * 35 + 60),
        margin=dict(l=150, r=50, t=50, b=40),
        yaxis=dict(autorange="reversed"),
        bargap=0.3,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
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

    numeric_keys = []
    for t in completed:
        for k, v in t.params.items():
            if isinstance(v, (int, float)) and k not in numeric_keys:
                numeric_keys.append(k)

    shared_keys = [
        k for k in numeric_keys if all(t.params.get(k) is not None for t in completed)
    ]
    if not shared_keys:
        st.info("无数值型参数可显示。")
        return

    dimensions = []
    for key in shared_keys:
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
        font=dict(color="#333"),
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


def _render_trials_table():
    runs = _load_runs()
    if not runs:
        st.info("暂无试验记录。")
        return

    rows = []
    for run in reversed(runs):
        p = run.get("params", {})
        m = run.get("metrics", {})
        rows.append(
            {
                "Trial": f"#{run.get('trial', '?')}",
                "状态": run.get("status", "?"),
                "score_mode": p.get("score_mode", ""),
                "beam_width": p.get("beam_width", ""),
                "lookahead": p.get("lookahead_steps", ""),
                "action_strategy": _ACTION_STRATEGY_LABELS.get(
                    p.get("action_strategy", ""), p.get("action_strategy", "")
                ),
                "min_visits": p.get("min_visits", ""),
                "max_revisits": p.get("max_state_revisits", ""),
                "min_cum_prob": f"{p.get('min_cum_prob', ''):.4f}"
                if isinstance(p.get("min_cum_prob"), float)
                else p.get("min_cum_prob", ""),
                "discount": f"{p.get('discount_factor', ''):.2f}"
                if isinstance(p.get("discount_factor"), float)
                else p.get("discount_factor", ""),
                "backup": "Yes" if p.get("enable_backup") else "No",
                "胜率": f"{m.get('win_rate', 0):.1%}" if m else "-",
                "平均得分": f"{m.get('avg_score', 0):.1f}" if m else "-",
                "稳定性": f"{m.get('stability', 0):.4f}" if m else "-",
                "目标值": f"{run.get('objective', 0):.4f}"
                if run.get("objective") is not None
                else "-",
            }
        )

    st.dataframe(rows, use_container_width=True, height=min(len(rows) * 35 + 50, 500))


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

    if st.button("刷新状态", key="monitor_refresh"):
        st.rerun()

    header_cols = st.columns([1, 1, 1.5, 2.5, 1.5, 1, 0.8])
    header_labels = ["Trial", "端口", "时间", "参数", "状态", "指标", "操作"]
    for col, label in zip(header_cols, header_labels):
        col.caption(f"**{label}**")

    for run in reversed(runs):
        trial_num = run.get("trial", "?")
        p = run.get("params", {})
        status = run.get("status", "unknown")
        port = run.get("port", 0)
        target_episodes = run.get("target_episodes", 0)

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

        cols = st.columns([1, 1, 1.5, 2.5, 1.5, 1, 0.8])
        cols[0].caption(f"#{trial_num}")
        cols[1].caption(str(port))
        cols[2].caption(run.get("start_time", ""))
        cols[3].caption(param_summary)
        if status == "running" and progress_info:
            parts = progress_info.strip().split("/")
            done_val = int(parts[0]) if parts[0].isdigit() else 0
            pct = done_val / target_episodes if target_episodes > 0 else 0
            cols[4].caption(status_display)
            cols[4].progress(min(pct, 1.0))
            cols[4].caption(progress_info)
        else:
            cols[4].caption(status_display)
        cols[5].caption(metric_str)
        del_key = f"_del_trial_{trial_num}"
        if cols[6].button("🗑", key=del_key):
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
    for run in reversed(runs):
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
        if st.button("重置数据库（清除所有试验记录）", key="learner_reset_db"):
            try:
                optuna.delete_study(
                    study_name="beam_search", storage=f"sqlite:///{_STUDY_DB}"
                )
                for f in _RUNS_DIR.glob("trial_*_run.json"):
                    f.unlink(missing_ok=True)
                if _SUMMARY_PATH.exists():
                    _SUMMARY_PATH.unlink()
                st.toast("数据库已重置", icon="🗑️")
                _load_study.clear()
                st.rerun()
            except Exception as e:
                st.error(f"重置失败: {e}")

    _render_best_params(study, summary)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["优化曲线", "参数分析", "参数关系", "试验记录", "运行状态监测"]
    )

    with tab1:
        _plot_objective_history(study)

    with tab2:
        col_left, col_right = st.columns(2)
        with col_left:
            _plot_importance(study)
        with col_right:
            _plot_param_correlation(study)

    with tab3:
        _plot_parallel_coordinates(study)

    with tab4:
        _render_trials_table()

    with tab5:
        _render_run_monitor()
