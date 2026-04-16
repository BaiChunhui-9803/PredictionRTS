import sys
import subprocess
import time
import requests
import streamlit as st
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import ROOT_DIR
from kg_web.constants import BRIDGE_API_URL, _ACTION_STRATEGY_LABELS
from kg_web.live_game_html import _build_live_game_html


def _render_live_game_sidebar(kg_entry: Optional[Dict] = None):
    kg_file = kg_entry.get("file", "") if kg_entry else ""
    kg_name = kg_entry.get("name", "") if kg_entry else ""
    kg_data_dir = kg_entry.get("data_dir", "") if kg_entry else ""
    if kg_file:
        st.caption(f"KG: {kg_name}")
        if kg_data_dir:
            st.caption(f"路径: {kg_data_dir}/bktree/")
            import json, glob as _glob

            _bkt_dir = ROOT_DIR / kg_data_dir / "bktree"
            _pri = _bkt_dir / "primary_bktree.json"
            _pri_cnt = 0
            if _pri.exists():
                try:
                    _d = json.load(open(str(_pri), "r"))
                    _stk = [_d]
                    while _stk:
                        _n = _stk.pop()
                        _pri_cnt += 1
                        _stk.extend(_n.get("children", {}).values())
                except Exception:
                    pass
            _sec_files = sorted(_glob.glob(str(_bkt_dir / "secondary_bktree_*.json")))
            _sec_cnt = len(_sec_files)
            _sn_path = ROOT_DIR / kg_data_dir / "graph" / "state_node.txt"
            _sn_cnt = 0
            if _sn_path.exists():
                try:
                    _sn_cnt = sum(1 for _ in open(str(_sn_path), "r") if _.strip())
                except Exception:
                    pass
            with st.expander("BKTree 详情", expanded=False):
                st.caption(f"Primary 节点: {_pri_cnt}")
                st.caption(f"Secondary 树: {_sec_cnt}")
                st.caption(f"State 映射: {_sn_cnt} 条")
                st.caption(
                    f"map_id: {kg_entry.get('map_id', '-')} | data_id: {kg_entry.get('data_id', '-')}"
                )

    col_port, col_btn = st.columns([3, 2])
    with col_port:
        port = st.number_input(
            "API 端口",
            min_value=1024,
            max_value=65535,
            value=8000,
            key="live_port",
        )
    _start_clicked = False
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            "一键启动", type="primary", use_container_width=True, key="live_start"
        ):
            _start_clicked = True

    st.caption("窗口位置")
    wc1, wc2, wc3, wc4 = st.columns(4)
    with wc1:
        st.number_input("X", value=2600, key="live_wx", label_visibility="collapsed")
    with wc2:
        st.number_input("Y", value=50, key="live_wy", label_visibility="collapsed")
    with wc3:
        st.number_input("W", value=640, key="live_ww", label_visibility="collapsed")
    with wc4:
        st.number_input("H", value=480, key="live_wh", label_visibility="collapsed")

    ap_port = int(st.session_state.get("live_port", 8000))
    api_base = f"http://localhost:{ap_port}"

    st.divider()
    start_paused = st.toggle(
        "启动后暂停",
        value=False,
        key="live_start_paused",
        help="开启后服务启动时自动进入暂停状态，需手动恢复",
    )

    st.subheader("Beam Search 参数")

    c1, c2 = st.columns(2)
    with c1:
        st.selectbox(
            "评分策略",
            options=["quality", "future_reward", "win_rate"],
            format_func={
                "quality": "Quality Score",
                "future_reward": "Future Reward",
                "win_rate": "Win Rate",
            }.get,
            key="live_sm",
        )
    with c2:
        st.slider("Beam Width", 1, 10, 3, key="live_bw")

    c3, c4 = st.columns(2)
    with c3:
        st.slider("前瞻步数", 1, 15, 5, key="live_la")
    with c4:
        st.slider("最低访问次数", 1, 10, 1, key="live_mv")

    c5, c6 = st.columns(2)
    with c5:
        st.slider(
            "最大状态重复",
            1,
            5,
            2,
            key="live_msr",
            help="同一条推演路径中同一状态最多出现的次数。",
        )
    with c6:
        st.slider(
            "累积概率阈值",
            0.001,
            0.1,
            0.01,
            step=0.001,
            format="%.3f",
            key="live_mcp",
        )

    c7, c8 = st.columns(2)
    with c7:
        st.slider(
            "折扣因子",
            0.5,
            1.0,
            0.9,
            step=0.05,
            format="%.2f",
            key="live_df",
            help="每步累积概率乘以此值的步数次幂。1.0=无折扣。",
        )
    with c8:
        st.selectbox(
            "动作选择",
            options=list(_ACTION_STRATEGY_LABELS.keys()),
            format_func=lambda x: _ACTION_STRATEGY_LABELS[x],
            key="live_as",
        )

    c9, c10 = st.columns(2)
    with c9:
        st.slider(
            "ε",
            0.01,
            0.5,
            0.1,
            step=0.01,
            format="%.2f",
            key="live_eps",
            disabled=(st.session_state.get("live_as") != "epsilon_greedy"),
        )
    with c10:
        st.slider("最大推演步数", 1, 100, 50, key="live_mrs")

    st.markdown("**推演模式**")
    live_mode = st.radio(
        "mode",
        options=["单步推演", "多步推演"],
        index=0,
        horizontal=True,
        key="live_mode",
        label_visibility="collapsed",
    )

    live_backup = st.toggle("启用备选路径", value=False, key="live_backup")
    if live_backup:
        bc1, bc2 = st.columns(2)
        with bc1:
            st.slider(
                "备选评分阈值",
                0.0,
                1.0,
                0.3,
                step=0.05,
                format="%.2f",
                key="live_backup_st",
            )
        with bc2:
            st.slider(
                "模糊匹配距离阈值",
                0.0,
                1.0,
                0.2,
                step=0.05,
                format="%.2f",
                key="live_backup_dt",
            )

    if _start_clicked:
        if "live_proc" in st.session_state and st.session_state.live_proc is not None:
            proc = st.session_state.live_proc
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
        cmd = [
            sys.executable,
            str(ROOT_DIR / "scripts" / "run_live_game.py"),
            "--mode",
            "all",
            "--port",
            str(port),
        ]
        if kg_file:
            cmd.extend(["--kg_file", kg_file])
        if kg_data_dir:
            cmd.extend(["--data_dir", kg_data_dir])
        cmd.extend(
            [
                "--window_x",
                str(st.session_state.get("live_wx", 50)),
                "--window_y",
                str(st.session_state.get("live_wy", 50)),
                "--window_w",
                str(st.session_state.get("live_ww", 640)),
                "--window_h",
                str(st.session_state.get("live_wh", 480)),
            ]
        )
        p = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32"
            else 0,
        )
        st.session_state.live_proc = p
        _beam_params = {
            "beam_width": st.session_state.get("live_bw", 3),
            "max_steps": st.session_state.get("live_mrs", 50),
            "score_mode": st.session_state.get("live_sm", "quality"),
            "min_visits": st.session_state.get("live_mv", 1),
            "min_cum_prob": st.session_state.get("live_mcp", 0.01),
            "max_state_revisits": st.session_state.get("live_msr", 2),
            "discount_factor": st.session_state.get("live_df", 0.9),
        }
        _ap_payload = {
            "enabled": True,
            "mode": "single_step" if live_mode == "单步推演" else "multi_step",
            "lookahead_steps": st.session_state.get("live_la", 5),
            "action_strategy": st.session_state.get("live_as", "best_beam"),
            "score_mode": st.session_state.get("live_sm", "quality"),
            "beam_width": st.session_state.get("live_bw", 3),
            "max_steps": st.session_state.get("live_mrs", 50),
            "min_visits": st.session_state.get("live_mv", 1),
            "min_cum_prob": st.session_state.get("live_mcp", 0.01),
            "max_state_revisits": st.session_state.get("live_msr", 2),
            "discount_factor": st.session_state.get("live_df", 0.9),
            "enable_backup": live_backup,
            "epsilon": st.session_state.get("live_eps", 0.1),
        }
        with st.spinner("等待服务启动..."):
            for _ in range(30):
                time.sleep(0.5)
                try:
                    r = requests.get(f"{api_base}/game/status", timeout=2)
                    if r.status_code == 200:
                        requests.post(
                            f"{api_base}/game/beam_params",
                            json=_beam_params,
                            timeout=5,
                        )
                        requests.post(
                            f"{api_base}/game/autopilot",
                            json=_ap_payload,
                            timeout=5,
                        )
                        if start_paused:
                            requests.post(
                                f"{api_base}/game/control",
                                json={"command": "pause"},
                                timeout=5,
                            )
                        st.success(
                            "服务已启动，自动决策已开启"
                            + (" (已暂停)" if start_paused else "")
                        )
                        break
                except Exception:
                    continue
            else:
                st.warning("服务启动超时，请检查日志")

    try:
        r = requests.get(f"{api_base}/game/autopilot/status", timeout=2)
        if r.status_code == 200:
            data = r.json()
            if data.get("enabled"):
                stats = data.get("stats", {})
                st.markdown(
                    f"**运行中** | {data.get('mode', '')} | "
                    f"决策:{stats.get('total_decisions', 0)} | "
                    f"重规划:{stats.get('total_replans', 0)} | "
                    f"偏离:{stats.get('total_divergences', 0)} | "
                    f"{data.get('plan_progress', '0/0')}"
                )
            else:
                st.caption("待机")
    except Exception:
        pass


def _render_live_game_content():
    port = int(st.session_state.get("live_port", 8000))
    html = _build_live_game_html(port)
    st.components.v1.html(html, height=950, scrolling=False)
