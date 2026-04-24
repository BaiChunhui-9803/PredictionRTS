# 参数寻优架构修复计划 V3

## 背景

Episode 计数严重不准确（实际跑了很多局但计数只有 12/100），根因是 `_ep_batch` 批量 flush 机制 + `run_game.py` 的 `continue` 跳过 `obs.last()` 路径。

## 方案

修复现有架构，保持单次启动 SC2，结果收集走本地文件，不依赖 HTTP 轮询。

## 改动清单

### 1. `src/sc2env/kg_guided_agent.py`

**1.1 Bug 修复：无条件 flush**
- 行 757-758：去掉 `_ep_batch >= _ep_push_batch_size` 条件，`end_game_flag` 触发时无条件 `_flush_ep_batch()`

**1.2 新增本地结果收集**
- `__init__` 新增 `local_result_dir: Optional[str] = None` 参数
- `new_game()` 末尾：如果 `self._local_result_dir` 存在，将 episode 记录追加写入 `episodes.jsonl`（JSON Lines），同时更新 `progress.json`（`{"completed": N}`）
- 进度每 10 个 episode 通过 bridge.put_event() 推送一次（保留）

**1.3 step() 中接受 local_result_dir 热切换**
- `param_update_queue` 检查中，如果 `new_params` 包含 `local_result_dir`，更新 `self._local_result_dir` 并清空 `progress.json`

### 2. `scripts/parameter_learner.py`

**2.1 `_objective()` 重写**
- 启动时为每个 trial 创建 `trials/trial_NNNN/` 目录
- 通过 `/game/beam_params` 传递参数 + `local_result_dir` 路径
- 不再调用 `/game/episodes/clear`（不再依赖 history_store）
- 不再调用 `/game/results/save`（不再从 HTTP 获取结果）
- 等待完成改为轮询 `progress.json` 文件（`completed >= target_episodes`）
- 统计指标改为读 `episodes.jsonl` 文件
- 日志直接 `print()` 输出（已重定向到 `trials/learner.log`）

**2.2 删除不再需要的函数**
- `_clear_history()` — 不再需要
- `_save_results()` — 不再需要
- `_wait_for_trial()` — 改为 `_wait_for_file_progress()`

**2.3 新增函数**
- `_wait_for_file_progress(trial_dir, target, poll_interval, timeout)` — 轮询 `progress.json`
- `_analyze_local_result(trial_dir, num_segments)` — 读 `episodes.jsonl` 计算胜率/得分/稳定性

### 3. `scripts/kg_web/learner_tab.py`

**3.1 运行状态监测改为读文件**
- `_render_run_monitor()` 中对 `status == "running"` 的 trial：
  - 不再发 HTTP 请求轮询 `/game/status`
  - 改为读 `trials/trial_NNNN/progress.json` 获取 `completed` 和 `target_episodes`
  - 进度条直接从文件数据计算

**3.2 日志查看改为读文件**
- 保留 `trials/learner.log` 的查看（整体日志）
- 新增查看 `trials/trial_NNNN/episodes.jsonl` 的选项（单 trial 的 episode 结果）

**3.3 删除不再需要的代码**
- `_check_port_alive()` — 不再需要（不通过 HTTP 判断运行状态）
- `_kill_port_process()` — 保留（停止 trial 时仍需按端口杀进程）

### 4. 不需要改动的文件
- `bridge_server.py` — `/game/beam_params` 端点保留
- `bridge.py` — `param_update_queue` 保留
- `run_live_game.py` — 不需要改
- `run_game.py` — 不需要改

## 文件结构

```
output/learner_results/trials/
├── trial_0000/
│   ├── episodes.jsonl      # 每行: {"episode_id":0,"result":"Win","score":45.0}
│   └── progress.json       # {"completed": 87, "target": 100}
├── trial_0001/
│   ...
└── learner.log             # parameter_learner.py 的 stdout
```

## 数据流

```
parameter_learner.py (_objective)
    → HTTP POST /game/beam_params (传递参数 + local_result_dir)
    → kg_guided_agent.step() 接收参数（param_update_queue）
    → kg_guided_agent.new_game() 写 episodes.jsonl + 更新 progress.json
    → parameter_learner.py 轮询 progress.json 等待完成
    → parameter_learner.py 读 episodes.jsonl 计算指标
    → 写入 run.json + optuna trial
```
