# 参数寻优架构改进方案

## 问题分析

### 当前架构
每个 trial 启动一次 SC2 进程（`Popen run_live_game.py`），跑完后 terminate → 重启。
`_wait_for_completion` 通过 HTTP 轮询 `_history_store` 的 `len()` 判断完成。

### 核心问题
1. **每轮重启 SC2 开销大** — 每个 trial 都要启动/关闭整个游戏进程
2. **`_history_store` FIFO 上限 100** — 可能导致轮询检测不准确
3. **`beam_params` 在 Agent 初始化时绑定** — `kg_guided_agent` 通过 `_local_decide()` 本地决策，不调用 `bridge.get_action()`，运行时无法更改参数，每次换参数必须重启 Agent
4. **日志路径错误** — `trial_XXXX.log` 应在 `trials/` 而非 `runs/`

## 改进方案：单次启动 SC2 + API 参数热切换

### 数据流
```
parameter_learner (一次启动)
  ├── _startup(): Popen run_live_game.py --max_episodes 0 (无限循环)
  ├── 循环 trial 1..N:
  │   ├── POST /game/beam_params {新参数}
  │   │     → bridge.param_update_queue → kg_guided_agent.step() 更新 _beam_params
  │   ├── POST /game/episodes/clear {max_history: target}
  │   │     → 清空 _history_store + 动态调整 FIFO 容量
  │   ├── GET /game/status → 记录 total_completed 起始值
  │   ├── 轮询 GET /game/status → total_completed - start >= target
  │   ├── POST /game/results/save → 保存结果
  │   └── 分析结果 → optuna
  └── _shutdown(): POST /game/shutdown
```

### 文件改动清单

| 文件 | 改动 |
|------|------|
| `src/sc2env/bridge.py` | 新增 `param_update_queue = Queue(maxsize=1)` |
| `src/sc2env/kg_guided_agent.py` | `step()` 开头检查 `param_update_queue`，更新 `self._beam_params` |
| `src/sc2env/bridge_server.py` | 新增 `/game/beam_params` 端点；新增 `_total_completed` 计数器（不清零）；`/game/episodes/clear` 接受 `max_history` 参数动态调整 FIFO 容量 |
| `scripts/parameter_learner.py` | 架构重写：`_startup()` 启动一次 SC2；`_objective()` 通过 API 切换参数+等待+保存；日志改到 `trials/` |
| `scripts/kg_web/learner_tab.py` | 日志路径改到 `trials/`；进度显示（`st.progress + 50/100`）；`run_record` 新增 `target_episodes` |

### 详细设计

#### 1. `bridge.py` — 新增参数更新队列
```python
self.param_update_queue: Queue = Queue(maxsize=1)
```
覆写式设计（maxsize=1），最新参数立即生效，旧参数丢弃。

#### 2. `kg_guided_agent.py` — 运行时参数更新
在 `step()` 开头（super().step() 之后）检查队列：
```python
try:
    while True:
        new_params = self.bridge.param_update_queue.get_nowait()
        self._beam_params.update(new_params)
except Exception:
    pass
```
开销极小（queue 空时 get_nowait 立即抛异常）。

#### 3. `bridge_server.py`
- **`_total_completed`**：在 `_drain_history()` 中每消费一个 episode 就 +1，永不归零
- **`/game/beam_params`**：POST 端点，写入 `bridge.param_update_queue`
- **`/game/episodes/clear`**：接受 `max_history` 参数，动态调整 `_history_max_episodes`

#### 4. `parameter_learner.py` — 架构重写
- `_startup()`：启动一次 `run_live_game.py --max_episodes 0`，等待服务器就绪
- `_objective()`：设置参数 → 清空历史+设置FIFO → 记录起始计数 → 轮询等待 → 保存结果 → 分析
- `_shutdown()`：`POST /game/shutdown` + terminate 进程
- 日志：`trials/trial_XXXX.log`（不再写入 `runs/`）

#### 5. `learner_tab.py`
- 新增 `_TRIALS_DIR = _RESULTS_DIR / "trials"`
- 查看日志从 `_TRIALS_DIR` 读取
- `run_record` 新增 `"target_episodes"` 字段
- `_render_run_monitor()` 对 running trial 轮询进度：`st.progress + 文字 50/100`
