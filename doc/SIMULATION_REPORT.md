# 决策过程模拟报告

## 概述

本报告描述了基于知识图谱的决策过程模拟系统的设计与实验结果。该系统使用状态转移网络和轮盘赌选择算法，从初始状态出发模拟完整的决策轨迹，并输出详细的状态-动作-奖励记录。

---

## 1. 方法设计

### 1.1 状态转移网络

从训练数据中构建状态转移网络：

```
transitions[state][action] = {
    'next_states': {next_state: count, ...},
    'quality_score': float,
    'win_rate': float
}
```

- **状态**：940 个可能的状态 ID（0-939）
- **动作**：55 个可能的动作（5 clusters × 11 action types）
- **转移概率**：基于训练数据中观察到的转移频次计算

### 1.2 动作选择策略：轮盘赌选择

基于质量分数（quality_score）进行概率化选择：

```python
quality_score = avg_future_reward × 0.7 + win_rate × 30.0

# 概率归一化
prob[action] = (quality_score[action] + 0.1) / sum(all_quality_scores + 0.1)

# 轮盘赌选择
r = random()
cumsum = 0
for action in actions:
    cumsum += prob[action]
    if r <= cumsum:
        return action
```

**优点**：
- 高质量动作更可能被选中
- 低质量动作仍有被选中的机会，保证多样性
- 避免贪婪选择的局部最优问题

### 1.3 模拟流程

```
初始化: state = start_state (默认 0)
for step in range(max_steps):
    1. 从知识图谱获取 top-k 高质量动作
    2. 使用轮盘赌选择一个动作
    3. 根据转移网络获取下一状态
    4. 记录 (state, action, next_state, rewards, quality)
    5. state = next_state
返回: 完整轨迹记录
```

---

## 2. 使用方法

### 2.1 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--kg-type` | simple | 知识图谱类型：simple 或 context |
| `--context-window` | 5 | 上下文窗口大小（仅 context 类型） |
| `--episodes` | 100 | 模拟回合数 |
| `--start-state` | 0 | 起始状态 |
| `--max-steps` | 30 | 每回合最大步数 |
| `--top-k` | 5 | 候选动作数量 |
| `--seed` | 42 | 随机种子 |
| `--verbose` | False | 显示每步详细信息 |
| `--detail-log` | False | 输出详细记录到屏幕 |
| `--output-file` | None | 输出详细记录到 JSON 文件 |

### 2.2 使用示例

```bash
# 基本运行（不显示详细日志）
python scripts/simulate_decision_process.py --kg-type simple --episodes 100

# 显示详细状态-动作-奖励记录
python scripts/simulate_decision_process.py --kg-type simple --episodes 10 --detail-log

# 将详细记录保存到文件
python scripts/simulate_decision_process.py --kg-type simple --episodes 100 \
    --output-file output/simulation_log.json

# 同时显示并保存
python scripts/simulate_decision_process.py --kg-type simple --episodes 10 \
    --detail-log --output-file output/simulation_log.json

# 使用不同的起始状态
python scripts/simulate_decision_process.py --kg-type simple --episodes 50 \
    --start-state 100 --detail-log
```

---

## 3. 输出格式

### 3.1 汇总报告

```
============================================================
       SIMULATION RESULTS
============================================================

Knowledge Graph: simple
Episodes: 100
Start State: 0

Metric                                   Value
--------------------------------------------------
Avg Episode Length             30.00 +/- 0.00
Avg Quality Score              48.40
Avg Win Rate                   99.5%

Trajectory Analysis:
--------------------------------------------------

Top 10 Actions:
  1d: 223 (7.4%)
  4d: 213 (7.1%)
  4b: 196 (6.5%)
  ...

Top 10 Visited States:
  State 109: 294 (9.8%)
  State 544: 278 (9.3%)
  State 396: 236 (7.9%)
  ...
```

### 3.2 详细日志（--detail-log）

```
================================================================================
       DETAILED SIMULATION LOG
================================================================================
Ep   Step  State  Action Next   StepRwd    FutRwd     Quality  WinRate
--------------------------------------------------------------------------------
0    0     0      4d     340    0.0000     42.6745    58.42    95.2%
0    1     340    4d     396    -0.0273    41.4806    58.90    99.5%
0    2     396    3f     396    3.0000     60.0000    72.00    100.0%
0    3     396    3f     396    3.0000     60.0000    72.00    100.0%
...
```

**字段说明**：

| 字段 | 说明 |
|------|------|
| Ep | 回合编号 |
| Step | 步骤编号 |
| State | 当前状态 ID |
| Action | 选择的动作（格式：cluster_id + action_letter） |
| Next | 下一状态 ID |
| StepRwd | 平均即时奖励（avg_step_reward） |
| FutRwd | 平均未来累计奖励（avg_future_reward） |
| Quality | 质量分数（quality_score） |
| WinRate | 该状态-动作对的胜率 |

### 3.3 JSON 文件格式（--output-file）

```json
[
  {
    "episode": 0,
    "step": 0,
    "state": 0,
    "action": "4d",
    "next_state": 340,
    "avg_step_reward": 0.0,
    "avg_future_reward": 42.6745,
    "quality_score": 58.42,
    "win_rate": 0.952
  },
  {
    "episode": 0,
    "step": 1,
    "state": 340,
    "action": "4d",
    "next_state": 396,
    "avg_step_reward": -0.0273,
    "avg_future_reward": 41.4806,
    "quality_score": 58.90,
    "win_rate": 0.995
  }
]
```

---

## 4. 实验结果

### 4.1 Simple vs Context 知识图谱对比

| 指标 | Simple KG | Context-5 KG |
|------|-----------|--------------|
| 平均回合长度 | 30.00 | 1.00 |
| 平均质量分数 | 48.40 | 55.24 |
| 平均胜率 | 99.5% | 93.9% |
| 可用性 | ✅ 可用 | ❌ 失败 |

**结论**：Simple KG 表现优异，Context KG 因历史匹配问题在模拟中失效。

### 4.2 100 回合模拟结果（Simple KG）

```
Start State: 0
Episodes: 100
Max Steps: 30

Metric                                   Value
--------------------------------------------------
Avg Episode Length             30.00 +/- 0.00
Avg Quality Score              48.40
Avg Win Rate                   99.5%
```

**动作分布**：
| 排名 | 动作 | 次数 | 占比 |
|------|------|------|------|
| 1 | 1d | 223 | 7.4% |
| 2 | 4d | 213 | 7.1% |
| 3 | 4b | 196 | 6.5% |
| 4 | 0e | 149 | 5.0% |
| 5 | 3c | 148 | 4.9% |

**状态访问分布**：
| 排名 | 状态 | 次数 | 占比 |
|------|------|------|------|
| 1 | 109 | 294 | 9.8% |
| 2 | 544 | 278 | 9.3% |
| 3 | 396 | 236 | 7.9% |
| 4 | 21 | 157 | 5.2% |
| 5 | 405 | 149 | 5.0% |

### 4.3 典型轨迹分析

从状态 0 出发的典型轨迹：

```
Step 0:  State 0   --4d--> State 340  (quality=58.42, win_rate=95.2%)
Step 1:  State 340 --4d--> State 396  (quality=58.90, win_rate=99.5%)
Step 2:  State 396 --3f--> State 396  (quality=72.00, win_rate=100.0%)
Step 3:  State 396 --3f--> State 396  (quality=72.00, win_rate=100.0%)
...
```

**观察**：
1. 状态 396 是一个高质量状态，win_rate 达到 100%
2. 动作 `3f` 在状态 396 下质量分数最高（72.00）
3. 轨迹倾向于收敛到高质量状态（396, 544, 109 等）

---

## 5. 质量指标定义

### 5.1 质量分数计算

```python
quality_score = avg_future_reward × 0.7 + win_rate × 30.0
```

- **avg_future_reward**：从当前步骤到回合结束的累计奖励平均值
- **win_rate**：该状态-动作对在训练数据中的胜率

### 5.2 权重设计理由

- `0.7` 权重给未来奖励：强调长期收益
- `30.0` 权重给胜率：将 0-1 的胜率放大到与奖励同量级

---

## 6. 代码结构

```
scripts/simulate_decision_process.py
├── StateTransitionSimulator 类
│   ├── __init__(kg, verbose)         # 初始化
│   ├── _build_transitions()          # 构建状态转移网络
│   ├── roulette_wheel_select()       # 轮盘赌选择
│   ├── get_next_state()              # 获取下一状态
│   ├── simulate_episode()            # 模拟单个回合
│   └── run_simulation()              # 运行多个回合
└── main()                            # 命令行入口
```

---

## 7. 后续工作

1. **轨迹对比验证**：将模拟轨迹与专家轨迹进行对比，验证模拟的真实性
2. **多起始状态测试**：从不同状态出发，测试系统的适应性
3. **与 AdaptiveDecisionAgent 集成**：将知识图谱推荐集成到实际决策代理中
4. **状态相似性编码器**：当状态不在知识图谱中时，使用相似状态进行推荐

---

## 8. 相关文件

| 文件 | 说明 |
|------|------|
| `scripts/simulate_decision_process.py` | 模拟脚本 |
| `scripts/build_knowledge_graph.py` | 知识图谱构建脚本 |
| `src/decision/knowledge_graph.py` | 知识图谱核心类 |
| `cache/knowledge_graph/kg_simple.pkl` | 简单版知识图谱 |
| `output/simulation_log.json` | 模拟详细记录（示例） |
| `doc/DECISION_KNOWLEDGE_SYSTEM.md` | 知识图谱系统设计文档 |

---

*报告生成日期：2026-03-21*
