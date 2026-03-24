# 决策轨迹模拟系统设计文档

**版本**: v1.0  
**日期**: 2026-03-23  
**状态**: 设计完成，准备实施

---

## 1. 系统概述

### 1.1 目标

创建一个决策轨迹模拟脚本，从初始状态（状态0）出发，基于知识图谱进行动作推荐，通过状态-动作对预测下一状态，迭代执行直到到达终端状态（胜利/失败），从而验证知识图谱在决策引导方面的有效性。

### 1.2 核心流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    决策轨迹模拟流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐            │
│  │ 初始化   │───►│ 知识图谱推荐 │───►│ 动作选择     │            │
│  │ state=0 │    │ top-k动作   │    │ (轮盘赌)    │            │
│  └─────────┘    └─────────────┘    └──────┬──────┘            │
│                                            │                    │
│                                            ▼                    │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐            │
│  │ 记录轨迹 │◄───│ 检查终端     │◄───│ 预测下一状态 │            │
│  │ 继续迭代 │    │ 胜利/失败?   │    │ 概率/网络   │            │
│  └─────────┘    └─────────────┘    └─────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据资源

### 2.1 知识图谱

| 文件 | 路径 | 用途 |
|------|------|------|
| Simple KG | `cache/knowledge_graph/kg_simple.pkl` | 状态-动作质量评估 |
| Context-5 KG | `cache/knowledge_graph/kg_context_5.pkl` | 带上下文的动作推荐 |
| Transitions | `cache/knowledge_graph/kg_simple_transitions.pkl` | 状态转移概率数据 |

### 2.2 神经网络模型

| 模型 | 路径 | 用途 |
|------|------|------|
| StateTransitionPredictor | `cache/model/state_predictor.pth` | 神经网络预测下一状态 |
| QNetwork | `cache/model/q_network.pth` | 状态-动作价值评估 |

### 2.3 状态转移数据格式

```python
kg_simple_transitions.pkl 结构:
{
    state_id: {
        action: {
            'next_states': {next_state_id: count, ...},  # 转移频次
            'wins': int,      # 该状态-动作对的胜利次数
            'total': int,     # 总访问次数
            'win_rate': float # 胜率
        },
        ...
    },
    ...
}

# 示例
{
    0: {
        '4d': {
            'next_states': {10: 435, 196: 444, 340: 442},
            'wins': 1257,
            'total': 1321,
            'win_rate': 0.9515
        },
        '4b': {...},
        ...
    },
    ...
}
```

### 2.4 游戏结果数据

```python
# loader.game_results 格式
[
    [outcome, steps, score, penalty],  # Episode 0
    ...
]

# outcome: 'Win' 或 'Loss'
# steps: 回合长度
# score: 友军存活总分
# penalty: 敌军存活总分

# loader.state_log 格式
[
    [state_0, state_1, ..., state_n],  # Episode 0 的状态序列
    ...
]
```

---

## 3. 核心设计

### 3.1 终端状态检测

#### 3.1.1 终端状态映射构建

从 `state_log` 和 `game_results` 中识别终端状态：

```python
def build_terminal_state_map(state_log, game_results):
    """
    构建终端状态映射
    
    Returns:
        terminal_states: {
            state_id: {
                'is_terminal': bool,
                'win_count': int,     # 作为终止状态时的胜利次数
                'loss_count': int,    # 作为终止状态时的失败次数
                'win_rate': float,    # 胜率
                'total_count': int    # 作为终止状态的总次数
            }
        }
    """
    terminal_states = defaultdict(lambda: {
        'is_terminal': False,
        'win_count': 0,
        'loss_count': 0,
        'total_count': 0
    })
    
    for ep_idx, (states, result) in enumerate(zip(state_log, game_results)):
        if not states:
            continue
        
        last_state = states[-1]
        outcome = result[0] if result else 'Unknown'
        
        terminal_states[last_state]['is_terminal'] = True
        terminal_states[last_state]['total_count'] += 1
        
        if outcome.lower() == 'win':
            terminal_states[last_state]['win_count'] += 1
        else:
            terminal_states[last_state]['loss_count'] += 1
    
    # 计算胜率
    for state_id, info in terminal_states.items():
        if info['total_count'] > 0:
            info['win_rate'] = info['win_count'] / info['total_count']
    
    return dict(terminal_states)
```

#### 3.1.2 终端判定逻辑

```python
def check_terminal(state, terminal_states):
    """
    检查状态是否为终端状态
    
    Returns:
        (is_terminal, outcome)
        - is_terminal: True/False
        - outcome: 'win' / 'loss' / None
    """
    if state not in terminal_states:
        return False, None
    
    info = terminal_states[state]
    if not info['is_terminal']:
        return False, None
    
    # 根据胜率判断胜利/失败终端
    if info['win_rate'] >= 0.5:
        return True, 'win'
    else:
        return True, 'loss'
```

### 3.2 动作选择策略

#### 3.2.1 轮盘赌选择 (Roulette Wheel Selection)

```python
def roulette_wheel_select(actions, quality_key='quality_score'):
    """
    基于质量分数的轮盘赌选择
    
    Args:
        actions: {action: stats_dict}
        quality_key: 排序指标
    
    Returns:
        selected_action
    """
    action_list = list(actions.keys())
    qualities = [actions[a][quality_key] for a in action_list]
    
    # 偏移到正值
    min_q = min(qualities)
    if min_q < 0:
        qualities = [q - min_q + 0.1 for q in qualities]
    else:
        qualities = [q + 0.1 for q in qualities]
    
    # 归一化为概率
    total = sum(qualities)
    probs = [q / total for q in qualities]
    
    # 轮盘赌采样
    r = np.random.random()
    cumsum = 0
    for action, prob in zip(action_list, probs):
        cumsum += prob
        if r <= cumsum:
            return action
    
    return action_list[-1]
```

### 3.3 下一状态预测

#### 3.3.1 历史概率模式 (Probability Mode)

```python
def predict_next_state_probability(state, action, transitions):
    """
    基于历史转移概率预测下一状态
    
    Args:
        state: 当前状态ID
        action: 选择的动作
        transitions: 状态转移字典
    
    Returns:
        next_state: 采样得到的下一状态
    """
    if state not in transitions:
        return state  # 无转移数据，保持原状态
    
    if action not in transitions[state]:
        return state
    
    next_states_dict = transitions[state][action]['next_states']
    if not next_states_dict:
        return state
    
    # 加权随机采样
    states = list(next_states_dict.keys())
    counts = list(next_states_dict.values())
    total = sum(counts)
    probs = [c / total for c in counts]
    
    return np.random.choice(states, p=probs)
```

#### 3.3.2 神经网络模式 (Network Mode)

```python
def predict_next_state_network(state, action, predictor, temperature=1.0):
    """
    使用神经网络预测下一状态
    
    Args:
        state: 当前状态ID (tensor)
        action: 动作ID (tensor, 需要从action_str转换)
        predictor: StateTransitionPredictor模型
        temperature: softmax温度
    
    Returns:
        next_state: 预测的下一状态
    """
    predictor.eval()
    with torch.no_grad():
        # 预测概率分布
        top_k_states, top_k_probs = predictor.predict_top_k(
            state, action, k=5, temperature=temperature
        )
        
        # 采样或贪婪选择
        if predictor.training:
            idx = torch.multinomial(top_k_probs, 1)
        else:
            idx = torch.argmax(top_k_probs)
        
        return top_k_states[idx].item()
```

---

## 4. 脚本架构

### 4.1 文件结构

```
scripts/simulate_kg_trajectory.py
│
├── class TrajectorySimulator
│   │
│   ├── __init__(self, kg, transitions, terminal_states, 
│   │             predictor=None, mode='probability')
│   │
│   ├── 动作选择
│   │   ├── select_action(state, strategy='roulette')
│   │   └── roulette_wheel_select(actions, metric)
│   │
│   ├── 状态预测
│   │   ├── predict_next_state(state, action) -> next_state
│   │   ├── predict_by_probability(state, action)
│   │   └── predict_by_network(state, action)
│   │
│   ├── 终端检测
│   │   ├── check_terminal(state) -> (is_terminal, outcome)
│   │   └── is_win_terminal(state) / is_loss_terminal(state)
│   │
│   ├── 轨迹模拟
│   │   ├── simulate_episode(start_state, max_steps) -> dict
│   │   └── run_simulation(n_episodes, start_state) -> dict
│   │
│   └── 工具方法
│       ├── build_terminal_state_map(state_log, game_results)
│       └── get_trajectory_summary(trajectory) -> dict
│
├── def load_predictor(model_path) -> StateTransitionPredictor
├── def print_results(results)
│
└── def main() -> CLI入口
```

### 4.2 类设计

```python
class TrajectorySimulator:
    """
    决策轨迹模拟器
    
    从初始状态出发，使用知识图谱进行动作推荐和状态转移预测，
    模拟完整的决策轨迹。
    """
    
    def __init__(
        self,
        kg: DecisionKnowledgeGraph,
        transitions: Dict,
        terminal_states: Dict,
        predictor: Optional[nn.Module] = None,
        mode: str = 'probability',  # 'probability' or 'network'
        verbose: bool = False
    ):
        self.kg = kg
        self.transitions = transitions
        self.terminal_states = terminal_states
        self.predictor = predictor
        self.mode = mode
        self.verbose = verbose
    
    def simulate_episode(
        self,
        start_state: int = 0,
        max_steps: int = 50,
        top_k: int = 5
    ) -> Dict:
        """
        模拟单回合
        
        Returns:
            {
                'trajectory': [
                    {
                        'step': int,
                        'state': int,
                        'action': str,
                        'next_state': int,
                        'quality_score': float,
                        'win_rate': float,
                        'is_terminal': bool,
                        'outcome': str/None
                    },
                    ...
                ],
                'final_state': int,
                'outcome': str,  # 'win', 'loss', 'max_steps'
                'length': int,
                'avg_quality': float,
                'avg_win_rate': float
            }
        """
        pass
    
    def run_simulation(
        self,
        n_episodes: int = 100,
        start_state: int = 0,
        max_steps: int = 50
    ) -> Dict:
        """
        运行多回合模拟
        
        Returns:
            {
                'n_episodes': int,
                'outcomes': {
                    'win': int,
                    'loss': int,
                    'max_steps': int
                },
                'outcome_rates': {
                    'win': float,
                    'loss': float,
                    'max_steps': float
                },
                'avg_length': float,
                'std_length': float,
                'avg_quality': float,
                'avg_win_rate': float,
                'episodes': List[Dict]
            }
        """
        pass
```

---

## 5. 命令行接口

### 5.1 参数定义

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--kg-type` | str | simple | 知识图谱类型 (simple/context) |
| `--mode` | str | probability | 状态预测模式 (probability/network) |
| `--strategy` | str | roulette | 动作选择策略 (roulette/greedy) |
| `--episodes` | int | 100 | 模拟回合数 |
| `--start-state` | int | 0 | 起始状态 |
| `--max-steps` | int | 50 | 每回合最大步数 |
| `--top-k` | int | 5 | 候选动作数量 |
| `--seed` | int | 42 | 随机种子 |
| `--verbose` | flag | False | 详细输出模式 |
| `--save-trajectories` | str | None | 保存轨迹到JSON文件 |

### 5.2 使用示例

```bash
# 基本用法：历史概率模式 + 轮盘赌选择
python scripts/simulate_kg_trajectory.py --episodes 100

# 使用神经网络预测
python scripts/simulate_kg_trajectory.py --mode network --episodes 100

# 贪婪策略
python scripts/simulate_kg_trajectory.py --strategy greedy --episodes 100

# 详细输出 + 保存轨迹
python scripts/simulate_kg_trajectory.py --verbose --save-trajectories output/trajectories.json

# 从不同起始状态开始
python scripts/simulate_kg_trajectory.py --start-state 100 --episodes 50
```

---

## 6. 输出格式

### 6.1 控制台输出

```
============================================================
       DECISION TRAJECTORY SIMULATION RESULTS
============================================================

Configuration:
  Knowledge Graph:     simple
  Prediction Mode:     probability
  Action Strategy:     roulette
  Episodes:            100
  Start State:         0
  Max Steps:           50

Trajectory Outcomes:
  ┌─────────────────────────────────────────────────────────┐
  │ Win Terminals:        75 (75.0%)                        │
  │ Loss Terminals:       15 (15.0%)                        │
  │ Max Steps Reached:    10 (10.0%)                        │
  └─────────────────────────────────────────────────────────┘

Metrics:
  Avg Episode Length:     23.5 ± 12.3
  Avg Quality Score:      45.2
  Avg Win Rate:           0.82

Top 5 Most Visited States:
  State 396: 245 visits (12.3%)
  State 544: 198 visits (9.9%)
  State 109: 167 visits (8.4%)
  ...

Top 5 Most Selected Actions:
  4d: 312 times (15.7%)
  1d: 289 times (14.5%)
  4b: 267 times (13.4%)
  ...

Sample Trajectories:
  Episode   0: 0 → 340 → 396 → 544 [WIN]
  Episode   1: 0 → 109 → 21 → 405 → 396 [WIN]
  Episode   2: 0 → 340 → 196 → 10 [LOSS]
  ...

Verbose Mode (--verbose):
  Step  0: State   0 --4d--> State 340 (quality=58.42, win_rate=95.2%)
  Step  1: State 340 --4d--> State 396 (quality=58.90, win_rate=99.5%)
  Step  2: State 396 --3f--> State 396 (quality=72.00, win_rate=100.0%)
  ...
```

### 6.2 JSON输出格式

```json
{
  "config": {
    "kg_type": "simple",
    "mode": "probability",
    "strategy": "roulette",
    "n_episodes": 100,
    "start_state": 0,
    "max_steps": 50
  },
  "summary": {
    "outcomes": {
      "win": 75,
      "loss": 15,
      "max_steps": 10
    },
    "outcome_rates": {
      "win": 0.75,
      "loss": 0.15,
      "max_steps": 0.10
    },
    "avg_length": 23.5,
    "std_length": 12.3,
    "avg_quality": 45.2,
    "avg_win_rate": 0.82
  },
  "episodes": [
    {
      "episode_id": 0,
      "trajectory": [
        {
          "step": 0,
          "state": 0,
          "action": "4d",
          "next_state": 340,
          "quality_score": 58.42,
          "win_rate": 0.952
        },
        ...
      ],
      "final_state": 544,
      "outcome": "win",
      "length": 12,
      "avg_quality": 52.3
    },
    ...
  ]
}
```

---

## 7. 实现步骤

### Phase 1: 基础框架 (30分钟)

1. 创建 `scripts/simulate_kg_trajectory.py`
2. 实现 `TrajectorySimulator` 类框架
3. 实现 `build_terminal_state_map()` 函数
4. 实现轮盘赌选择逻辑

### Phase 2: 概率模式 (20分钟)

1. 加载 `kg_simple_transitions.pkl`
2. 实现 `predict_by_probability()` 方法
3. 实现轨迹模拟循环
4. 测试概率模式

### Phase 3: 网络模式 (20分钟)

1. 加载 `state_predictor.pth`
2. 实现 `predict_by_network()` 方法
3. 处理动作字符串到ID的转换
4. 测试网络模式

### Phase 4: 报告与输出 (15分钟)

1. 实现统计报告生成
2. 实现JSON输出
3. 添加详细输出模式
4. 完善CLI参数

### Phase 5: 验证与优化 (15分钟)

1. 运行完整测试
2. 验证终端状态检测
3. 对比概率模式与网络模式
4. 生成最终报告

---

## 8. 预期结果

### 8.1 成功指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| Win Terminal 到达率 | > 70% | 成功到达胜利终端的比例 |
| Avg Quality Score | > 40 | 平均轨迹质量分数 |
| Avg Win Rate | > 0.80 | 平均动作胜率 |

### 8.2 模式对比预期

| 模式 | 优点 | 缺点 | 预期表现 |
|------|------|------|----------|
| Probability | 基于真实数据，稳定可靠 | 无泛化能力 | Win率70-80% |
| Network | 有泛化能力，可预测未见状态 | 依赖训练质量 | Win率60-75% |

---

## 9. 后续扩展

1. **多起始状态测试**: 从不同状态出发验证系统适应性
2. **策略对比**: 轮盘赌 vs 贪婪 vs ε-贪婪
3. **与专家轨迹对比**: 验证模拟轨迹的真实性
4. **集成到决策系统**: 将模拟器集成到 AdaptiveDecisionAgent
5. **可视化轨迹**: 生成轨迹可视化图表

---

## 10. 相关文件

| 文件 | 说明 |
|------|------|
| `scripts/simulate_kg_trajectory.py` | 轨迹模拟脚本 (待创建) |
| `scripts/simulate_decision_process.py` | 现有模拟脚本 (参考) |
| `src/decision/knowledge_graph.py` | 知识图谱核心类 |
| `src/models/StateTransitionPredictor.py` | 状态转移预测器 |
| `cache/knowledge_graph/kg_simple_transitions.pkl` | 状态转移数据 |
| `cache/model/state_predictor.pth` | 神经网络权重 |

---

**文档版本**: v1.0  
**最后更新**: 2026-03-23  
**下一步**: 执行 Phase 1 - 创建基础框架
