# 基于 Knowledge Graph 的 RTS 游戏决策推演系统

> 技术报告 v1.0 | 2026-04-10

---

## 目录

1. [引言与问题背景](#1-引言与问题背景)
2. [关键科学问题与挑战](#2-关键科学问题与挑战)
3. [系统架构与技术路线](#3-系统架构与技术路线)
4. [核心算法](#4-核心算法)
5. [数据流](#5-数据流)
6. [学术创新点](#6-学术创新点)
7. [总结与展望](#7-总结与展望)

---

## 1. 引言与问题背景

### 1.1 项目目标

本项目构建了一个**交互式知识图谱驱动的决策推演系统**，用于 RTS（实时策略）游戏中的智能决策研究与可视化分析。系统从历史对局数据中提取决策经验，构建结构化的知识图谱，并在此基础上实现 Beam Search 多步推理与链式推演，支持决策路径的交互式浏览、评估与追溯。

### 1.2 应用场景

- **决策路径分析**：研究者可交互式浏览 KG 中的状态-动作转移关系，评估不同动作选择的质量
- **多步推理验证**：给定任意起始状态，Beam Search 生成多条候选路径，综合评分排序后推荐最优动作序列
- **推演模拟**：模拟从起始状态到终端的完整决策过程，支持单步/多步两种推演模式，评估策略鲁棒性
- **历史经验检索**：将推理路径与历史对局片段做序列匹配，验证决策的可靠性与历史依据

### 1.3 技术栈

| 层次 | 技术 |
|------|------|
| 数据存储 | Python pickle / YAML / NumPy (.npy) |
| 知识图谱 | `DecisionKnowledgeGraph` (自研) |
| 推理引擎 | `BeamSearchPredict` / `ChainRollout` (自研) |
| 可视化 | Streamlit 1.40.1 + pyvis + Plotly |
| 降维/分析 | scikit-learn MDS / scipy griddata |

---

## 2. 关键科学问题与挑战

### 2.1 状态空间稀疏性与泛化问题

RTS 游戏的状态空间通常是连续且高维的。通过离散化映射到有限状态集后，知识图谱仅覆盖实际访问过的状态子集，大量状态可能缺乏充分的统计样本。

**具体表现**：
- 部分状态只有少量访问记录（低 visits），其动作质量评估的统计置信度较低
- 状态转移存在概率性：同一 `(state, action)` 对可能转移到不同的下一状态，KG 中以转移计数近似概率分布
- KG 中约 43.1% 的状态存在自环（self-loop），即同一状态可能被重复访问

**当前应对策略**：
- `min_visits` 参数过滤低置信度动作，避免噪声统计误导决策
- 累积概率折扣 (`discount_factor`) 对远期预测施加指数衰减，控制不确定传播

### 2.2 多步决策中的不确定性传播

在多步推演场景中，每一步的状态转移都带有概率性，随着推演深度增加，实际轨迹偏离预测路径的概率迅速上升。

**数学建模**：

设状态转移概率为 $p(s_{t+1} | s_t, a_t)$，则一条长度为 $T$ 的路径的实际概率为：

$$P(\text{path}) = \prod_{t=0}^{T-1} p(s_{t+1} | s_t, a_t)$$

引入折扣因子 $\gamma$ 后，系统计算的累积概率为：

$$C_T = \prod_{t=0}^{T-1} p(s_{t+1} | s_t, a_t) \cdot \gamma^{t+1}$$

当 $\gamma = 0.9$ 时，5 步后的累积折扣为 $\gamma^5 \approx 0.59$，即路径末尾的不确定性权重已被削减近一半。

**挑战**：如何在不确定的环境下，生成尽可能长且可靠的决策路径？系统通过两种推演模式应对此问题（详见 4.4 节）。

### 2.3 探索-利用平衡 (Exploration-Exploitation Trade-off)

Beam Search 天然是一种偏向利用 (exploitation) 的贪心搜索策略——每步按质量评分排序后只保留 top-K 个 beam。这可能导致：
- 某些低即时质量但高长期价值的分支被过早剪枝
- 搜索集中在 KG 中已充分探索的"热门"区域

**当前应对策略**：
- 候选过采样：每个 beam 展开时从 KG 中取 `beam_width × 3`（至少 9）个候选动作，扩大候选池后再精选
- Epsilon-Greedy 策略：以概率 $\epsilon$ 随机选择动作，提供探索能力
- 6 种动作选择策略可切换（best_beam / best_subtree_quality / best_subtree_winrate / highest_transition_prob / random_beam / epsilon_greedy）

### 2.4 路径质量的多维评估

单一指标无法全面反映一条决策路径的质量。系统设计了四维综合评分体系：

| 维度 | 含义 | 权重 | 数据来源 |
|------|------|------|----------|
| 置信度 (Confidence) | 路径累积概率 | 0.30 | 转移概率 × 折扣因子 |
| 平均质量 (Avg Quality) | KG quality_score 均值 | 0.30 | 历史经验统计 |
| 平均胜率 (Avg Win Rate) | 历史胜率均值 | 0.30 | 对局结果统计 |
| 平均奖励 (Avg Reward) | 未来累积奖励均值 | 0.10 | 奖励信号统计 |

**归一化方法**：对同一批路径做 Min-Max 归一化到 $[0, 1]$，消除不同指标的尺度差异：

$$\text{score} = 0.30 \cdot \hat{c} + 0.30 \cdot \hat{q} + 0.30 \cdot \hat{w} + 0.10 \cdot \hat{r}$$

其中 $\hat{x} = (x - \min) / (\max - \min)$。

**设计意图**：三个等权维度（置信度、质量、胜率）反映三个不同的可靠性视角——路径的概率置信度、KG 经验推荐强度、历史胜负表现。奖励维度作为辅助参考，权重较低。

### 2.5 偏差恢复与路径切换

多步推演的核心难题：当实际状态转移偏离预测路径时，系统需要快速且有效地恢复。这涉及三个子问题：

1. **偏差检测**：实际到达状态 $\neq$ 预测状态
2. **恢复策略选择**：是否有可用的备选路径，还是需要重新搜索
3. **备选路径构建**：如何在搜索阶段预生成高质量的备选方案

系统借鉴了 CPU 分支预测中的"分支目标缓冲区"（Branch Target Buffer, BTB）思想，在搜索阶段预构建切换点映射（详见 4.4.2 节）。

---

## 3. 系统架构与技术路线

### 3.1 四层架构

```
┌─────────────────────────────────────────────────────────┐
│                    可视化层 (Presentation)                 │
│  Streamlit Web UI │ pyvis 图谱 │ Plotly 图表 │ 表格      │
├─────────────────────────────────────────────────────────┤
│                    决策层 (Decision)                      │
│  Beam Search 引擎 │ Chain Rollout 推演 │ 路径匹配        │
├─────────────────────────────────────────────────────────┤
│                    知识图谱层 (Knowledge)                  │
│  DecisionKnowledgeGraph │ 质量评分 │ 动作排名 │ 置信度   │
├─────────────────────────────────────────────────────────┤
│                    数据层 (Data)                          │
│  Episode 回放 │ 状态转移 │ HP 数据 │ 距离矩阵 │ MDS    │
└─────────────────────────────────────────────────────────┘
```

### 3.2 模块关系

```
visualize_kg_web.py (主 UI, ~3200 行)
├── kg_beam_search.py (Beam Search 引擎, ~370 行)
│   ├── BeamSearchResult (数据类)
│   ├── beam_search_predict() (核心搜索)
│   ├── find_optimal_action() (便捷决策接口)
│   └── get_beam_paths() (路径回溯)
├── chain_rollout.py (推演引擎, ~1060 行)
│   ├── RolloutNode / RolloutResult (数据类)
│   ├── SwitchPoint / PlanSegment (多步模式数据类)
│   ├── build_switching_map() (切换点构建)
│   ├── find_closest_switch_point() (偏差匹配)
│   ├── _chain_rollout_single_step() (单步模式)
│   └── _chain_rollout_multi_step() (多步模式)
├── knowledge_graph.py (知识图谱, ~460 行)
│   ├── ActionStats (统计单元)
│   └── DecisionKnowledgeGraph (图谱核心)
├── beam_matcher.py (路径匹配, ~156 行)
│   ├── MatchResult (匹配结果)
│   └── match_beam_paths() (滑动窗口匹配)
└── configs/kg_catalog.yaml (配置)
```

### 3.3 UI 功能模块

| Tab | 功能 | 核心交互 |
|-----|------|----------|
| 图谱可视化 | KG 网络图浏览 | 聚焦模式、质量筛选、终端高亮、布局切换 |
| 链式推理 | 单次 Beam Search 推理 | 路径推荐表 (14列) → 树图高亮 → 路径详情 → 片段匹配 |
| 滚动推演 | 端到端推演模拟 | 单步/多步模式、摘要统计、趋势图、束搜索追溯、全局树 |
| 原始数据 | 历史对局查询 | Episode/距离/HP 查询、MDS 地形图 |

### 3.4 技术路线

```
Phase 0: 数据采集
  原始对局 (node_log / action_log / game_result)
  → 状态序列 + 动作序列 + 奖励 + 结果
  → 状态距离矩阵 (pairwise distance)

Phase 1: 知识图谱构建
  Episode 数据 → DecisionKnowledgeGraph.build_from_data()
  → state_action_map: Dict[state, Dict[action, ActionStats]]
  → 状态转移图: Dict[state, Dict[action, {next_states: counts}]]
  → pickle 序列化

Phase 2: Beam Search 推理
  起始状态 → find_optimal_action()
  → beam_search_predict(kg, transitions, state, beam_width, max_steps, ...)
  → List[BeamSearchResult] (扁平节点列表, parent_idx 重建树)
  → get_beam_paths() → List[List[BeamSearchResult]] (根到叶路径)
  → 综合评分排序 → 路径推荐

Phase 3: 链式推演
  起始状态 → chain_rollout(kg, transitions, state, ..., rollout_mode)
  ├── single_step: 每步 beam search → 选最优动作 → 执行一步
  └── multi_step: 每段 beam search → 选主路径 → 沿路径多步执行
                  → 偏差时备选切换 / 重新搜索
  → RolloutResult (统一搜索树 + 主路径 + 分段记录)

Phase 4: 可视化与分析
  推理/推演结果 → 交互式 Web UI
  → 路径树图 (pyvis) + 趋势图 (Plotly) + 表格 (Streamlit)
  → 片段匹配 → 历史验证
```

---

## 4. 核心算法

### 4.1 知识图谱构建

#### 4.1.1 数据结构

知识图谱的核心是一个两级字典：

$$\text{KG} = \{(s, a) \rightarrow \text{ActionStats}\}$$

其中 `ActionStats` 记录了状态-动作对 $(s, a)$ 的全部历史统计：

| 字段 | 计算方式 |
|------|----------|
| `visits` | 累计访问次数 |
| `avg_step_reward` | $\frac{1}{N}\sum_{i=1}^{N} r_i^{\text{step}}$ |
| `avg_future_reward` | $\frac{1}{N}\sum_{i=1}^{N} R_i^{\text{future}}$ |
| `win_rate` | $\frac{|\{\text{win}\}|}{N}$ |
| `quality_score` | $0.7 \cdot \bar{R}^{\text{future}} + 30 \cdot \text{win\_rate}$ |

**质量分设计**：`quality_score` 以 70% 权重赋予累积未来奖励（长期价值），30% 权重赋予胜率（二元结果信号放大 30 倍使两者贡献量级相当）。

#### 4.1.2 累积未来奖励计算

对每个 episode，从末尾向前累加计算累积奖励：

$$R_t^{\text{future}} = \sum_{k=t}^{T-1} r_k$$

这确保每个时间步的"未来奖励"反映的是从该步到 episode 结束的总回报，与强化学习中的一致。

#### 4.1.3 Context-Aware 模式

KG 支持上下文感知的查询模式：key 为 $(s, h)$，其中 $h$ 是最近 $W$ 个动作的历史窗口（默认 $W=5$）。不同决策上下文下的同一状态拥有独立的动作统计，提高了决策精度。

### 4.2 Beam Search 推理引擎

#### 4.2.1 算法概述

Beam Search 是本系统的核心推理算法，用于从给定起始状态出发，在 KG 状态转移图上进行受限的多步搜索，生成一组候选决策路径。

**参数空间**：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `beam_width` | 3 | 每步保留的 beam 数 |
| `max_steps` | 5 | 最大搜索深度 |
| `score_mode` | quality | 评分维度 (quality/future_reward/win_rate) |
| `min_visits` | 1 | 动作最低访问次数 |
| `min_cum_prob` | 0.01 | 累积概率剪枝阈值 |
| `max_state_revisits` | 2 | 同一状态最大回访次数 |
| `discount_factor` | 0.9 | 概率折扣因子 |

#### 4.2.2 搜索流程

```
初始化: root = BeamNode(state=start, cum_prob=1.0, visited={start: 1})
beam_heads = [root]

for step = 0 to max_steps - 1:
    expanded = []
    
    for each beam_head in beam_heads:
        candidates = kg.get_top_k_actions(state, k=beam_width*3, min_visits)
        for each (action, quality) in candidates:
            for each (next_state, count) in transitions[action]:
                // 状态回访检查
                if visited[next_state] >= max_state_revisits: skip
                
                // 概率计算
                prob = count / total_count
                cum_prob = beam.cum_prob * prob * γ^(step+1)
                if cum_prob < min_cum_prob: skip  // 概率剪枝
                
                // 展开新节点
                expanded.append((score, new_node))
    
    // 排序 + 修剪
    expanded.sort(key=(score, cum_prob), reverse=True)  // 双键排序
    beam_heads = expanded[:beam_width]                   // 保留 top-K
```

#### 4.2.3 双键排序策略

展开节点的排序采用双键策略：

$$\text{primary key} = \text{score}_\text{mode}(s, a), \quad \text{secondary key} = C_\text{cum}$$

这意味着**质量优先**：在质量相同的情况下才选择概率更高的分支。这是一个贪心质量优先的策略，确保每个 beam 尽可能沿着高质量方向搜索。

#### 4.2.4 路径回溯

搜索完成后，所有节点被存储在一个扁平列表 `all_nodes` 中，每个节点携带 `parent_idx`（父节点在列表中的索引）。路径回溯通过 `get_beam_paths()` 实现：

1. **识别叶节点**：所有不被其他节点引用的索引 = 叶节点
2. **从叶到根回溯**：沿 `parent_idx` 链向上遍历
3. **反转**：得到根到叶的完整路径

```
      root (idx=0, parent=None)
       / \
      A     B (beam_0, beam_1)
     / \     \
    C   D     E (leaf nodes)
    
all_nodes: [root, A, B, C, D, E]
parent_idx: [None,  0, 0, 1, 1, 2]
leaf_indices: [3, 4, 5]

paths: [[root,A,C], [root,A,D], [root,B,E]]
```

### 4.3 路径片段匹配

#### 4.3.1 问题定义

给定一条 Beam Search 生成的查询路径 $Q = (q_1, q_2, \ldots, q_L)$（状态序列），在历史 episode 集合 $\mathcal{E}$ 中搜索最相似的子序列片段。

#### 4.3.2 匹配算法

采用**滑动窗口**策略：对每个 episode $\mathcal{E}_i$ 的状态序列，用长度为 $L$ 的窗口滑动，在每个位置计算两个维度的相似度。

**状态相似度**（有距离矩阵时）：

$$\text{sim}_\text{state} = \max\left(0, 1 - \frac{\frac{1}{L}\sum_{j=1}^{L} d(s_{i,t+j}, q_j)}{d_\text{max}}\right)$$

其中 $d(s, q)$ 为状态距离矩阵中的预计算距离，$d_\text{max}$ 为矩阵中的最大距离值。若平均距离超过 `max_state_distance` 阈值则直接剪枝。

**动作匹配率**：

$$\text{match}_\text{action} = \frac{|\{j : a_{i,t+j} = q_j^\text{action}\}|}{L}$$

**综合评分**：

$$\text{score} = \alpha \cdot \text{sim}_\text{state} + (1-\alpha) \cdot \text{match}_\text{action}, \quad \alpha = 0.6$$

状态相似度权重更高（60%），反映"相似的决策上下文比具体的动作选择更能反映局势"这一设计理念。

### 4.4 链式推演 (Chain Rollout)

#### 4.4.1 单步推演模式 (Single Step)

单步模式是传统的逐步决策流程：每步执行一次完整的 Beam Search，从中选出一个最优动作并执行一步。

```
for step = 0 to max_steps:
    1. beam search → results
    2. 挂载 beam 树到统一树
    3. 按动作策略排序候选动作
    4. 尝试执行排序最高的有效动作
    5. 累积概率检查
    6. 推进到下一状态
```

**特点**：搜索频率 = 执行步数（1:1），每步都重新评估，适应高不确定性环境，但计算开销大。

#### 4.4.2 多步推演模式 (Multi Step) — 核心创新

多步模式采用 **"计划-执行-偏转" (Plan-Execute-Diverge)** 循环架构：

```
while 有余量:
    ── Phase 1: 规划 (Planning) ──
    beam search → all_paths
    主路径 = argmax(composite_score(all_paths))
    备选路径 = build_switching_map(main_path, all_paths)
    
    ── Phase 2: 执行 (Execution) ──
    for each step in main_path:
        执行计划动作
        actual_state = 转移采样()
        
        if actual_state == predicted:
            continue  // 按计划执行
        
        ── Phase 3: 偏转 (Divergence) ──
        if find_match(actual_state, switching_map):
            切换到备选路径, 执行剩余动作
        else:
            重新搜索  // re-search
```

#### 4.4.3 备选路径切换机制

**切换点构建 (`build_switching_map`)**：

此机制借鉴 CPU 分支预测中的**分支目标缓冲区**思想。在搜索阶段，系统为每个分叉位置预计算备选方案：

1. **评分阈值初筛**：丢弃综合评分低于主路径 $30\%$ 的路径
2. **分叉点分析**：找到每条备选路径与主路径的第一个分歧位置
3. **分叉点去重**：每个分叉位置最多保留评分最高的 3 条备选路径
4. **切换点生成**：对保留的备选路径，从分叉点开始为每个步骤生成 `SwitchPoint`

```
主路径: S0 ─A─→ S1 ─B─→ S3 ─C─→ S5
                       \
备选路径:               D─→ S4 ─E─→ S6

分叉位置: step=1 (S1)
SwitchPoint: {plan_position=1, predicted_state=S3, backup_path_idx=1, 
              backup_step=2, remaining=[E]}
```

**两阶段偏差匹配 (`find_closest_switch_point`)**：

1. **精确匹配**：线性扫描切换点，若 `predicted_state == actual_state` 则直接匹配
2. **距离回退**：若提供了距离矩阵，找距离最小且 $< \text{threshold}$ 的切换点

这允许即使实际状态不是精确预测的状态，只要在距离阈值内（默认 0.2），也能利用备选路径。

#### 4.4.4 偏差类型分类

| 偏差类型 | 含义 | 处理方式 |
|----------|------|----------|
| `none` | 无偏差，正常走完主路径 | 终止当前 segment |
| `backup_switch` | 实际状态匹配到备选路径 | 切换后继续执行 |
| `re_search` | 实际状态无匹配，需重新规划 | 结束 segment，进入下一轮搜索 |
| `no_valid_transition` | 动作无有效转移 | 结束 segment |
| `low_cum_prob` | 累积概率过低 | 终止推演 |

#### 4.4.5 推演摘要指标

多步模式额外提供以下统计：

- **规划段数** (`plan_segments` 数量)：反映了决策环境的不确定性程度
- **重新规划次数** (`total_re_searches`)：主路径预测失败需要完全重新搜索的次数
- **备选切换次数** (`total_backup_switches`)：通过预构建的备选路径快速恢复的次数
- **路径命中率**：`主路径执行步数 / 总规划步数`，衡量规划质量

**高命中率 + 低重规划次数** = 决策环境可预测，策略有效
**低命中率 + 高备选切换** = 环境有不确定性，但备选机制有效

---

## 5. 数据流

### 5.1 端到端数据链路

```
┌──────────────────────────────────────────────────────────────────┐
│  数据采集层                                                       │
│                                                                  │
│  node_log.txt ──→ 状态序列 List[List[int]]                       │
│  action_log.csv ──→ 动作序列 List[List[str]]                      │
│  game_result.txt ──→ 结果 List[str] + 分数 List[float]           │
│  pairwise_distance ──→ 距离矩阵 np.ndarray (N×N)                 │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  知识图谱层                                                       │
│                                                                  │
│  DecisionKnowledgeGraph.build_from_data()                        │
│  ├── 遍历所有 episodes                                            │
│  ├── 计算累积未来奖励 (反向累加)                                   │
│  ├── 构建 state_action_map: {state → {action → ActionStats}}     │
│  ├── 生成状态转移图: {state → {action → {next_state → count}}}  │
│  └── pickle 序列化 → cache/knowledge_graph/*.pkl                 │
│                                                                  │
│  状态转移图 → pickle → cache/knowledge_graph/*_transitions.pkl   │
│  距离矩阵 → npy → cache/npy/state_distance_matrix_*.npy         │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  决策层                                                           │
│                                                                  │
│  链式推理 (find_optimal_action)                                   │
│  ├── KG 查询 → top-K 候选动作                                    │
│  ├── beam_search_predict() → List[BeamSearchResult]              │
│  ├── get_beam_paths() → List[List[BeamSearchResult]]              │
│  ├── _compute_composite_scores() → 排序                         │
│  ├── 路径推荐表 (14列 + 综合评分)                                │
│  └── match_beam_paths() → 历史片段匹配                          │
│                                                                  │
│  链式推演 (chain_rollout)                                         │
│  ├── single_step: 每步 beam search → 选动作 → 执行一步          │
│  ├── multi_step: 每段 beam search → 选主路径 → 多步执行          │
│  │   ├── build_switching_map() → 备选切换点                     │
│  │   ├── find_closest_switch_point() → 偏差匹配                 │
│  │   └── PlanSegment 记录 → 分段统计                            │
│  └── RolloutResult (统一树 + 主路径 + beam 快照 + 分段记录)      │
│                                                                  │
│  序列化桥接 (st.cache_data 不支持 dataclass)                      │
│  ├── BeamSearchResult → dict → 缓存 → dict → BeamSearchResult   │
│  └── RolloutNode → dict → 缓存 → dict → RolloutNode            │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  可视化层                                                         │
│                                                                  │
│  pyvis 图渲染                                                     │
│  ├── 图谱可视化: BFS 聚焦 + quality 着色 + 终端高亮             │
│  ├── 路径树图: parent_idx 重建分层树 + highlight 联动            │
│  └── 推演全局树: RolloutResult 统一树 + 主路径蓝色高亮           │
│                                                                  │
│  Plotly 图表                                                      │
│  ├── 胜率/质量趋势 (双 Y 轴折线图 + 50% 基准线)                │
│  └── MDS 地形图 (降维 + HP 差值等高线插值)                      │
│                                                                  │
│  Streamlit 表格                                                   │
│  ├── 路径推荐 (14列 + on_select 联动)                            │
│  ├── 路径详情 (分叉标记 ↑/➤)                                    │
│  ├── 推演轨迹 (段类型标记)                                       │
│  └── 片段匹配 (状态相似度 + 动作匹配率 + 综合评分)              │
│                                                                  │
│  iframe 嵌入 (base64 data URI + 版本号软刷新)                    │
│  导出 (HTML/JSON 下载链接)                                       │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 BeamSearchResult 数据结构

```python
@dataclass
class BeamSearchResult:
    step: int                  # 搜索步数 (0=根)
    state: int                 # 状态 ID
    action: str                # 动作
    cumulative_probability: float  # 累积转移概率 (含折扣)
    quality_score: float       # KG 质量分
    win_rate: float            # 历史胜率
    avg_step_reward: float     # 平均即时奖励
    avg_future_reward: float   # 平均未来奖励
    beam_id: int               # beam 编号 (仅单步内有效)
    parent_idx: Optional[int]  # 父节点在列表中的索引
```

`parent_idx` 是重建搜索树的关键：所有节点存储在一个扁平列表中，通过 `parent_idx` 链式回溯可以恢复完整的树结构和任意根到叶路径。

### 5.3 RolloutNode 数据结构

```python
@dataclass
class RolloutNode:
    id: int                        # 节点 ID
    parent_id: Optional[int]       # 父节点 ID
    children_ids: List[int]        # 子节点 ID
    state: int                     # 决策状态
    action: Optional[str]          # 执行的动作
    beam_id: Optional[int]         # beam 编号 (None=非 beam 预测)
    quality_score / win_rate / avg_future_reward / avg_step_reward: float
    visits: int                    # KG 访问次数
    transition_prob: float         # 实际转移概率
    cumulative_probability: float  # 累积概率
    rollout_depth: int             # 推演深度 (-1=根)
    is_on_chosen_path: bool        # 是否在选定路径上
    is_terminal: bool              # 是否终端状态
    is_beam_root: bool             # 是否为 beam search 挂载根
```

### 5.4 缓存策略

系统采用双层缓存架构：

| 缓存类型 | 技术 | 对象 | 用途 |
|----------|------|------|------|
| 内存缓存 | `@st.cache_data` | KG 数据、推理结果、MDS | 快速响应重复查询 |
| 内存缓存 | `@st.cache_resource` | KG 对象实例、Episode 数据、距离矩阵 | 避免重复反序列化 |
| 磁盘缓存 | `.npy` 文件 | MDS 降维结果 | 跨会话持久化 |

**序列化桥接**：`BeamSearchResult` 和 `RolloutNode` 是自定义 dataclass，不直接支持 Streamlit 的 `cache_data` 序列化。通过手动转为 dict（移除类型信息）再恢复实现缓存穿透。

### 5.5 UI 交互数据流

**链式推理 Tab 的 highlight 联动**：

```
st.dataframe(on_select="rerun")
  → event.selection.rows[0] → int 索引
  → rec_rows[索引]["Path"] → 原始路径索引
  → new_path != old_path? → st.session_state.pred_selected_path = new_path
  → st.rerun()
  → selected_path = beam_paths[selected_path_idx]
  → highlight_set = {i for i, r in enumerate(results) if r in selected_path}
  → render_beam_tree(results, highlight_indices=highlight_set)
  → 金色+加粗选中路径, 其余半透明
```

**关键守卫**：`on_select="rerun"` 后 `event.selection.rows` 仍非空（选中状态保留），必须通过 `new_path != old_path` 比较防止无限 rerun 死循环。

---

## 6. 学术创新点

### 6.1 备选路径切换机制 — 类 BTB 的决策恢复

**核心思想**：借鉴 CPU 分支预测中的**分支目标缓冲区（BTB）**设计，在 Beam Search 的搜索阶段预构建"偏差恢复方案"，而非等到偏差发生后再实时计算。

**创新之处**：
1. **预计算切换点**：`build_switching_map()` 在搜索阶段就为每个分叉位置准备多条备选路径，偏差发生时 O(1) 查表切换
2. **动态裁剪**：评分阈值 + 分叉点去重的组合策略，在保证备选质量的同时控制切换点数量
3. **两阶段匹配**：精确优先 + 距离回退的分层匹配策略，兼顾精确性和鲁棒性

**对比传统方法**：

| 方法 | 偏差恢复策略 | 预计算开销 | 恢复延迟 |
|------|-------------|-----------|---------|
| 纯 Re-search | 重新执行完整 beam search | 无 | O(beam_search_cost) |
| MCTS 备选 | 保留所有子树 | 高（存储所有节点） | O(查表) |
| 本方法 (BTB) | 预构建切换点映射 | 低（仅保留 top-3/fork） | O(查表+执行) |

### 6.2 四维综合路径评分

**核心思想**：决策路径的质量无法由单一指标衡量，需要同时考虑路径的概率置信度、经验质量、历史胜率和奖励信号。

**创新之处**：
1. **Min-Max 归一化**：消除不同指标尺度差异，使权重具有可比性
2. **等权三维 + 辅助一维**：置信度/质量/胜率各占 30% 反映"多角度验证"，奖励 10% 作为辅助
3. **趋势指标**：Q趋势和WR趋势（前半段 vs 后半段比较）反映路径的"改善/恶化"方向

### 6.3 多步推演的计划-执行-偏转循环

**核心思想**：区别于传统的"搜索一步执行一步"的贪心方法，系统采用"搜索一段执行多步"的策略，在偏离时利用预构建的备选方案快速恢复，减少不必要的搜索开销。

**创新之处**：
1. **搜索-执行比优化**：multi_step 模式的搜索频率远低于执行步数（1:N），显著降低计算开销
2. **PlanSegment 记录**：完整记录每段的类型（初始规划/重新规划）、执行的动作列表、偏离原因，支持事后分析
3. **路径命中率指标**：量化评估规划质量，"主路径执行步数 / 总规划步数"直观反映策略的可预测性

### 6.4 统一树结构的可追溯性

**核心思想**：推演过程中所有的 beam search 子树都被挂载到一棵统一的 `RolloutResult` 树中，支持任意粒度的追溯。

**创新之处**：
1. **Beam 节点复用**：如果 beam search 曾经预测过某个转移，实际执行时直接复用该节点（`_advance_to_next_state` 中的 `beam_predicted_child` 匹配），保持树的连通性
2. **多维度追溯**：束搜索追溯区域可以查看任意推演步骤的 beam search 子树，与链式推理使用完全一致的渲染逻辑（路径推荐表 + highlight + 路径详情）

---

## 7. 总结与展望

### 7.1 系统总结

本系统实现了一个完整的基于知识图谱的 RTS 决策推演框架，包含：

- **知识图谱构建**：从历史对局中自动构建状态-动作经验统计表，支持上下文感知查询
- **Beam Search 推理**：多步受限搜索，双键排序（质量优先 + 概率次之），路径回溯与综合评分
- **多步推演**：计划-执行-偏转循环，BTB 式备选路径切换，支持单步/多步两种模式
- **交互式可视化**：路径树图 highlight 联动、趋势分析、片段匹配、MDS 地形图

### 7.2 关键设计权衡

| 设计选择 | 优势 | 劣势 |
|----------|------|------|
| Beam Search (vs MCTS) | 速度快，路径可解释 | 贪心偏向，缺乏模拟验证 |
| 质量优先排序 | 每步选择局部最优 | 可能错过长期价值高的低质量动作 |
| 折扣累积概率 | 控制远期不确定性 | 可能过早剪枝有价值的长路径 |
| BK-Tree HP 查找表 | 快速 O(1) 查询 | 需要预构建，覆盖范围有限 |
| 暴力片段匹配 | 实现简单，结果精确 | O(B×E×T×L) 复杂度，大数据集慢 |

### 7.3 未来展望

1. **搜索算法升级**：引入 Monte Carlo Tree Search (MCTS) 或 AlphaZero 式的自评估搜索，通过模拟验证改善贪心偏差
2. **神经网络增强**：用神经网络学习状态表示和动作价值函数，替代 KG 的频率统计，提升泛化能力
3. **近似索引加速**：用 FAISS / KD-Tree 等近似最近邻索引加速片段匹配，支持大规模 episode 数据
4. **增量 KG 更新**：支持在线增量更新 KG（当前需要完整重建），适应持续学习场景
5. **多智能体扩展**：从单智能体决策扩展到多智能体协同决策
6. **路径树图着色优化**：将 beam_id 着色改为路径索引着色，更直观地展示路径分叉与合并关系
