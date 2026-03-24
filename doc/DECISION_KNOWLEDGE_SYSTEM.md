# Decision Knowledge System Design Document

## 技术报告：实时决策知识系统

**版本**: v1.0  
**日期**: 2026-03-21  
**状态**: 设计完成，准备实施

---

## 1. 项目背景

### 1.1 问题分析

#### 原始问题：准确率评估的根本性错误

原始评估方案将决策问题视为**分类问题**：
- 假设训练数据中的动作是"正确答案"
- 评估指标：预测动作与训练数据匹配的"准确率"

**问题**:
- 训练数据是**专家轨迹**，不是标签
- 同一状态下可能有多个合理动作
- 实时决策是**生成**，不是**分类**

#### 示例说明

```
状态 S1 下:
  - 样本1: 采取动作 a，最终胜利
  - 样本2: 采取动作 b，最终失败
  - 样本3: 采取动作 a，最终胜利
  - 样本4: 采取动作 b，最终胜利

模型预测动作 a:
  - "准确率"角度: 50% (2/4样本)
  - 但这毫无意义！a和b都可能是合理选择
```

### 1.2 系统目标重新定义

**核心目标**: 为实时决策提供**多样性的高质量动作**

关键特性:
- **多样性**: 不是预测单个动作，而是推荐动作集合
- **高质量**: 每个推荐都有质量保证（预期回报）
- **可解释**: 每个推荐都有置信度（数据支持）

---

## 2. 数据分析

### 2.1 原始数据结构

```
数据来源: D:/白春辉/实验平台/pymarl/results_HRL_new/Q-bktree/MarineMicro_MvsM_4/6/
```

| 数据类型 | 格式 | 说明 |
|---------|------|------|
| **动作** | `4d` (2字符) | 第1字符=聚类索引(0-4)，第2字符=动作字母(a-k) |
| **状态** | 状态ID (0-939) | 映射到(cluster_primary, cluster_secondary) |
| **奖励** | 即时奖励 | 每个时间步的奖励值 (0, 15, -18, -12 等) |
| **结果** | [Win/Loss, ...] | 最终结果 (胜/负, 友军损失, 敌军损失, 总分) |

### 2.2 BK-Tree结构

```
Primary BK-Tree:
  - Root cluster_id: 1
  - Children: 21 clusters
  
Secondary BK-Trees:
  - 23 secondary trees
  - Each tree: (primary_cluster, secondary_cluster) -> states
  
State Mapping:
  - State 0: cluster=(1, 1), score=0.00
  - State 1: cluster=(1, 2), score=10.54
  - State 3: cluster=(2, 1), score=-3.20
  - ...
```

### 2.3 状态-动作多样性分析

```
Total unique states: 940
Action diversity per state:
  Min: 1
  Max: 11
  Mean: 3.61
  Median: 2.0

Distribution:
  1 action:  33.6% states (305)
  2 actions: 17.8% states (162)
  3 actions: 11.2% states (102)
  4 actions:  7.5% states (68)
  5+ actions: 29.9% states (273)
```

**关键发现**: 同一状态下动作选择高度多样化，证明这不是简单的分类问题。

### 2.4 动作质量分析

| State | Action | Visits | Step_Reward | Future_Reward | Win_Rate |
|-------|--------|--------|-------------|---------------|----------|
| 0 | 4c | 2 | 0.0 | 39.0 | 100.0% |
| 0 | 4b | 360 | 0.0 | 34.3 | 88.9% |
| 0 | 0e | 4 | 0.0 | 19.5 | 75.0% |
| 0 | 4d | 33 | 0.0 | 19.2 | 75.8% |
| 0 | 3h | 1 | 0.0 | 18.0 | 100.0% |

**关键发现**: 
- 不同动作的质量差异巨大 (win_rate: 75% ~ 100%)
- 动作频率 ≠ 动作质量 (4b有360次访问但win_rate=88.9%，4c只有2次但win_rate=100%)
- **future_reward + win_rate** 是最佳质量指标

---

## 3. 系统设计

### 3.1 核心架构

```
┌──────────────────────────────────────────────────────────────┐
│                Decision Knowledge System                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────────────┐      ┌─────────────────────────┐ │
│  │   Knowledge Graph      │      │  Similarity Encoder     │ │
│  │   (Two Versions)       │◄────►│  (Distance-based)       │ │
│  │                        │      │                         │ │
│  │  - Simple (state-only) │      │  - Cluster Encoder      │ │
│  │  - Context-Aware       │      │  - State Encoder        │ │
│  │    (state + history)   │      │  - Distance Prediction  │ │
│  └────────────────────────┘      └─────────────────────────┘ │
│           │                              │                    │
│           ▼                              ▼                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Quality Evaluation                        │  │
│  │                                                        │  │
│  │  Metrics:                                              │  │
│  │  - Quality Score (future_reward + win_rate)            │  │
│  │  - Coverage (hits in knowledge graph top-k)            │  │
│  │  - Diversity (entropy of recommendations)              │  │
│  │  - Confidence (frequency in data)                      │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 知识图谱设计

#### 3.2.1 数据结构

```python
class DecisionKnowledgeGraph:
    """
    决策知识图谱 - 支持两个版本
    
    Version 1 (Simple): 
      Key = state_id
      优点: 快速，数据充足
      缺点: 忽略历史上下文
      
    Version 2 (Context-Aware):
      Key = (state_id, history_hash)
      history_hash = hash of recent N actions
      优点: 更精确的上下文匹配
      缺点: 需要更多数据，稀疏性问题
    """
    
    # 核心数据结构
    state_action_map: Dict[Key, Dict[Action, ActionStats]]
    
    # ActionStats 包含:
    {
        'visits': int,           # 访问次数
        'avg_step_reward': float,    # 平均即时奖励
        'avg_future_reward': float,  # 平均未来累积奖励
        'win_rate': float,           # 胜率
        'trajectories': List[int],   # 关联的episode索引
    }
```

#### 3.2.2 关键方法

| 方法 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `get_key(state, history)` | 生成键 | state_id, 可选history | Key |
| `get_top_k_actions(state, history, k, metric)` | 获取top-k动作 | state, k, 指标 | [(action, quality), ...] |
| `get_action_quality(state, action, history)` | 获取动作质量 | state, action | ActionStats |
| `get_similar_states(state, k)` | 获取相似状态 | state, k | [state_id, ...] |
| `get_action_confidence(state, action)` | 获取置信度 | state, action | frequency |

### 3.3 状态相似性编码器

#### 3.3.1 设计动机

**问题**: 
- distance_matrix (940x940) 太大，加载慢 (98MB)
- 推理时需要快速计算状态相似度
- 需要泛化到未见过的状态组合

**方案**: 训练编码器学习距离映射

#### 3.3.2 模型架构

```python
class StateSimilarityEncoder(nn.Module):
    """
    混合编码器架构
    
    组成:
      1. Cluster Encoder: 从BK-Tree结构学习cluster embedding
      2. State Encoder: 学习fine-grained state embedding
      3. Distance Predictor: 预测两状态间的距离
    """
    
    def __init__(self, state_dim=940, embed_dim=64, cluster_dim=24):
        # Cluster Encoder: (primary, secondary) -> 24 dim
        self.cluster_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, cluster_dim)
        )
        
        # State Encoder: state_id -> 40 dim
        self.state_encoder = nn.Sequential(
            nn.Embedding(state_dim + 1, 128),
            nn.Linear(128, embed_dim - cluster_dim)
        )
        
        # Final embedding: 24 + 40 = 64 dim
    
    def forward(self, state_ids, cluster_tuples):
        """
        Returns: embeddings [batch, 64]
        """
        state_emb = self.state_encoder(state_ids)        # [batch, 40]
        cluster_emb = self.cluster_encoder(cluster_tuples)  # [batch, 24]
        return torch.cat([state_emb, cluster_emb], dim=-1)
    
    def compute_distance(self, state1, cluster1, state2, cluster2):
        """
        Predict distance between two states
        """
        emb1 = self.forward(state1, cluster1)
        emb2 = self.forward(state2, cluster2)
        cos_sim = F.cosine_similarity(emb1, emb2)
        predicted_dist = (1 - cos_sim) * max_distance
        return predicted_dist
```

#### 3.3.3 训练策略

```
训练数据:
  - 采样N对(state_i, state_j)
  - 查询distance_matrix[i, j]作为监督信号
  
损失函数:
  loss = MSE(predicted_distance, actual_distance)
  
训练配置:
  - N_pairs: 100,000
  - Batch size: 1024
  - Epochs: 50
  - Learning rate: 1e-3
  
验证:
  - 在测试集上计算预测误差
  - 对比直接使用distance_matrix的准确性
```

### 3.4 评估指标设计

#### 3.4.1 核心指标

| 指标 | 定义 | 计算方式 | 目标 |
|------|------|---------|------|
| **Quality Score** | 推荐动作的平均质量 | mean(future_reward of top-k) | 越高越好 |
| **Win Rate** | 推荐动作的平均胜率 | mean(win_rate of top-k) | 越高越好 |
| **Coverage** | 推荐动作覆盖知识图谱的比例 | \|recommended ∩ kg_top_k\| / k | 越高越好 |
| **Diversity** | 推荐动作的多样性 | entropy(action_distribution) | 适中 |
| **Confidence** | 推荐动作在数据中的频率 | mean(frequency of actions) | 越高越可靠 |

#### 3.4.2 评估流程

```python
def evaluate_decision_quality(agent, knowledge_graph, test_states):
    results = {
        'quality_scores': [],
        'win_rates': [],
        'coverage_scores': [],
        'diversity_scores': [],
        'confidence_scores': [],
    }
    
    for state in test_states:
        # 1. 获取模型推荐
        recommended = agent.get_top_k_actions(state, k=5)
        
        # 2. 获取知识图谱ground truth
        kg_top_k = knowledge_graph.get_top_k_actions(state, k=5)
        
        # 3. 计算各项指标
        quality = mean([kg.get_action_quality(state, a)['avg_future_reward'] 
                       for a in recommended])
        win_rate = mean([kg.get_action_quality(state, a)['win_rate'] 
                        for a in recommended])
        coverage = len(set(recommended) & set(kg_top_k)) / k
        diversity = entropy([1/len(recommended)] * len(recommended))
        confidence = mean([kg.get_action_confidence(state, a) 
                          for a in recommended])
        
        results['quality_scores'].append(quality)
        results['win_rates'].append(win_rate)
        results['coverage_scores'].append(coverage)
        results['diversity_scores'].append(diversity)
        results['confidence_scores'].append(confidence)
    
    return results
```

---

## 4. 实施计划

### 4.1 文件结构

```
PredictionRTS/
├── src/
│   ├── decision/
│   │   ├── knowledge_graph.py          # NEW: 决策知识图谱
│   │   ├── model_pool.py               # (existing)
│   │   └── strategy_router.py          # (existing)
│   │
│   └── models/
│       ├── StateSimilarityEncoder.py   # NEW: 相似性编码器
│       ├── DecisionTransformer.py      # (existing)
│       ├── QNetwork.py                 # (existing)
│       └── StateTransitionPredictor.py # (existing)
│
├── scripts/
│   ├── build_knowledge_graph.py        # NEW: 构建知识图谱
│   ├── train_similarity_encoder.py     # NEW: 训练编码器
│   ├── evaluate_decision_quality.py    # NEW: 质量评估
│   └── run_adaptive_agent.py           # (existing)
│
├── cache/
│   ├── knowledge_graph/
│   │   ├── kg_simple.pkl               # 知识图谱 (无上下文)
│   │   ├── kg_context_5.pkl            # 知识图谱 (5步上下文)
│   │   └── kg_context_10.pkl           # 知识图谱 (10步上下文)
│   │
│   └── model/
│       ├── similarity_encoder.pth      # 相似性编码器
│       └── ...
│
└── output/
    └── evaluation/
        └── decision_quality_YYYYMMDD.md
```

### 4.2 实施阶段

#### Phase 1.1: 基础知识图谱 (预计30分钟)

**任务**:
1. 创建 `src/decision/knowledge_graph.py`
   - 实现DecisionKnowledgeGraph类
   - 支持Simple和Context两个版本
   - 实现核心方法

2. 创建 `scripts/build_knowledge_graph.py`
   - 加载原始数据
   - 构建统计信息
   - 保存到cache

3. 执行构建
   - Simple版本
   - Context版本 (window=5, 10)

**验证点**: 
- 知识图谱是否成功构建？
- Top-k动作是否合理？（人工检查几个case）

#### Phase 1.2: 质量评估系统 (预计20分钟)

**任务**:
1. 创建 `scripts/evaluate_decision_quality.py`
   - 实现评估指标
   - 生成Markdown报告
   - 对比不同模型/策略

2. 评估当前模型
   - DT_ctx5
   - DT_ctx10
   - DT_ctx20

**验证点**:
- 评估指标是否有效？
- 能否区分好坏模型？

#### Phase 1.3: 状态相似性编码器 (预计40分钟，可选)

**任务**:
1. 创建 `src/models/StateSimilarityEncoder.py`
   - 实现混合编码器架构
   - 实现距离预测

2. 创建 `scripts/train_similarity_encoder.py`
   - 采样训练数据
   - 训练模型
   - 评估准确性

3. 集成到知识图谱
   - 在get_similar_states中使用编码器
   - 对比distance_matrix的准确性

**验证点**:
- 编码器预测误差 < 10%？
- 推理速度是否显著提升？

---

## 5. 使用示例

### 5.1 构建知识图谱

```python
from src.decision.knowledge_graph import DecisionKnowledgeGraph
from src.data.loader import DataLoader

# 加载数据
loader = DataLoader(cfg)

# 构建Simple版本
kg_simple = DecisionKnowledgeGraph(use_context=False)
kg_simple.build_from_data(
    state_episodes=loader.state_log,
    action_episodes=loader.action_log_raw,  # 需要包含cluster信息
    rewards=loader.r_log,
    game_results=loader.game_results
)
kg_simple.save('cache/knowledge_graph/kg_simple.pkl')

# 构建Context版本
kg_context = DecisionKnowledgeGraph(use_context=True, context_window=5)
kg_context.build_from_data(...)
kg_context.save('cache/knowledge_graph/kg_context_5.pkl')
```

### 5.2 查询高质量动作

```python
# 加载知识图谱
kg = DecisionKnowledgeGraph.load('cache/knowledge_graph/kg_simple.pkl')

# 获取state 0的top-5高质量动作
top_actions = kg.get_top_k_actions(
    state=0,
    k=5,
    metric='future_reward'  # 或 'win_rate', 'avg_step_reward'
)

# 输出:
# [('4c', {'visits': 2, 'avg_future_reward': 39.0, 'win_rate': 1.0}),
#  ('4b', {'visits': 360, 'avg_future_reward': 34.3, 'win_rate': 0.889}),
#  ('0e', {'visits': 4, 'avg_future_reward': 19.5, 'win_rate': 0.75}),
#  ('4d', {'visits': 33, 'avg_future_reward': 19.2, 'win_rate': 0.758}),
#  ('3h', {'visits': 1, 'avg_future_reward': 18.0, 'win_rate': 1.0})]
```

### 5.3 使用Context版本

```python
# 加载Context版本
kg_context = DecisionKnowledgeGraph.load('cache/knowledge_graph/kg_context_5.pkl')

# 当前状态和历史
current_state = 10
history = ['4d', '1b', '0c', '4a']  # 最近4个动作

# 查询
top_actions = kg_context.get_top_k_actions(
    state=current_state,
    history=history,
    k=5
)

# 获取相似状态
similar_states = kg_context.get_similar_states(current_state, k=10)
```

### 5.4 评估决策系统

```python
from scripts.evaluate_decision_quality import evaluate_decision_quality

# 评估
results = evaluate_decision_quality(
    agent=adaptive_agent,
    knowledge_graph=kg,
    test_states=test_states
)

# 生成报告
generate_report(results, output_path='output/evaluation/decision_quality_20260321.md')
```

---

## 6. 预期效果

### 6.1 知识图谱

**优势**:
- ✅ 提供多样性的高质量动作推荐
- ✅ 包含动作质量评估（win_rate, future_reward）
- ✅ 可解释（基于实际数据）
- ✅ 支持上下文感知

**局限**:
- ⚠️ 数据覆盖度问题（某些state-action对样本不足）
- ⚠️ Context版本可能有稀疏性问题
- ⚠️ 不考虑状态转移的长期影响

### 6.2 相似性编码器

**优势**:
- ✅ 快速推理 (从O(n)到O(1))
- ✅ 泛化能力强
- ✅ 利用BK-Tree结构信息

**局限**:
- ⚠️ 预测误差（需要验证可接受范围）
- ⚠️ 训练需要额外时间

### 6.3 评估系统

**优势**:
- ✅ 多维度评估（质量、覆盖、多样性）
- ✅ 生成详细报告
- ✅ 支持模型对比

**预期改进**:
- 🔍 发现当前模型的弱点
- 🔍 指导后续优化方向
- 🔍 建立评估基准

---

## 7. 后续优化方向

### 7.1 短期优化 (Phase 2)

1. **改进训练目标**
   - 从单标签分类改为多标签分布学习
   - 添加多样性损失
   - 考虑动作质量加权

2. **集成到决策系统**
   - AdaptiveDecisionAgent使用知识图谱
   - 实时推荐 + 质量评估
   - 置信度引导的fallback机制

### 7.2 长期优化 (Phase 3+)

1. **在线学习**
   - 实时更新知识图谱
   - 从新数据中学习

2. **多步推理**
   - 考虑状态转移序列
   - 优化长期回报

3. **集成强化学习**
   - 使用知识图谱指导探索
   - 加速收敛

---

## 8. 风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| 数据覆盖度不足 | 知识图谱稀疏 | 中 | 使用相似状态聚合；编码器泛化 |
| Context版本效果不佳 | 精度没有提升 | 中 | 对比实验；动态调整window |
| 编码器预测误差大 | 相似状态不准确 | 低 | 增加训练数据；调整架构 |
| 评估指标无效 | 无法有效评估 | 低 | 人工验证；多指标综合 |

---

## 9. 总结

### 核心创新

1. **重新定义问题**: 从分类问题改为推荐问题
2. **质量导向**: 使用future_reward和win_rate作为质量指标
3. **多样性保证**: 提供top-k推荐而非单一预测
4. **上下文感知**: 支持Simple和Context两个版本
5. **高效推理**: 使用编码器替代distance_matrix

### 预期收益

- 🎯 更合理的评估指标
- 🎯 更高质量的决策推荐
- 🎯 更好的可解释性
- 🎯 为后续优化提供方向

---

**文档版本**: v1.0  
**最后更新**: 2026-03-21  
**下一步**: 执行 Phase 1.1 - 构建基础知识图谱
