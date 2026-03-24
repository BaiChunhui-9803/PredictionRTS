# AIIDE 2026 投稿准备清单

## 投稿目标

**会议**: AIIDE 2026  
**主题**: New Grounds - 探索游戏AI的新问题设置、应用和场景  
**核心论点**: RTS状态具有几何结构，距离感知的序列建模能提升预测和决策质量

---

## 一、项目主要创新

### 1.1 核心技术创新

| 创新点 | 技术细节 | 文件位置 | 创新程度 |
|--------|---------|---------|---------|
| 状态几何离散化 | BK-Tree + 聚类 + 自定义距离（坐标+生命值+匈牙利匹配） | `structure/` | ★★★☆☆ |
| 距离诱导 Embedding | MDS 初始化，使表征空间与状态距离一致 | `utils/model_utils.py` | ★★★☆☆ |
| SGTransformer | 注意力机制中引入外部 `sim_bias`，可学习缩放 | `models/SGTransformer.py` | ★★★★☆ |
| 空间先验预测 | 自回归预测时用高斯先验偏向几何近邻 | `train/train_state_seq_prediction.py` | ★★★★☆ |
| RTS 场景 DT | Decision Transformer 应用于离散 RTS 状态 | `models/DecisionTransformer.py` | ★★★☆☆ |
| Fitness Landscape | DTW + MDS + 插值生成可视化景观 | `structure/fitness_landscpae/` | ★★☆☆☆ |

### 1.2 核心代码片段

**SGTransformer 注意力偏置**:
```python
# models/SGTransformer.py
self.bias_scale = nn.Parameter(torch.ones(1) * 0.5)
combined_mask = attn_mask + (sim_bias * self.bias_scale)
```

**空间先验引导预测**:
```python
# train/train_state_seq_prediction.py
spatial_prior = torch.exp(-dists ** 2 / (2 * 1.5 ** 2))
probs[:len(dists)] *= spatial_prior
```

**MDS 初始化 Embedding**:

```python
# utils/model_utils.py
mds = MDS(n_components=embedding_dim, dissimilarity='precomputed')
weights = mds.fit_transform(norm_dm)
model.embedding.weight[:num_states].copy_(torch.from_numpy(weights))
```

---

## 二、AIIDE 2026 "New Grounds" 契合度

### 2.1 主题匹配分析

| AIIDE 2026 主题要求 | 契合度 | 说明 |
|-------------------|--------|------|
| New problem settings | ★★★★☆ | RTS 状态离散化 + 距离感知序列建模 |
| New applications | ★★★☆☆ | DT 应用于 RTS 离线数据 |
| New contexts | ★★★☆☆ | 将几何结构引入 Transformer |
| Extensions to new domains | ★★★★☆ | 离线 RL/序列方法扩展到 RTS 微操 |

### 2.2 "New Grounds" 卖点

1. **新问题视角**: 不是直接端到端学习，而是状态离散化 + 几何距离 + Transformer
2. **方法扩展**: Decision Transformer 从传统 RL 扩展到 RTS 微操场景
3. **跨领域融合**: 游戏 AI + 离线强化学习 + 几何建模

---

## 三、必须完成的改进 (高优先级)

### 3.1 增加 Baseline 对比 [重要程度: ⭐⭐⭐⭐⭐]

**状态**: ❌ 未完成

**需要对比的方法**:

- [ ] Vanilla Transformer (无 MDS、无 SG、无先验)
- [ ] LSTM / GRU 序列模型
- [ ] 标准 Decision Transformer (无 MDS 初始化)
- [ ] BC (Behavior Cloning)
- [ ] 其他离线 RL 方法 (如 CQL, IQL)

**实现建议**:
```
1. 在 train/ 目录下创建 train_baseline.py
2. 统一评估接口，确保公平对比
3. 使用相同的训练/测试集划分
4. 记录训练曲线和最终性能
```

### 3.2 完善 Ablation Study [重要程度: ⭐⭐⭐⭐⭐]

**状态**: ❌ 未完成

**消融实验清单**:

| 组件 | 对比项 | 预期结论 |
|------|--------|---------|
| MDS 初始化 | 随机初始化 vs MDS初始化 | MDS 应提升收敛速度和最终性能 |
| SG 注意力偏置 | 无偏置 vs 有偏置 | 距离引导应提升预测准确率 |
| 空间先验 | 无先验 vs 高斯先验 | 先验应提升近邻状态预测 |
| 偏置缩放因子 | 固定 vs 可学习 | 可学习应更灵活 |

**参数配置**:
```python
# 需要测试的 params 组合
configs = [
    {"mdl_init_embedding_train": False, "mdl_attn_sim_bias": False, "mdl_spatial_prior": False},  # Baseline
    {"mdl_init_embedding_train": True,  "mdl_attn_sim_bias": False, "mdl_spatial_prior": False},  # +MDS
    {"mdl_init_embedding_train": True,  "mdl_attn_sim_bias": True,  "mdl_spatial_prior": False},  # +SG
    {"mdl_init_embedding_train": True,  "mdl_attn_sim_bias": True,  "mdl_spatial_prior": True},   # Full
]
```

### 3.3 丰富评估指标 [重要程度: ⭐⭐⭐⭐]

**状态**: ⚠️ 部分完成

**当前指标**:
- ✅ 预测步距离误差
- ✅ DT 动作准确率

**需要增加**:
- [ ] 按步预测准确率 (Top-1, Top-3, Top-5)
- [ ] 策略质量评估 (在 PySC2 环境中执行预测动作)
- [ ] 胜率/得分提升 (与随机/贪婪策略对比)
- [ ] Fitness 相关性分析 (预测轨迹与高 fitness 轨迹的相似度)
- [ ] 计算效率对比 (训练时间、推理速度)

**评估代码位置**: 
- `train/train_state_seq_prediction.py`: evaluate_test_performance()
- `train/train_decision_transformer.py`: evaluate_action_accuracy_detailed()

### 3.4 强化故事线和贡献声明 [重要程度: ⭐⭐⭐⭐]

**状态**: ❌ 未完成

**核心论点草稿**:
```
RTS 游戏状态具有内在的几何结构（单位位置、生命值分布），
传统序列建模方法忽略了这种结构信息。本文提出距离感知的
Transformer 序列建模方法，通过：(1) MDS 初始化使 embedding
空间与状态距离一致；(2) 注意力偏置引导模型关注相似状态；
(3) 空间先验使预测偏向几何近邻。实验表明该方法在 RTS 
状态预测和决策任务上优于传统方法。
```

**Contribution Statement**:
1. 提出基于 BK-Tree 和自定义距离的 RTS 状态离散化方法
2. 设计 SGTransformer，将状态距离信息注入注意力机制
3. 提出空间先验引导的自回归预测策略
4. 在 StarCraft II 微操场景上验证方法有效性

---

## 四、可选增强项 (中优先级)

### 4.1 数据和实验增强

- [ ] 多地图验证 (当前仅 MarineMicro_MvsM_4)
- [ ] 多场景测试 (不同单位数量、不同初始距离)
- [ ] 与人类玩家数据对比
- [ ] 数据规模分析 (轨迹数量对性能的影响)

### 4.2 可视化增强

- [ ] 预测轨迹 vs 实际轨迹对比图
- [ ] Embedding 空间可视化 (t-SNE/UMAP)
- [ ] 注意力热力图分析
- [ ] 训练过程动画

### 4.3 工程改进

- [ ] 引入 argparse 或 Hydra 参数管理
- [ ] 实验结果自动记录 (wandb/tensorboard)
- [ ] 提供 Docker 环境或 requirements.txt
- [ ] 整理 README 和运行脚本

---

## 五、论文结构建议

### 5.1 章节规划

```
1. Introduction
   - RTS AI 的挑战
   - 离线强化学习与序列建模
   - 本文贡献

2. Related Work
   - 游戏 AI 与 RTS 微操
   - Transformer 在序列建模中的应用
   - Decision Transformer 与离线 RL
   - 状态表示与距离度量

3. Method
   3.1 问题形式化
   3.2 状态离散化与距离度量
   3.3 距离诱导的 Embedding 初始化
   3.4 Similarity-Guided Transformer
   3.5 空间先验预测策略
   3.6 Decision Transformer 适配

4. Experiments
   4.1 实验设置 (PySC2, 地图, 数据)
   4.2 Baseline 对比
   4.3 Ablation Study
   4.4 可视化分析

5. Conclusion
```

### 5.2 图表规划

| 图表 | 内容 | 优先级 |
|------|------|--------|
| Fig 1 | 方法整体框架图 | 高 |
| Fig 2 | 状态离散化流程 | 高 |
| Fig 3 | SGTransformer 架构 | 高 |
| Fig 4 | Baseline 对比结果 | 高 |
| Fig 5 | Ablation 结果 | 高 |
| Fig 6 | Fitness Landscape 可视化 | 中 |
| Fig 7 | 预测轨迹案例 | 中 |
| Fig 8 | 注意力可视化 | 低 |

---

## 六、时间线建议

假设投稿截止日期前有 **8-10 周**:

| 阶段 | 时间 | 任务 | 产出 |
|------|------|------|------|
| 第1-2周 | Week 1-2 | Baseline 实现 | 对比实验代码 |
| 第3-4周 | Week 3-4 | Ablation Study | 消融实验结果 |
| 第5周 | Week 5 | 评估指标完善 | 完整评估报告 |
| 第6周 | Week 6 | 多场景验证 | 泛化性分析 |
| 第7周 | Week 7 | 可视化与图表 | 论文用图 |
| 第8周 | Week 8 | 论文初稿 | 完整初稿 |
| 第9-10周 | Week 9-10 | 修改润色 | 最终投稿版本 |

---

## 七、当前项目状态

### 7.1 已完成 ✅

- [x] 数据采集 Pipeline (PySC2 + SmartAgent)
- [x] BK-Tree 状态离散化
- [x] 自定义状态距离计算
- [x] TrajectoryTransformer 实现
- [x] SGTransformer 实现
- [x] DecisionTransformer 实现
- [x] MDS Embedding 初始化
- [x] 空间先验预测
- [x] Fitness Landscape 可视化
- [x] 基础评估指标

### 7.2 待完成 ❌

- [ ] Baseline 对比实验
- [ ] Ablation Study
- [ ] 完整评估指标
- [ ] 多场景验证
- [ ] 论文撰写

### 7.3 代码改进建议

**参数管理**:
```python
# 当前: config/base_config.py (Python 变量)
# 建议: 引入 argparse 或 Hydra

# 示例 Hydra 配置
# config/experiment.yaml
model:
  d_model: 128
  nhead: 8
  num_layers: 4
training:
  epochs: 20
  batch_size: 32
  lr: 1e-4
ablation:
  use_mds_init: true
  use_sim_bias: true
  use_spatial_prior: true
```

---

## 八、投稿前最终检查

### 8.1 技术检查

- [ ] 所有实验可复现 (固定随机种子)
- [ ] 代码有详细注释
- [ ] 提供运行脚本和文档
- [ ] 数据和模型权重可访问

### 8.2 论文检查

- [ ] 贡献声明清晰
- [ ] 实验充分支持结论
- [ ] 图表清晰专业
- [ ] 参考文献完整
- [ ] 符合 AIIDE 格式要求

### 8.3 补充材料

- [ ] 代码仓库链接
- [ ] 补充实验细节
- [ ] 额外可视化结果

---

## 九、总结

### 投稿可行性评估

| 维度 | 当前评分 | 目标评分 | 差距 |
|------|---------|---------|------|
| 技术创新 | 7/10 | 8/10 | 需强化故事线 |
| 实验完整性 | 5/10 | 8/10 | 需增加 Baseline 和 Ablation |
| 论文准备度 | 4/10 | 8/10 | 需完成撰写 |

### 结论

**当前状态**: 有技术亮点，但实验不充分，不建议直接投稿

**建议**: 

- 短期 (2-3周): 完成 Baseline 对比和 Ablation Study
- 中期 (1-2月): 完善评估和多场景验证后投稿 AIIDE 2026
- 备选: 先投 Workshop 或 Poster，积累反馈后再投主会

---

*文档创建时间: 2026-03-18*  
*最后更新: 2026-03-18*
