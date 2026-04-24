# PredictionRTS

基于 **Experience Transition Graph (ETG)** 的《星际争霸2》微操作决策系统。

从 RL 训练轨迹中构建经验转移图，利用图上的束搜索（Beam Search）与链式推演（Chain Rollout）进行在线规划决策，并通过 Optuna 贝叶斯优化自动搜索最优参数。无需训练神经网络即可实现可解释的微操作策略。

## 核心思路

```mermaid
flowchart LR
    A["RL 对局采集"] --> B["BK-Tree<br/>两级状态聚类"]
    B --> C["构建 ETG<br/>统计转移概率"]
    C --> D["束搜索规划<br/>在线决策"]
    D --> E["结果分析<br/>+ 参数寻优"]

    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#e8f5e9
    style D fill:#fce4ec
    style E fill:#f3e5f5
```

1. **数据采集** — 在 SC2 中运行 Q-Learning Agent 采集对战轨迹
2. **状态抽象** — BK-Tree 两级聚类（空间坐标 + HP 分布）将连续状态离散化
3. **ETG 构建** — 统计 (状态, 动作) 对的访问次数、胜率、质量评分，构建带权有向图
4. **图上规划** — Beam Search 多步前瞻 + Chain Rollout 滚动推演，实时生成决策
5. **自动寻优** — Optuna TPE 采样自动搜索最优束搜索参数组合

## 系统架构

系统由两条完全解耦的流水线组成：

| 流水线 | 说明 | 入口 |
|--------|------|------|
| **离线数据管道** | 采集 → 聚类 → ETG 构建 → 距离矩阵 | `build_from_collected.py` |
| **在线实时系统** | SC2 进程 + FastAPI Bridge + Streamlit UI（3 进程） | `run_live_game.py` |

详细架构文档见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## Web 可视化功能

```bash
streamlit run scripts/visualize_kg_web.py
```

| Tab | 功能 | 说明 |
|-----|------|------|
| 项目介绍 | README 展示 | 本文件 |
| 转移图可视化 | ETG 交互式浏览 | PyVis 力导向布局、聚焦模式、质量过滤、终端高亮 |
| 束搜索规划 | 离线 Beam Search | 6 种评分策略、路径树可视化、片段匹配、CompositeScore 排名 |
| 滚动推演 | Chain Rollout 模拟 | 单步/多步推演、BTB 备选路径切换、推演树全局回溯 |
| 原始数据 | Episode 数据探索 | MDS 降维地形图、Episode 查询、状态分布统计 |
| 实时对局 | 在线控制 SC2 | 4 种自动模式、帧级 plan 对齐、event_type 着色、SSE 事件流 |
| 结果分析 | 对局记录管理 | 多组对比图表、参数化文件名、核心参数图例 |
| 参数寻优 | Optuna 贝叶斯优化 | 一键启动、优化曲线、参数重要性、相关性矩阵、运行监测 |

## 项目结构

```
PredictionRTS/
├── configs/                         # YAML 配置文件
│   ├── config.yaml                  #   Hydra 主配置入口
│   ├── kg_catalog.yaml              #   ETG 目录（9 条记录）
│   ├── learner_config.yaml          #   参数寻优搜索空间
│   ├── env/starcraft.yaml           #   6 个地图配置
│   ├── model/                       #   模型配置 (DT / SG / Trajectory Transformer)
│   └── experiment/                  #   实验配置
│
├── src/                             # 核心 Python 包
│   ├── decision/                    #   ETG + 束搜索 + 链式推演
│   │   ├── knowledge_graph.py       #     DecisionKnowledgeGraph 构建与查询
│   │   ├── kg_beam_search.py        #     Beam Search 前瞻搜索
│   │   ├── chain_rollout.py         #     Chain Rollout 多步推演 + BTB
│   │   ├── beam_matcher.py          #     Beam 路径片段匹配
│   │   ├── kg_decision_helper.py    #     决策辅助接口
│   │   ├── model_pool.py            #     多模型池管理
│   │   ├── strategy_router.py       #     自适应策略路由
│   │   ├── selector.py              #     DT+Q 混合动作选择
│   │   └── evaluator.py             #     决策质量评估
│   ├── structure/                   #   BK-Tree + 距离计算
│   │   ├── BKTree_sc2.py            #     SC2 专用 BK-Tree (两级聚类)
│   │   ├── custom_distance_sc2.py   #     匈牙利匹配分布距离
│   │   ├── state_distance.py        #     状态距离函数
│   │   └── generate_FL.py           #     适应度景观生成
│   ├── sc2env/                      #   SC2 游戏环境
│   │   ├── agent.py                 #     基础 Agent (SmartAgent)
│   │   ├── kg_guided_agent.py       #     KG 引导 Agent (在线决策)
│   │   ├── bridge_server.py         #     FastAPI Bridge Server (REST API)
│   │   ├── bridge.py                #     GameBridge (6 Queue + 2 Event)
│   │   ├── run_game.py              #     游戏运行入口
│   │   ├── replay_collector.py      #     回放数据采集器
│   │   ├── config.py                #     地图参数配置
│   │   └── utils.py                 #     状态编码/动作解析
│   ├── models/                      #   神经网络模型
│   │   ├── DecisionTransformer.py   #     DT (GPT, RTG-State-Action 交织)
│   │   ├── QNetwork.py              #     Q(s,a) 估值网络
│   │   └── StateTransitionPredictor.py  # 状态转移预测
│   ├── data/                        #   数据加载器
│   ├── config/                      #   实验配置
│   ├── train/                       #   训练逻辑
│   ├── algorithms/                  #   模式挖掘
│   ├── utils/                       #   通用工具
│   └── visualization/               #   KG 可视化器
│
├── scripts/                         # 可执行脚本
│   ├── build_from_collected.py      #   从采集数据构建 ETG
│   ├── deploy_augmented.py          #   部署增强 ETG
│   ├── run_live_game.py             #   实时对局启动器
│   ├── parameter_learner.py         #   Optuna 参数寻优
│   ├── simulate_kg_trajectory.py    #   ETG 轨迹模拟
│   ├── simulate_decision_process.py #   决策过程模拟
│   ├── evaluate_decision_quality.py #   决策质量评估
│   ├── run_train_dt.py              #   DT 训练
│   ├── run_train_q_network.py       #   Q-Network 训练
│   ├── run_decision_system.py       #   决策系统全流程
│   ├── run_sc2_optimal_agent.py     #   最优决策 Agent 离线评估
│   ├── visualize_kg_web.py          #   Streamlit Web 入口
│   └── kg_web/                      #   Web UI 模块
│       ├── loaders.py               #     数据加载 (带 st.cache)
│       ├── graph_builder.py         #     过滤子图构建
│       ├── pyvis_renderer.py        #     PyVis 网络图渲染
│       ├── beam_utils.py            #     Beam 路径评分/排名
│       ├── raw_data_utils.py        #     MDS 降维 + 地形图
│       ├── live_game_html.py        #     实时对局 HTML/CSS/JS SPA
│       ├── viz_tab.py               #     转移图可视化 Tab
│       ├── prediction_tab.py        #     束搜索规划 Tab
│       ├── rollout_tab.py           #     滚动推演 Tab
│       ├── raw_data_tab.py          #     原始数据 Tab
│       ├── live_game_tab.py         #     实时对局 Tab
│       ├── results_tab.py           #     结果分析 Tab
│       └── learner_tab.py           #     参数寻优 Tab
│
├── data/                            # 训练数据 (6 个 SC2 地图)
├── cache/
│   ├── knowledge_graph/             #   ETG pickle 文件
│   └── npy/                         #   距离矩阵缓存
├── output/
│   ├── collected_data/              #   采集的对局数据
│   ├── game_results/                #   对局记录 JSON
│   └── learner_results/             #   参数寻优结果 (study.db + runs/)
├── tex/                             #   论文 LaTeX 与技术文档
├── doc/                             #   项目文档
├── ARCHITECTURE.md                  #   系统架构详细文档
└── README.md                        #   本文件
```

## 快速开始

### 环境要求

- Python 3.8+
- StarCraft II (PySC2)
- 依赖见 `requirements.txt`

### 安装依赖

```bash
pip install -r requirements.txt
pip install optuna streamlit plotly pyyaml numpy requests
```

### 启动 Web 可视化

```bash
streamlit run scripts/visualize_kg_web.py
```

### 构建 ETG

从采集的对局数据构建：

```bash
python scripts/build_from_collected.py \
    --input output/collected_data/<data_dir> \
    --bktree-dir output/collected_data/<data_dir> \
    --output-dir cache/knowledge_graph/<kg_name>
```

部署增强 ETG：

```bash
python scripts/deploy_augmented.py \
    --collected-dir output/collected_data/<data_dir> \
    --kg-dir cache/knowledge_graph/<kg_name>
```

### 实时对局

```bash
python scripts/run_live_game.py --mode all \
    --map_key sce-1 \
    --kg_file <kg_name>/kg_simple.pkl \
    --data_dir data/<map_id>/<data_id>
```

### 参数寻优

Web UI 一键启动（推荐），或命令行：

```bash
python scripts/parameter_learner.py \
    --trials 50 --episodes 100 \
    --kg_file <kg_name>/kg_simple.pkl \
    --resume
```

## 支持的地图

| 配置键 | 地图名 | 单位数 | 说明 |
|--------|--------|--------|------|
| `sce-1` | `local_enemy_test_1` | 4 vs 4 | 基础地图 |
| `sce-1m` | `local_enemy_test_1_mirror` | 4 vs 4 | 镜像版 |
| `sce-2` | `MarineMicro_MvsM_4_dist` | 4 vs 4 | 有距离差异 |
| `sce-2m` | `MarineMicro_MvsM_4_dist_mirror` | 4 vs 4 | 距离差异镜像 |
| `sce-3` | `MarineMicro_MvsM_8_far` | 8 vs 8 | 大规模远距离 |
| `sce-3m` | `MarineMicro_MvsM_8_far_mirror` | 8 vs 8 | 大规模镜像 |

## 关键概念

### Experience Transition Graph (ETG)

从 RL 训练轨迹构建的带权有向图：
- **节点** = 抽象状态（由 BK-Tree 两级聚类得到）
- **边** = 动作 + 统计量（访问次数、胜率、质量评分、累积奖励）

核心类：`src/decision/knowledge_graph.py` → `DecisionKnowledgeGraph`

### BK-Tree 两级聚类

将连续状态空间离散化为 `(primary_id, secondary_id)` 对：
- **Primary** — 基于空间坐标的 BK-Tree 聚类（5 级：`k_means_000` ~ `k_means_100`）
- **Secondary** — 每个 Primary 内基于 HP 分布的 BK-Tree 聚类

### 动作编码

格式：`{cluster_index}{action_letter}`，共 55 种组合

| 字母 | 动作 |
|------|------|
| `a` | `ATK_nearest` — 攻击最近敌人 |
| `b` | `ATK_clu_nearest` — 聚类内攻击最近 |
| `c` | `ATK_nearest_weakest` — 攻击最近最弱 |
| `d` | `ATK_clu_nearest_weakest` — 聚类内攻击最近最弱 |
| `e` | `ATK_threatening` — 攻击威胁最高 |
| `f` | `DEF_clu_nearest` — 防御：聚类内最近 |
| `g-k` | 混合/随机/无操作 |

示例：`"1c"` = `k_means_025` 聚类 + `ATK_nearest_weakest`

### Beam Search 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `score_mode` | `quality` | 评分策略：`quality` / `future_reward` / `win_rate` |
| `beam_width` | 3 | 每步保留的候选路径数 |
| `lookahead_steps` | 5 | 单次搜索深度 |
| `action_strategy` | `best_beam` | 路径选择策略（6 种） |
| `min_visits` | 1 | 最低访问次数过滤 |
| `min_cum_prob` | 0.01 | 累积概率剪枝阈值 |
| `discount_factor` | 0.9 | 累积概率步数折扣 |
| `max_state_revisits` | 2 | 同路径状态最大重复次数 |
| `epsilon` | 0.1 | ε-greedy 探索率 |
| `enable_backup` | `False` | 启用 BTB 备选路径 |

## 配置文件

| 文件 | 说明 |
|------|------|
| `configs/config.yaml` | Hydra 主配置入口（模型/环境/实验） |
| `configs/kg_catalog.yaml` | ETG 目录（9 条记录，含地图/类型/路径） |
| `configs/learner_config.yaml` | 参数寻优：搜索空间 + 目标权重 + 执行参数 |
| `configs/env/starcraft.yaml` | 6 个地图配置 + 环境参数 + 算法参数 |
| `configs/paths.yaml` | 外部数据路径（机器特定，需本地配置） |

## 数据文件格式

| 文件 | 格式 | 内容 |
|------|------|------|
| `kg_simple.pkl` | Pickle dict | `{(state_id, action_code): {visits, win_rate, quality_score, ...}}` |
| `kg_simple_transitions.pkl` | Pickle dict | `{(state, action): {next_state: probability}}` |
| `state_node.txt` | Text | `(primary, secondary) → node_id` 映射表 |
| `primary_bktree.json` | JSON | Primary BK-Tree 序列化 |
| `state_distance_matrix_*.npy` | NumPy | 状态间匈牙利距离矩阵 |

## 详细文档

| 文档 | 说明 |
|------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 系统架构、模块依赖、数据流、接口速查 |
| `tex/技术细节/` | 技术细节文档（参数学习器设计方案等） |
| `doc/` | KG 构建指南、使用手册等 |

## 依赖

- Python 3.8+
- PyTorch 1.10+
- PySC2（数据采集，可选）
- Optuna（参数寻优）
- Streamlit + Plotly（Web 可视化）
- NumPy, PyYAML, Requests

## License

MIT
