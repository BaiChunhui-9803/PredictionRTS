# PredictionRTS 使用指南

> **版本**: v0.1.0  
> **最后更新**: 2026-03

---

## 目录

1. [项目简介](#1-项目简介)
2. [环境配置](#2-环境配置)
3. [项目结构](#3-项目结构)
4. [快速开始](#4-快速开始)
5. [模块详解](#5-模块详解)
6. [配置说明](#6-配置说明)
7. [数据格式](#7-数据格式)
8. [脚本使用](#8-脚本使用)
9. [API参考](#9-api参考)
10. [常见问题](#10-常见问题)

---

## 1. 项目简介

### 1.1 项目目标

**PredictionRTS** 是一个基于深度学习的即时战略游戏（RTS）行为预测系统。该项目利用 StarCraft II 游戏环境收集的数据，通过 Transformer 架构对玩家行为序列进行建模和预测。

### 1.2 核心技术

```
┌─────────────────────────────────────────────────────────────────┐
│                    PredictionRTS 系统架构                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │  StarCraft  │───▶│   BK-Tree   │───▶│   模型训练   │          │
│  │     II      │    │  状态聚类    │    │              │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │  数据收集     │    │  状态编码    │    │  行为预测     │          │
│  │  (pysc2)    │    │  (聚类ID)    │    │  (DT/TT)    │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| 技术组件 | 说明 |
|---------|------|
| **BK-Tree** | 用于游戏状态聚类，将连续状态空间映射到离散状态ID |
| **Decision Transformer** | 基于回报条件的行为序列预测模型 |
| **Trajectory Transformer** | 轨迹序列预测模型 |
| **StarCraft II (pysc2)** | 游戏环境，提供数据采集接口 |

### 1.3 适用场景

- 游戏AI行为建模
- 玩家策略预测
- 强化学习离线数据处理
- 序列决策研究

---

## 2. 环境配置

### 2.1 系统要求

- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.9+
- **CUDA**: 11.0+ (GPU训练推荐)
- **内存**: 16GB+ (处理大规模数据集)

### 2.2 依赖安装

```bash
# 克隆项目
git clone <repository_url>
cd PredictionRTS

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2.3 核心依赖

```
torch>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
pyyaml>=6.0
pandas>=1.3.0
pysc2>=4.0.0        # StarCraft II 接口（可选）
omegaconf>=2.3.0    # 配置管理（可选）
```

### 2.4 StarCraft II 安装（可选）

如需采集新数据，需安装 StarCraft II 和 pysc2：

```bash
# 1. 下载并安装 StarCraft II
# Windows: 从 Battle.net 下载
# Linux: 参考 pysc2 官方文档

# 2. 安装 pysc2
pip install pysc2

# 3. 下载迷你游戏地图
python -m pysc2.bin.download_maps
```

---

## 3. 项目结构

### 3.1 目录树

```
PredictionRTS/
├── configs/                    # 配置文件目录
│   ├── config.yaml            # 主配置文件
│   ├── paths.yaml             # 路径配置
│   ├── model/                 # 模型配置
│   │   ├── decision_transformer.yaml
│   │   ├── sg_transformer.yaml
│   │   └── trajectory_transformer.yaml
│   ├── env/                   # 环境配置
│   │   └── starcraft.yaml
│   └── experiment/            # 实验配置
│       └── default.yaml
│
├── src/                       # 源代码目录
│   ├── __init__.py           # 包初始化，提供 get_config(), set_seed()
│   ├── config/               # 配置模块
│   │   └── base_config.py    # 基础配置参数
│   ├── data/                 # 数据处理模块
│   │   ├── loader.py         # 数据加载器（推荐使用）
│   │   ├── load_data.py      # 传统数据加载
│   │   └── global_variable.py # 全局变量
│   ├── models/               # 模型定义
│   │   ├── DecisionTransformer.py  # Decision Transformer
│   │   ├── Transformer.py    # Trajectory Transformer
│   │   └── SGTransformer.py  # Spatial-Guided Transformer
│   ├── structure/            # 数据结构
│   │   ├── bk_tree.py        # BK-Tree 实现
│   │   ├── state_distance.py # 状态距离计算
│   │   └── custom_distance.py # 自定义距离函数
│   ├── train/                # 训练模块
│   │   ├── trainer.py        # 基础训练器
│   │   ├── train_decision_transformer.py
│   │   └── train_state_seq_prediction.py
│   ├── env/                  # StarCraft II 环境
│   │   ├── starcraft.py      # 环境主程序
│   │   ├── run_loop.py       # 运行循环
│   │   ├── env_conf.py       # 环境配置
│   │   └── agents/           # 智能体
│   │       ├── Agent.py
│   │       └── SmartAgent.py
│   ├── utils/                # 工具函数
│   │   ├── path_utils.py     # 路径工具
│   │   ├── load_utils.py     # 加载工具
│   │   ├── metrics.py        # 评估指标
│   │   └── visualization.py  # 可视化
│   ├── plot/                 # 绘图模块
│   │   ├── plot_dt.py        # DT 结果绘图
│   │   └── plot_prediction.py # 预测结果绘图
│   └── algorithms/           # 算法模块
│       └── pattern_analysis.py
│
├── scripts/                   # 运行脚本
│   ├── run_train.py          # 训练脚本
│   ├── run_evaluate.py       # 评估脚本
│   ├── run_predict.py        # 预测脚本
│   └── run_collect_data.py   # 数据采集脚本
│
├── data/                      # 数据目录（外部数据）
├── cache/                     # 缓存目录（模型权重等）
├── output/                    # 输出目录（日志、图表）
├── doc/                       # 文档目录
│   └── USAGE.md              # 本文档
│
├── requirements.txt           # 依赖列表
└── README.md                  # 项目说明
```

### 3.2 核心模块说明

| 模块 | 路径 | 功能 |
|------|------|------|
| 配置管理 | `src/config/`, `src/__init__.py` | 加载和管理配置 |
| 数据加载 | `src/data/loader.py` | 懒加载数据、预处理 |
| 模型定义 | `src/models/` | Transformer 架构定义 |
| BK-Tree | `src/structure/bk_tree.py` | 状态聚类树结构 |
| 训练器 | `src/train/` | 模型训练逻辑 |
| 环境接口 | `src/env/` | StarCraft II 交互 |

---

## 4. 快速开始

### 4.1 完整训练流程

```bash
# 1. 配置数据路径（修改 configs/paths.yaml）
# data_root: "your/data/path"
# map_id: "MarineMicro_MvsM_4"
# data_id: "6"

# 2. 运行训练
python scripts/run_train.py

# 3. 评估模型
python scripts/run_evaluate.py

# 4. 行为预测
python scripts/run_predict.py
```

### 4.2 Python API 使用示例

```python
from src import get_config, set_seed
from src.data.loader import DataLoader
from src.models.DecisionTransformer import DecisionTransformer

# 加载配置
cfg = get_config()
set_seed(cfg.get("seed", 42))

# 加载数据
data_loader = DataLoader(cfg)
data_loader.load_all()  # 加载所有数据

# 获取数据
state_log = data_loader.state_log      # List[List[int]]
action_log = data_loader.action_log    # List[List[str]]
r_log = data_loader.r_log              # List[List[float]]
dt_data = data_loader.dt_data          # 预处理后的DT数据

# 创建模型
state_dim = len(data_loader.state_node_dict)
act_vocab_size = len(data_loader.action_vocab)

model = DecisionTransformer(
    state_dim=state_dim,
    act_vocab_size=act_vocab_size,
    n_layer=4,
    n_head=4,
    n_embd=128,
    max_len=100
)

# 模型使用...
```

### 4.3 预期输出

训练脚本执行成功后，输出示例：

```
2026-03-20 17:42:22 - INFO - Training DecisionTransformer...
2026-03-20 17:42:22 - INFO - Config: {'seed': 42, 'device': 'cuda', ...}
2026-03-20 17:42:23 - INFO - Model created: DecisionTransformer
2026-03-20 17:42:23 - INFO - Output paths: {'models': ...}
2026-03-20 17:42:23 - INFO - Training setup completed. Model ready for training.
```

---

## 5. 模块详解

### 5.1 配置模块 (`src/config/`, `src/__init__.py`)

#### 功能

- `get_config()`: 从 YAML 文件加载配置，返回字典
- `set_seed(seed)`: 设置全局随机种子

#### 使用示例

```python
from src import get_config, set_seed, ROOT_DIR, CONFIG_DIR

cfg = get_config()
print(cfg)
# {'defaults': [...], 'seed': 42, 'device': 'cuda', 'mode': 'train', 'paths': {...}}

set_seed(42)

# 常用路径常量
print(ROOT_DIR)    # 项目根目录
print(CONFIG_DIR)  # 配置目录
print(DATA_DIR)    # 数据目录
```

#### 配置访问方式

```python
# 推荐：使用 get() 方法安全访问
seed = cfg.get("seed", 42)
device = cfg.get("device", "cuda")

# 嵌套访问
paths = cfg.get("paths", {})
data_root = paths.get("data_root", "default/path")

# 模型配置
model_cfg = cfg.get("model", {})
n_layer = model_cfg.get("n_layer", 4)
```

---

### 5.2 数据模块 (`src/data/`)

#### DataLoader 类

主要的数据加载器，支持懒加载和缓存。

**位置**: `src/data/loader.py`

```python
class DataLoader:
    def __init__(self, cfg):
        """
        初始化数据加载器
        
        Args:
            cfg: 配置字典，需包含 'paths' 键
        """
    
    # 懒加载属性（首次访问时加载）
    @property
    def primary_bk_tree(self) -> BKTree:
        """主 BK-Tree"""
    
    @property
    def secondary_bk_trees(self) -> Dict[int, BKTree]:
        """次级 BK-Tree 字典"""
    
    @property
    def state_node_dict(self) -> Dict:
        """状态到节点ID的映射"""
    
    @property
    def state_log(self) -> List[List[int]]:
        """状态序列日志"""
    
    @property
    def action_log(self) -> List[List[str]]:
        """动作序列日志"""
    
    @property
    def r_log(self) -> List[List[float]]:
        """奖励序列日志"""
    
    @property
    def dt_data(self) -> Dict:
        """预处理后的 Decision Transformer 数据"""
    
    @property
    def action_vocab(self) -> Dict[str, int]:
        """动作词汇表（动作字符 -> ID）"""
    
    def load_all(self):
        """加载所有数据"""
```

#### 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                       数据加载流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   外部数据文件                      DataLoader 属性              │
│   ─────────────                    ───────────────              │
│                                                                 │
│   primary_bktree.json       ──▶   primary_bk_tree               │
│   secondary_bktree_*.json   ──▶   secondary_bk_trees            │
│   state_node.txt            ──▶   state_node_dict, reverse_dict │
│   node_log.txt              ──▶   state_log                     │
│   action_log.csv            ──▶   action_log                    │
│   sub_episode/*.csv         ──▶   r_log (奖励计算)              │
│                                 ──▶   dt_data (预处理)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 使用示例

```python
from src.data.loader import DataLoader

loader = DataLoader(cfg)

# 懒加载 - 首次访问时加载数据
print(f"Episodes: {len(loader.state_log)}")
print(f"Actions: {loader.action_vocab}")

# 预加载所有数据
loader.load_all()
```

---

### 5.3 模型模块 (`src/models/`)

#### 5.3.1 Decision Transformer

**位置**: `src/models/DecisionTransformer.py`

**架构**:

```
输入: states, actions, rtgs, timesteps
         │
         ▼
┌─────────────────────────────────────┐
│  Embedding Layers                   │
│  ├── embed_state: Embedding(S, D)   │
│  ├── embed_action: Embedding(A, D)  │
│  ├── embed_rtg: Linear(1, D)        │
│  └── embed_timestep: Embedding(T, D)│
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Interleaved Sequence               │
│  [R1, S1, A1, R2, S2, A2, ...]      │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Transformer Encoder                │
│  (n_layer layers, n_head heads)     │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Action Prediction Head             │
│  Linear(D, act_vocab_size)          │
└─────────────────────────────────────┘
```

**使用示例**:

```python
from src.models.DecisionTransformer import DecisionTransformer

model = DecisionTransformer(
    state_dim=1000,        # 状态空间大小
    act_vocab_size=11,     # 动作词汇表大小
    n_layer=4,             # Transformer 层数
    n_head=4,              # 注意力头数
    n_embd=128,            # 嵌入维度
    max_len=100            # 最大序列长度
)

# 前向传播
logits = model(states, actions, rtgs, timesteps)
# logits.shape: (batch_size, seq_len, act_vocab_size)
```

#### 5.3.2 Trajectory Transformer

**位置**: `src/models/Transformer.py`

**使用示例**:

```python
from src.models.Transformer import TrajectoryTransformer

model = TrajectoryTransformer(
    vocab_size=1000,       # 词汇表大小（状态数）
    d_model=128,           # 模型维度
    nhead=8,               # 注意力头数
    num_layers=4,          # 层数
    max_len=512            # 最大序列长度
)

# 前向传播
logits = model(state_sequence)
# logits.shape: (batch_size, seq_len, vocab_size)
```

---

### 5.4 结构模块 (`src/structure/`)

#### 5.4.1 BK-Tree

**位置**: `src/structure/bk_tree.py`

BK-Tree（Burkhard-Keller Tree）是一种用于高效近似最近邻搜索的树结构，用于游戏状态聚类。

```python
from src.structure.bk_tree import BKTree, BKTreeNode, get_max_cluster_id

# 创建树
tree = BKTree()

# 插入节点
tree.insert(state_dict, cluster_id, distance_func)

# 搜索相似节点
results = tree.search(query_state, max_distance=5.0, distance_func=distance_func)

# 获取最大聚类ID
max_id = get_max_cluster_id(tree)
```

#### 5.4.2 状态距离计算

**位置**: `src/structure/state_distance.py`

```python
from src.structure.state_distance import (
    euclidean_distance,     # 欧氏距离
    multi_distance,         # 多维距离（位置+血量）
    custom_distance,        # 自定义状态距离
    hungarian_distance,     # 匈牙利算法距离
    calculate_state_distance_matrix  # 计算距离矩阵
)

# 计算两个状态之间的距离
dist = custom_distance(state1, state2)
```

---

### 5.5 训练模块 (`src/train/`)

#### Trainer 基类

**位置**: `src/train/trainer.py`

```python
from src.train.trainer import Trainer

class MyTrainer(Trainer):
    def train_epoch(self, train_loader) -> float:
        """训练一个 epoch，返回平均损失"""
        pass
    
    def validate(self, val_loader) -> float:
        """验证，返回验证损失"""
        pass

# 使用
trainer = MyTrainer(model, cfg, device="cuda")
trainer.train(train_loader, val_loader, save_dir=Path("cache/model"))
```

---

### 5.6 环境模块 (`src/env/`)

#### StarCraft II 环境接口

**位置**: `src/env/starcraft.py`

```python
from src.env.starcraft import run_single_env, main_run

# 运行单个环境
run_single_env(runner_id=0)

# 运行主程序
main_run()
```

#### 智能体

**位置**: `src/env/agents/`

- `Agent.py`: 基础智能体
- `SmartAgent.py`: 智能决策智能体
- `qlearning.py`: Q-Learning 智能体

---

### 5.7 工具模块 (`src/utils/`)

#### 路径工具

**位置**: `src/utils/path_utils.py`

```python
from src.utils.path_utils import get_data_paths, get_output_paths, generate_suffix

# 获取数据路径
paths = get_data_paths(cfg)
# {
#     "primary_bktree": Path(".../bktree/primary_bktree.json"),
#     "state_node": Path(".../graph/state_node.txt"),
#     "action_log": Path(".../action_log.csv"),
#     ...
# }

# 获取输出路径
output = get_output_paths(cfg)
# {
#     "models": Path("cache/model"),
#     "logs": Path("output/logs"),
#     "figures": Path("output/figures"),
# }
```

#### 加载工具

**位置**: `src/utils/load_utils.py`

```python
from src.utils.load_utils import (
    preprocess_decision_transformer_data,  # DT 数据预处理
    create_action_dictionary,              # 创建动作字典
    get_sampling_masks                     # 获取采样掩码
)

# DT 数据预处理
dt_data, action_vocab = preprocess_decision_transformer_data(
    state_log, action_log, r_log
)
```

---

## 6. 配置说明

### 6.1 主配置文件 (`configs/config.yaml`)

```yaml
# 默认配置组
defaults:
  - model: decision_transformer
  - env: starcraft
  - experiment: default
  - _self_

# 全局设置
seed: 42          # 随机种子
device: cuda      # 设备 (cuda/cpu)
mode: train       # 运行模式 (train/eval/predict)
```

### 6.2 路径配置 (`configs/paths.yaml`)

```yaml
# 数据根目录 - 根据实际情况修改
data_root: "D:/白春辉/实验平台/pymarl/results_HRL_new/Q-bktree"

# 数据集标识
map_id: "MarineMicro_MvsM_4"   # 地图名称
data_id: "6"                    # 数据集编号

# 输出目录
output_dir: "output"   # 日志、图表输出目录
cache_dir: "cache"     # 模型、中间结果缓存目录
```

### 6.3 模型配置 (`configs/model/decision_transformer.yaml`)

```yaml
name: DecisionTransformer

# 模型超参数
d_model: 128      # 嵌入维度
n_head: 4         # 注意力头数
n_layer: 4        # Transformer 层数
max_len: 100      # 最大序列长度
dropout: 0.1      # Dropout 率

# 训练参数
training:
  epochs: 50
  batch_size: 64
  lr: 1.0e-4
  weight_decay: 0.01
  grad_clip: 1.0

# 模型特定参数
params:
  mdl_spatial_prior: false        # 是否使用空间先验
  mdl_init_embedding_freeze: false # 是否冻结初始嵌入
  mdl_init_embedding_train: false
  mdl_attn_sim_bias: false
```

---

## 7. 数据格式

### 7.1 目录结构

外部数据目录结构：

```
{data_root}/{map_id}/{data_id}/
├── bktree/
│   ├── primary_bktree.json      # 主 BK-Tree
│   └── secondary_bktree_*.json  # 次级 BK-Tree
├── graph/
│   ├── state_node.txt           # 状态-节点映射
│   └── node_log.txt             # 节点序列日志
├── sub_episode/
│   ├── 0.csv                    # 回合详细数据
│   ├── 1.csv
│   └── ...
├── action_log.csv               # 动作序列日志
├── game_result.txt              # 游戏结果
└── distance/                    # 距离矩阵缓存
```

### 7.2 文件格式详解

#### `state_node.txt` - 状态节点映射

```
{state_tuple}    {node_id}    {score}
((0,1,2,3),)    0    0.85
((0,1,2,4),)    1    0.72
```

- 每行三个字段，Tab 分隔
- 字段1：状态元组（Python 字面量格式）
- 字段2：节点ID（整数）
- 字段3：节点分数（浮点数）

#### `node_log.txt` - 状态序列

```
0 1 2 3 4 5
0 1 3 5 2 4 7 8
```

- 每行一个回合的状态序列
- 空格分隔的节点ID

#### `action_log.csv` - 动作序列

```
3k3h1g0a0b3k0h3e0f2f4e2d2e4e1e4j
4i4h3d0c2g2i0b2f1b2j4k3c4b2a2i2f4h
```

- 每行一个回合的动作序列
- 格式：`{cluster_id}{action_letter}` 交替
- 例如 `3k` 表示聚类3执行动作 `k`

#### 动作编码表

| 字符 | 动作名称 | 描述 |
|-----|---------|------|
| a | action_ATK_nearest | 攻击最近敌人 |
| b | action_ATK_clu_nearest | 攻击最近聚类 |
| c | action_ATK_nearest_weakest | 攻击最近弱敌 |
| d | action_ATK_clu_nearest_weakest | 攻击最近弱聚类 |
| e | action_ATK_threatening | 攻击威胁单位 |
| f | action_DEF_clu_nearest | 防守最近聚类 |
| g | action_MIX_gather | 集结 |
| h | action_MIX_lure | 诱敌 |
| i | action_MIX_sacrifice_lure | 牺牲诱敌 |
| j | do_randomly | 随机动作 |
| k | do_nothing | 无操作 |

#### `game_result.txt` - 游戏结果

```
win    [200]    150    10
loss   [150]    80     20
```

- 字段1：结果（win/loss/draw）
- 字段2：步数
- 字段3：得分
- 字段4：惩罚

#### `sub_episode/*.csv` - 回合详细数据

```
step[0]
cluster_1:
100,50,30,40,80;101,55,35,45,85;
cluster_-1:
200,60,40,50,90;
step[1]
...
```

- 记录每步的双方单位状态
- 单位格式：`unit_id,x,y,hp,hp_percent`

### 7.3 BK-Tree JSON 格式

```json
{
  "state": {"state": [{...}]},
  "cluster_id": 0,
  "children": {
    "1.5": {
      "state": {"state": [{...}]},
      "cluster_id": 1,
      "children": {}
    }
  }
}
```

---

## 8. 脚本使用

### 8.1 训练脚本 (`scripts/run_train.py`)

**功能**：训练模型

**用法**：

```bash
# 基本用法
python scripts/run_train.py

# 指定配置（如果使用 OmegaConf）
python scripts/run_train.py model=decision_transformer
```

**流程**：

```
1. 加载配置
2. 设置随机种子
3. 初始化 DataLoader
4. 创建模型
5. 输出路径信息
```

**输出示例**：

```
2026-03-20 17:42:22 - INFO - Training DecisionTransformer...
2026-03-20 17:42:23 - INFO - Model created: DecisionTransformer
2026-03-20 17:42:23 - INFO - Output paths: {'models': WindowsPath('cache/model'), ...}
2026-03-20 17:42:23 - INFO - Training setup completed. Model ready for training.
```

---

### 8.2 评估脚本 (`scripts/run_evaluate.py`)

**功能**：评估训练好的模型

**用法**：

```bash
python scripts/run_evaluate.py
```

**前置条件**：
- 存在训练好的模型文件：`cache/model/best_model.pth`

**输出**：

```
Loading model from cache/model/best_model.pth
Evaluating DecisionTransformer...
Decision Transformer Accuracy: 0.8542
Evaluation complete. Accuracy: 0.8542
```

---

### 8.3 预测脚本 (`scripts/run_predict.py`)

**功能**：使用模型进行行为预测

**用法**：

```bash
# 默认参数
python scripts/run_predict.py

# 自定义输入序列
python scripts/run_predict.py input="[0,1,2,3,4,5]" k=10
```

**参数**：
- `input`: 输入状态序列（JSON 格式列表）
- `k`: 预测步数

**输出**：

```
Input: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Prediction: [10, 11, 12, 13, 14]
```

---

### 8.4 数据采集脚本 (`scripts/run_collect_data.py`)

**功能**：从 StarCraft II 环境采集游戏数据

**用法**：

```bash
python scripts/run_collect_data.py
```

**前置条件**：
- 安装 StarCraft II
- 安装 pysc2
- 下载必要地图

---

## 9. API参考

### 9.1 核心类

#### DataLoader

```python
class DataLoader:
    def __init__(self, cfg: Dict) -> None
    
    @property
    def primary_bk_tree(self) -> BKTree
    
    @property
    def secondary_bk_trees(self) -> Dict[int, BKTree]
    
    @property
    def state_node_dict(self) -> Dict
    
    @property
    def reverse_dict(self) -> Dict
    
    @property
    def state_value(self) -> List[float]
    
    @property
    def state_log(self) -> List[List[int]]
    
    @property
    def action_log(self) -> List[List[str]]
    
    @property
    def r_log(self) -> List[List[float]]
    
    @property
    def dt_data(self) -> Dict[str, List]
    
    @property
    def action_vocab(self) -> Dict[str, int]
    
    def load_all(self) -> None
```

#### DecisionTransformer

```python
class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_vocab_size: int,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        max_len: int = 100
    ) -> None
    
    def forward(
        self,
        states: torch.Tensor,      # (B, S)
        actions: torch.Tensor,     # (B, S)
        rtgs: torch.Tensor,        # (B, S)
        timesteps: torch.Tensor,   # (B, S)
        padding_mask: torch.Tensor = None
    ) -> torch.Tensor              # (B, S, act_vocab_size)
```

#### BKTree

```python
class BKTree:
    def __init__(self) -> None
    
    def insert(
        self,
        state: Dict,
        cluster_id: int,
        distance_func: Callable
    ) -> None
    
    def search(
        self,
        query: Dict,
        max_distance: float,
        distance_func: Callable
    ) -> List[tuple]
```

### 9.2 工具函数

```python
# src/__init__.py
def get_config() -> Dict
def set_seed(seed: int) -> None

# src/utils/path_utils.py
def get_data_paths(cfg) -> Dict[str, Path]
def get_output_paths(cfg) -> Dict[str, Path]
def generate_suffix(params: Dict) -> str

# src/utils/load_utils.py
def preprocess_decision_transformer_data(
    state_log: List[List[int]],
    action_log: List[List[str]],
    r_log: List[List[float]]
) -> Tuple[Dict, Dict[str, int]]

def create_action_dictionary(action_path: str) -> Dict[str, str]
```

---

## 10. 常见问题

### Q1: 配置文件报错 `AttributeError: 'dict' object has no attribute 'model'`

**原因**：配置是普通字典，不支持属性访问

**解决**：使用 `cfg.get()` 方法

```python
# 错误
model_name = cfg.model.get("name")

# 正确
model_name = cfg.get("model", {}).get("name", "DecisionTransformer")
```

### Q2: 找不到模块 `ImportError: No module named 'src.models.xxx'`

**原因**：Python 大小写敏感，文件名与导入名不一致

**解决**：检查文件名大小写

```python
# 错误（文件名是 DecisionTransformer.py）
from src.models.decision_transformer import DecisionTransformer

# 正确
from src.models.DecisionTransformer import DecisionTransformer
```

### Q3: 数据长度不匹配 `AssertionError`

**原因**：state_log、action_log、r_log 长度不一致

**解决**：检查数据格式，确保 action_log 正确解析

```python
# action_log.csv 格式为 "3k3h1g..."（每两个字符一组）
# 需要提取动作字母（第2个字符）
actions = [row[0][i+1] for i in range(0, len(row[0]), 2)]
```

### Q4: CUDA 内存不足

**解决**：
1. 减小 batch_size
2. 减小模型大小（n_layer, n_embd）
3. 使用梯度累积
4. 使用 `device="cpu"`

### Q5: 评估时找不到模型文件

**解决**：
1. 先运行 `python scripts/run_train.py` 训练模型
2. 检查 `cache/model/` 目录是否存在 `best_model.pth`

---

## 附录

### A. 完整训练流程示例

```python
"""
完整的 Decision Transformer 训练流程示例
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src import get_config, set_seed, ROOT_DIR
from src.data.loader import DataLoader
from src.models.DecisionTransformer import DecisionTransformer

# 1. 加载配置
cfg = get_config()
set_seed(cfg.get("seed", 42))
device = cfg.get("device", "cuda")

# 2. 加载数据
data_loader = DataLoader(cfg)
dt_data = data_loader.dt_data
action_vocab = data_loader.action_vocab

print(f"Episodes: {len(dt_data['states'])}")
print(f"Actions: {len(action_vocab)}")

# 3. 创建数据集
class DTDataset(Dataset):
    def __init__(self, data):
        self.states = data["states"]
        self.actions = data["actions"]
        self.rtgs = data["rtgs"]
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.long),
            "actions": torch.tensor(self.actions[idx], dtype=torch.long),
            "rtgs": torch.tensor(self.rtgs[idx], dtype=torch.float),
        }

dataset = DTDataset(dt_data)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. 创建模型
state_dim = len(data_loader.state_node_dict)
model = DecisionTransformer(
    state_dim=state_dim,
    act_vocab_size=len(action_vocab),
    n_layer=4,
    n_head=4,
    n_embd=128,
    max_len=100
).to(device)

# 5. 训练设置
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
epochs = 10

# 6. 训练循环
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        rtgs = batch["rtgs"].to(device)
        timesteps = torch.arange(states.shape[1]).unsqueeze(0).expand(states.shape[0], -1).to(device)
        
        # 前向传播
        logits = model(states, actions, rtgs, timesteps)
        
        # 计算损失
        loss = criterion(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            actions[:, 1:].reshape(-1)
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# 7. 保存模型
save_path = ROOT_DIR / "cache" / "model"
save_path.mkdir(parents=True, exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "action_vocab": action_vocab,
}, save_path / "best_model.pth")

print(f"Model saved to {save_path / 'best_model.pth'}")
```

---

### B. 项目依赖完整列表

```txt
# requirements.txt

# 深度学习
torch>=2.0.0
numpy>=1.21.0

# 科学计算
scipy>=1.7.0

# 数据处理
pandas>=1.3.0
pyyaml>=6.0

# 配置管理（可选）
omegaconf>=2.3.0

# StarCraft II 环境（可选）
pysc2>=4.0.0
absl-py>=1.0.0

# 可视化
matplotlib>=3.5.0
seaborn>=0.11.0

# 其他
tqdm>=4.62.0
```

---

### C. 更新日志

| 版本 | 日期 | 更新内容 |
|-----|------|---------|
| v0.1.0 | 2026-03 | 项目重构，迁移到 src/ 结构 |

---

*文档结束*
