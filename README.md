# PredictionRTS

RTS（即时战略游戏）中的智能体行为预测与决策研究项目。

## 项目结构

```
PredictionRTS/
├── configs/                 # YAML 配置文件
│   ├── config.yaml         # 主配置
│   ├── paths.yaml          # 数据路径配置
│   ├── model/              # 模型配置
│   ├── env/                # 环境配置
│   └── experiment/         # 实验配置
├── src/                    # 源代码模块
│   ├── config/             # 配置模块
│   ├── data/               # 数据加载
│   ├── models/             # 模型定义
│   ├── env/                # 环境与智能体
│   ├── train/              # 训练逻辑
│   ├── structure/          # 数据结构 (BK-Tree等)
│   ├── algorithms/         # 算法模块
│   ├── utils/              # 工具函数
│   └── plot/               # 可视化
├── scripts/                # 入口脚本
│   ├── run_train.py
│   ├── run_evaluate.py
│   ├── run_predict.py
│   └── run_collect_data.py
├── data/                   # 原始数据文件
├── cache/                  # 模型和数据缓存
├── output/                 # 输出结果
├── tex/                    # 报告文档
└── doc/                    # 项目文档
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置数据路径

复制并编辑路径配置文件：

```bash
cp configs/paths.yaml.example configs/paths.yaml
# 编辑 paths.yaml 设置您的数据路径
```

### 训练模型

```bash
# 使用 Python 直接运行
python scripts/run_train.py

# 或使用配置文件
python -c "from src.train import run_training; run_training()"
```

### 评估模型

```bash
python scripts/run_evaluate.py
```

## 核心模型

| 模型 | 说明 |
|------|------|
| **DecisionTransformer** | 对"状态 + 动作 + 回报"三元组建模，预测下一步动作 |
| **TrajectoryTransformer** | 对"状态索引序列"做自回归预测 |
| **SGTransformer** | 在注意力中引入状态距离引导的相似度偏置 |

## 依赖

- Python 3.8+
- PyTorch 1.10+
- numpy, pandas, scikit-learn, matplotlib
- pysc2 (用于数据采集，可选)

## License

MIT
