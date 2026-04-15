# 知识图谱构建与目录管理指南

> **最后更新**: 2026-04-08
> **关联文件**: `scripts/build_knowledge_graph.py`、`configs/kg_catalog.yaml`、`scripts/visualize_kg_web.py`

---

## 1. 概述

知识图谱（Knowledge Graph）从游戏 episode 数据中提取状态-动作决策统计，用于可视化分析和实时决策辅助。本文档描述如何为新地图/新数据集构建 KG 并注册到可视化工具。

---

## 2. 目录结构

### 2.1 数据源

外部数据目录（`configs/paths.yaml` 中 `data_root` 指定）：

```
<data_root>/<map_id>/<data_id>/
├── bktree/
│   ├── primary_bktree.json
│   └── secondary_bktree_*.json
├── graph/
│   ├── state_node.txt
│   └── node_log.txt
├── action_log.csv          ← 原始动作数据（2字符一组，如 "4d3a"）
├── game_result.txt         ← 胜负结果
├── sub_q_table/
├── sub_episode/
└── distance/
```

### 2.2 输出目录

构建后自动按地图分子目录存放：

```
cache/knowledge_graph/
├── MarineMicro_MvsM_4/
│   ├── kg_simple.pkl                    ← simple KG (context_window=0)
│   ├── kg_simple_transitions.pkl        ← simple 版 transitions
│   ├── kg_context_5.pkl                 ← context KG (可选)
│   ├── kg_context_5_transitions.pkl     ← context 版 transitions (可选)
│   ├── kg_context_10.pkl
│   └── kg_context_10_transitions.pkl
├── MarineMicro_MvsM_4_mirror/
│   ├── kg_simple.pkl
│   └── kg_simple_transitions.pkl
├── MarineMicro_MvsM_4_dist/
├── MarineMicro_MvsM_4_dist_mirror/
├── MarineMicro_MvsM_8/
└── MarineMicro_MvsM_8_mirror/
```

### 2.3 catalog 配置文件

`configs/kg_catalog.yaml` 是可视化工具的 KG 注册表：

```yaml
knowledge_graphs:
  - name: "MvsM4 - Simple"                # 下拉框显示名称
    file: "MarineMicro_MvsM_4/kg_simple.pkl"  # 相对于 cache/knowledge_graph/ 的路径
    transitions: "MarineMicro_MvsM_4/kg_simple_transitions.pkl"
    type: "simple"                          # simple | context
    context_window: 0                       # 上下文窗口大小
    map_id: "MarineMicro_MvsM_4"           # 地图标识
    data_id: "6"                            # 数据集标识
    description: "MvsM4 (data_id=6) 纯状态模式"  # 侧边栏描述
```

---

## 3. 环境要求

### 3.1 必须使用 .venv Python 执行

**这是最关键的前置条件。**

```
项目 .venv: Python 3.8.10 + numpy 1.24.4  (Streamlit 运行环境)
系统 Python: numpy 2.1.3                   (不兼容!)
```

pkl 文件的序列化格式依赖 numpy 版本。如果用系统 Python 构建，Streamlit（.venv）反序列化时会报 `ModuleNotFoundError: No module named 'numpy._core'`。

**所有构建命令必须使用：**

```bash
.venv\Scripts\python.exe scripts/build_knowledge_graph.py ...
```

而非 `python scripts/build_knowledge_graph.py`。

### 3.2 验证环境

```bash
# 确认 venv numpy 版本
.venv\Scripts\python.exe -c "import numpy; print(numpy.__version__)"
# 期望输出: 1.24.x

# 确认依赖完整
.venv\Scripts\python.exe -c "from src import get_config; from src.data.loader import DataLoader; print('OK')"
```

---

## 4. 构建命令

### 4.1 单地图构建

```bash
# 基本用法（默认只构建 simple 版本）
.venv\Scripts\python.exe scripts/build_knowledge_graph.py --map-id <地图名> --data-id <数据ID>

# 同时构建 simple + context 版本
.venv\Scripts\python.exe scripts/build_knowledge_graph.py --map-id <地图名> --data-id <数据ID> --context-windows 0 5 10

# 指定输出目录（覆盖默认的 cache/knowledge_graph/<map_id>/）
.venv\Scripts\python.exe scripts/build_knowledge_graph.py --map-id <地图名> --data-id <数据ID> --output-dir cache/knowledge_graph/custom_path
```

**参数说明：**

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--map-id` | 否 | paths.yaml 中的值 | 地图 ID，覆盖 paths.yaml |
| `--data-id` | 否 | paths.yaml 中的值 | 数据集 ID，覆盖 paths.yaml |
| `--context-windows` | 否 | `[0]` | 要构建的 context window 列表，0=simple |
| `--output-dir` | 否 | `cache/knowledge_graph/<map_id>` | 输出目录 |
| `--validate` | 否 | False | 构建后验证 |
| `--seed` | 否 | 42 | 随机种子 |

### 4.2 批量构建（当前项目全量）

```bash
# MvsM4 (data_id=6) — 含 context 5/10
.venv\Scripts\python.exe scripts/build_knowledge_graph.py --map-id MarineMicro_MvsM_4 --data-id 6 --context-windows 0 5 10

# MvsM4-mirror (data_id=3)
.venv\Scripts\python.exe scripts/build_knowledge_graph.py --map-id MarineMicro_MvsM_4_mirror --data-id 3

# MvsM4-dist (data_id=1)
.venv\Scripts\python.exe scripts/build_knowledge_graph.py --map-id MarineMicro_MvsM_4_dist --data-id 1

# MvsM4-dist-mirror (data_id=3)
.venv\Scripts\python.exe scripts/build_knowledge_graph.py --map-id MarineMicro_MvsM_4_dist_mirror --data-id 3

# MvsM8 (data_id=1)
.venv\Scripts\python.exe scripts/build_knowledge_graph.py --map-id MarineMicro_MvsM_8 --data-id 1

# MvsM8-mirror (data_id=1)
.venv\Scripts\python.exe scripts/build_knowledge_graph.py --map-id MarineMicro_MvsM_8_mirror --data-id 1
```

### 4.3 构建产物

每个 context_window 构建后生成两个文件：

| 文件 | 说明 |
|------|------|
| `kg_simple.pkl` 或 `kg_context_{N}.pkl` | 知识图谱（state-action 统计） |
| `kg_simple_transitions.pkl` 或 `kg_context_{N}_transitions.pkl` | 状态转移网络 |

transitions 文件结构：
```python
{
    state_id: {
        action: {
            "next_states": {next_state: count},
            "wins": int,
            "total": int,
            "win_rate": float
        }
    }
}
```

---

## 5. 注册到可视化工具

### 5.1 编辑 catalog

构建完成后，在 `configs/kg_catalog.yaml` 中添加条目：

```yaml
knowledge_graphs:
  # ... 已有条目 ...

  # 新增地图示例
  - name: "新地图 - Simple"
    file: "NewMap/kg_simple.pkl"
    transitions: "NewMap/kg_simple_transitions.pkl"
    type: "simple"
    context_window: 0
    map_id: "NewMap"
    data_id: "1"
    description: "新地图 (data_id=1) 纯状态模式"
```

### 5.2 清除 Streamlit 缓存

catalog 修改后，在 Streamlit 页面右上角菜单 → **"Clear cache"** → 刷新页面。

或直接删除缓存目录：
```bash
rd /s /q "%USERPROFILE%\.streamlit\cache"
```

### 5.3 启动可视化工具

```bash
.venv\Scripts\python.exe -m streamlit run scripts/visualize_kg_web.py
```

---

## 6. 已有地图数据一览

| 地图 (map_id) | data_id | 状态数 | 总访问 | KG 文件 | 可用 context 版本 |
|---|---|---|---|---|---|
| MarineMicro_MvsM_4 | 6 | 909 | 71,860 | MarineMicro_MvsM_4/ | simple, ctx5, ctx10 |
| MarineMicro_MvsM_4_mirror | 3 | 1,062 | 66,483 | MarineMicro_MvsM_4_mirror/ | simple |
| MarineMicro_MvsM_4_dist | 1 | 2,513 | 49,955 | MarineMicro_MvsM_4_dist/ | simple |
| MarineMicro_MvsM_4_dist_mirror | 3 | 1,566 | 58,871 | MarineMicro_MvsM_4_dist_mirror/ | simple |
| MarineMicro_MvsM_8 | 1 | 22,696 | 82,598 | MarineMicro_MvsM_8/ | simple |
| MarineMicro_MvsM_8_mirror | 1 | 13,039 | 73,692 | MarineMicro_MvsM_8_mirror/ | simple |

### 可用 data_id 列表（供参考）

| 地图 | data_id |
|---|---|
| MarineMicro_MvsM_4 | 1, 2, 3_, 3_1, 4, 5, 6, 7, 8, 9, 10 |
| MarineMicro_MvsM_4_mirror | 1, 2, 3, 4, 5 |
| MarineMicro_MvsM_4_dist | 1, 2, 3_ |
| MarineMicro_MvsM_4_dist_mirror | 1, 2, 3, 4 |
| MarineMicro_MvsM_8 | 1, 2, 3, 4, 5, 6, 7, 8 |
| MarineMicro_MvsM_8_mirror | 1, 2, 3, 4 |

---

## 7. 故障排查

### 7.1 `ModuleNotFoundError: No module named 'numpy._core'`

**原因**：KG pkl 文件用 numpy 2.x 序列化，但 Streamlit 运行在 numpy 1.24.4 环境中。

**解决**：用 `.venv\Scripts\python.exe` 重新构建所有 KG。

### 7.2 可视化工具中只显示旧条目

**原因**：Streamlit 缓存了 `load_kg_catalog()` 的旧结果。

**解决**：页面右上角菜单 → "Clear cache" → 刷新页面。

### 7.3 `FileNotFoundError: 文件不存在`

**原因**：catalog 中的 `file` 或 `transitions` 路径与实际文件不匹配。

**解决**：检查 `cache/knowledge_graph/<map_id>/` 下文件是否存在，确认 catalog 路径正确。

---

## 8. 关键代码改动说明

以下文件在 2026-04-08 的改动中涉及 KG 多地图支持：

### 8.1 `scripts/build_knowledge_graph.py`

- 新增 `--map-id`、`--data-id` 参数，覆盖 `paths.yaml` 配置
- 新增 `build_transitions()` 函数，构建后自动生成 transitions 文件
- 默认输出目录改为 `cache/knowledge_graph/<map_id>/`（子目录组织）
- `load_raw_action_data()` 从 `cfg["paths"]` 子字典读取路径参数
- `--context-windows` 默认值改为 `[0]`（只构建 simple）

### 8.2 `src/decision/kg_decision_helper.py`

- `__init__()` 新增 `cfg` 可选参数，避免硬编码 `get_config()`
- `_load_transitions()` 优先使用 `self.cfg`，无则 fallback 到 `get_config()`
- 路径参数从 `cfg["paths"]` 子字典读取

### 8.3 `scripts/visualize_kg_web.py`

- 新增 `load_kg_catalog()` 从 `configs/kg_catalog.yaml` 加载目录
- `load_kg()` 改为接受 `kg_file` 文件路径（不再需要 type + context_window）
- `load_transitions()` 改为接受 `transitions_file` 文件路径
- 侧边栏用 KG 选择器（selectbox + caption）替换了 KG 类型 / context window 控件
- 聚焦模式下不存在的状态 ID 会弹出 `st.toast` 提示而非渲染全部边
