# 项目总结文档（导师汇报用）

## 编译

- 在 **本目录** 下执行：`xelatex main.tex` 或 `pdflatex main.tex`（需安装 ctex 及常用宏包）。
- 图片路径以 `main.tex` 所在目录为基准：项目已有图使用 `../../output/`、`../../plot/`；自备图使用 `figures/`。

## 图片放置

- **已有项目图**：在 `main.tex` 中可将占位框替换为例如：
  - 训练/消融：`../../plot/training_comparison.png`、`../../plot/prediction_results.png`
  - DT 评估：`../../output/dt_performance_analysis.png`、`../../output/dt_results.png`
  - 地形轨迹：`../../output/figures/pred_FL/traj_analysis_10.pdf` 等
- **自备图**：放入本目录下 `figures/` 文件夹，在正文中写 `figures/您的文件名.pdf`（或 .png）。

## 附录

文档末尾附录给出了用于 **Nano Banana Pro** 的文生图提示词，用于生成“项目整体架构图”和“方法机理图”风格的科研示意图。
