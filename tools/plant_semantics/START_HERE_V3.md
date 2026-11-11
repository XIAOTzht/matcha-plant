# V3 使用说明

版本 `0.2.5`。本流程使用 GroundingDINO-Tiny、CLIPSeg、MobileSAM，完成五类语义分割、叶缘修复、历史不确定标记复核，再增加视角复核。可在本机独立运行，不需要 Codex。V3 运行入口不使用 SAM3。

## 先选入口

| 需求 | 入口 | 最终模型 |
| --- | --- | --- |
| 新植物，分割并增加视角复核 | `run_v3.ps1`，或拖放到 `segment_v3.cmd` | `输出目录/extra_views/complete/semantic_mesh.ply` |
| 新植物，只做基础 V3 分割和已有证据复核 | `run_v3.ps1 -ExtraViews 0`，或原 `run_auto.ps1` | 新入口为 `输出目录/segmentation/complete/semantic_mesh.ply`；旧入口为 `输出目录/complete/semantic_mesh.ply` |
| 已有 V3 结果，只增加视角 | `run_extra_views.ps1` | `输出目录/complete/semantic_mesh.ply` |

推荐新任务用 `run_v3.ps1`。原入口与已生成结果路径保留，便于复查。`SamCheckpoint` 参数指 MobileSAM 权重。

## 本机运行新植物

在 PowerShell 中执行，替换输入与输出路径；每株植物使用独立输出目录：

```powershell
& 'F:\植物建模\tools\plant_semantics\run_v3.ps1' `
  -InputMesh 'D:\plants\new.ply' `
  -OutputDirectory 'D:\results\new_v3' `
  -UpAxis auto `
  -ExtraViews 12
```

也可将 PLY 拖到工具目录的 `segment_v3.cmd`：结果在输入文件旁的 `原文件名_v3/`。拖放使用 `auto`；若需明确指定竖直轴，请用上面的命令。

`auto` 只按最大坐标范围估计 x/y/z 轴，不旋转模型、不判断轴的正负。原始 `Plant_cropped.ply` 应指定 `-UpAxis y`。旋转或很宽的模型应先摆正，再明确指定轴。

`ExtraViews` 是 1–24 的覆盖选择预算；设为 0 则跳过新增视角。预算 ≥6 且存在不确定基质时会补足最多 3 个俯视角，因此本株预算 12、实际增加 14 个。相机数不会超过候选集的 24 个。

新入口输出布局：

```text
new_v3/
  result_paths.txt          运行成功后的完整路径表
  result_paths.json         同一份路径表，便于程序读取（UTF-8，可带 BOM）
  segmentation/
    baseline/              全局渲染与推理缓存
    detail/                局部细化缓存
    complete_before_*/     叶缘及不确定性处理的中间版本
    complete/              未新增视角的 V3 结果
  extra_views/
    views/                 新增视角的图像、射线、预测
    complete/              开启新增视角时的最终模型和标签
```

入口在两步和最终校验全部成功后才写路径表；运行报错时不要把旧路径表当作本次成功结果。相同输入、参数和输出目录可重跑复用缓存。更换模型、输入、提示、视角预算时使用新目录。所有缓存/manifest 路径互相引用，保留整个输出树和原始输入，不要只搬动缓存的某个子目录。

## 已有结果只加视角

```powershell
& 'F:\植物建模\tools\plant_semantics\run_extra_views.ps1' `
  -SourceDirectory 'F:\植物建模\outputs\plant_semantics_complete_v3_reviewed' `
  -OutputDirectory 'F:\植物建模\outputs\plant_semantics_v3_extra_views' `
  -Views 12
```

源目录需包含完整 `labels.npz`、manifest 及对应原模型/基础推理配置。不要以已做过新增视角复核的结果再次复核；增加预算时从同一源结果开始，另用新输出目录。

## 最终文件与类别

| 文件 | 用途 |
| --- | --- |
| `semantic_mesh.ply` | 五类语义颜色；顶点和面均有 `label` |
| `labeled_original_rgb.ply` | 原 RGB，加同样的类别和不确定标记 |
| `labels.npz` | 程序读取的顶点/面标签、分数、不确定性及历史记录 |
| `classes.json` | 类别编号、名称和语义颜色 |
| `report.json` | 分类数量、面积及当前不确定比例 |
| `extra_view_report.json` | 开启加视角时，当前标记的逐类数量与新增冲突 |
| `uncertainty_report.json` | 已有证据复核阶段的报告；加视角输出中它属于历史步骤 |
| `uncertainty_comparison.jpg` | 本次最终复核前后的标记对比，橙色为需复核 |
| `extra_view_audit.json` | 加视角前后语义标签、分数和历史字段不变检查 |
| `validation.json` | 坐标、面索引、类别、颜色和元数据导出检查 |

| label | 类别 | 颜色 |
| --- | --- | --- |
| 1 | 茎枝、叶柄、花梗 | 橙 |
| 2 | 叶片 | 绿 |
| 3 | 花、花苞 | 粉 |
| 4 | 花盆、托盘 | 蓝 |
| 5 | 土壤、基质 | 棕 |

自动读取最终路径和标签示例：

```python
import json
from pathlib import Path
import numpy as np

paths = json.loads(Path(r'D:\results\new_v3\result_paths.json').read_text(encoding='utf-8-sig'))
with np.load(paths['labels']) as data:
    vertex_labels = data['vertex_labels']  # 与输入 PLY 顶点顺序相同
    face_labels = data['face_labels']
    uncertain = data['uncertain'].astype(bool)
```

## 本机当前模型

原始输入：`F:\植物建模\outputs\Plant_cropped.ply`。

| 版本 | 结果目录 | 不确定点 |
| --- | --- | ---: |
| 原 V3（窄叶缘补全） | `F:\植物建模\outputs\plant_semantics_complete_v3` | 477,072（30.43%） |
| V3 已有证据复核 | `F:\植物建模\outputs\plant_semantics_complete_v3_reviewed` | 391,087（24.95%） |
| 当前 V3 新增 14 视角复核 | `F:\植物建模\outputs\plant_semantics_v3_extra_views\complete` | 323,052（20.61%） |

当前模型的语义标签、分数和几何与前两版一致。新增视角解除 75,475 个旧标记，也新增 7,440 个冲突；基质标记由 6,901 增到 7,271。减少标记不能解释为准确率提高。

## 源码与迁移

- 工具目录：`F:\植物建模\tools\plant_semantics`。
- 推荐 V3 源码包：`plant_semantics_v3_source.zip`，不收录独立 SAM3 程序/说明和历史实验目录。
- 完整维护源码包：`plant_semantics_source.zip`，保留独立历史入口。
- 两种包都不含 Python、模型权重、vendor 依赖、本机配置和输出模型；不等于离线环境包。均含逐文件 SHA256 清单和外部 `.sha256` 校验。
- 本机三个权重位于 `models/grounding-dino-tiny/`、`models/clipseg-rd64-refined/`、`models/mobile_sam_parallel.pt`，哈希见 `models/weight_hashes.json`。

另一台 Windows 电脑：安装 Python 3.10–3.12，在解压目录创建环境并准备公开模型：

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& '.\setup_local.ps1' -PythonExecutable '.\.venv\Scripts\python.exe' -InstallDependencies -DownloadModels
& '.\run_v3.ps1' -InputMesh 'D:\plants\new.ply' -OutputDirectory 'D:\results\new_v3' -UpAxis y
```

这是 CPU 示例；GPU 先按 PyTorch 官方说明安装匹配版本。复制权重时只需上面三个模型及 `third_party/MobileSAM/`，重新配置本机 Python 路径。完整条件与排错见 [部署说明](docs/DEPLOYMENT.md)。

维护检查：`check_code.ps1`。重建 V3 包：`python build_package.py --profile v3`；重建完整包：`python build_package.py`。源码组织见 [维护说明](docs/DEVELOPMENT.md)，算法见 [算法说明](docs/ALGORITHM.md)，多视角规则见 [不确定性说明](docs/UNCERTAINTY.md)。

## 使用范围

输入须为带逐顶点 RGB 的三角网格 PLY，单株模型应摆正，背景物体先裁除。纯点云、仅有贴图但无顶点颜色的网格需先转换。当前尚未完成跨植物、花色和枯叶的真值精度评价。

新增视角来自已有模型渲染，不能恢复缺失的扫描表面；相同模型可能在多个角度重复误判。`uncertain` 是诊断标记，分数不是校准概率。后续浇水规划需读取类别与几何并单独建立代价；本工具尚未实现机器人路径规划或机械臂控制。
