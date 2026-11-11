# 植物三维网格五类自动分割

当前版本：`0.2.5`。先看 [V3 使用说明](START_HERE_V3.md)。推荐新入口 `run_v3.ps1`（拖放用 `segment_v3.cmd`），串联 V3 分割和新增视角复核，成功后生成 `result_paths.txt/json`。原 `run_auto.ps1` 保持原用法：全局推理、局部细化、五类完整赋类、多视角叶缘修复、窄叶缘补全、已有证据复核和导出校验。默认主线为 v3 的 GroundingDINO + CLIPSeg + MobileSAM，不调用或加载 SAM3。已有 v3 结果可单独用 `run_extra_views.ps1` 增加视角复核，不修改分类。

Linux server entry points are `setup_server.sh`, `check_install_server.py`, and
`run_v3_server.sh`. They use a separate Linux environment and normal Linux
wheels; the Windows `vendor` tree is not imported on Linux.

详细资料：

- [算法流程与参数](docs/ALGORITHM.md)
- [部署、运行与排错](docs/DEPLOYMENT.md)
- [代码维护与复现](docs/DEVELOPMENT.md)
- [验证与适用范围](docs/VALIDATION.md)
- [不确定标记与加视角复核](docs/UNCERTAINTY.md)
- [版本记录](CHANGELOG.md)
- [可选 SAM3 试验入口](docs/SAM3.md)

此工具可在本机独立运行，无需 Codex 会话。输入新的 PLY 后，自动执行多视角渲染、二维语义推理、局部细化、三维融合和五类完整赋类。当前是可运行的实验工具，尚未完成跨植物精度验证。

## 本机直接使用

本机依赖、三个模型权重及 Python 配置已准备好。

最简单：将带颜色的植物三角网格 PLY 拖到 `segment.cmd`。结果保存在输入文件旁的 `原文件名_semantic/complete/`。

也可以在普通 PowerShell 中运行，替换输入、输出路径：

```powershell
& 'F:\植物建模\tools\plant_semantics\run_auto.ps1' `
  -InputMesh 'D:\plants\new.ply' `
  -OutputDirectory 'D:\results\new' `
  -UpAxis auto
```

最终模型是 `D:\results\new\complete\semantic_mesh.ply`。

`auto` 只按 x、y、z 坐标范围最大的轴估计竖直轴。它不执行任意姿态校正，也不判断轴的正负。模型应沿所选轴正方向向上；较宽或旋转的植物请先摆正，并指定 `-UpAxis x`、`y` 或 `z`。原始 `Plant_cropped.ply` 使用 `y`。

每个新输入使用独立输出目录。同一输入、同一配置可以重跑并复用缓存；修改模型、提示词或相机后使用新输出目录。缓存不兼容时程序会报错，避免混合不同配置的预测。运行时间取决于视角数、网格规模、CPU/GPU；本机当前为 CPU 推理。中间渲染和分数文件会占用磁盘空间。

## 输入及类别

输入要求：三角网格 PLY，含逐顶点 red/green/blue 颜色，坐标有效。输入表面应只包括以下五类；地面、墙壁、标签牌等应先裁掉。纯点云、无 RGB、仅有纹理贴图的模型需要先转换，不由本入口直接处理。

| label | 类别 | 语义颜色 |
| --- | --- | --- |
| 1 | 枝干、茎、叶柄、花梗 | 橙色 |
| 2 | 叶片 | 绿色 |
| 3 | 花、花苞 | 粉色 |
| 4 | 花盆、托盘 | 蓝色 |
| 5 | 土壤、基质 | 棕色 |

每个顶点、每个三角面最终都有 1–5 标签，没有灰色未知类。某种器官不存在时，该类数量可以为 0，无需让五类都出现。不确定性另存于 `uncertain` 属性，完整赋类不等于全部正确。

目前优先适用于单株、竖直摆放、带较清楚 RGB 外观的植物模型。盆口细化包含紧凑近圆单盆的几何假设；多盆、长方盆、严重粘连、极粗糙或缺失器官的重建尚未验证。没有检测到盆口时会跳过对应细化，流程仍可完成。

## 输出及检查

主结果位于输出目录的 `complete/`：

- `semantic_mesh.ply`：五类纯语义着色，逐顶点与逐面 `label`。
- `labeled_original_rgb.ply`：原 RGB，加相同标签；原色可能是灰色，这不表示未分类。
- `labels.npz`：标签、非校准分数、边界、不确定性、多视角记录及推断标志。
- `leaf_rim_comparison.jpg`、`leaf_rim_compact.jpg`：窄叶缘补全前后局部对比；没有可靠叶片区域时不生成。
- `rim_config.json`、`rim_diagnostics.npz`：窄叶缘补全配置、候选顶点索引、归一化网格距离、邻域叶片支持及邻点数。`leaf_rim_refined` 记录在 `labels.npz`。
- `uncertainty_report.json`、`uncertainty_config.json`、`uncertainty_audit.json`、`uncertainty_comparison.jpg`：最终历史不确定标记的证据复核结果、规则、无语义改动检查及对比图。
- `comparison_preview.jpg`、`comparison_compact.jpg`：原色、中间阶段、最终结果对照。
- `detail_preview.jpg`：存在局部视角时输出的细节对照。
- `report.json`：各类数量、面积占比、不确定比例和方法限制。
- `validation.json`：检查原坐标、顶点/面顺序、面索引、标签、颜色及元数据，并检查顶点和面均无 label=0。失败时入口报错。
- `manifest.json`、`inference_config.json`：输入身份、模型及推理配置记录。

`baseline/`、`detail/` 为中间缓存和诊断结果，可能包含灰色未知区域；`complete_before_boundary/` 是多视角叶缘修复前的完整五类结果，`complete_before_rim/` 是窄叶缘补全前的结果。应使用 `complete/` 中的最终模型。更新工具后再次使用旧输出目录，程序会重建中间五类结果并优化边界；原始输入不变。结果保留输入的顶点数量、顺序及三角面索引；导出坐标为 float32，原 alpha、法线等其他属性不全部复制。若输入是需要保留更高精度的 float64 坐标，请先评估；几何校验会拒绝精度变化的结果。

`uncertain=1` 是诊断标志，`inferred=1` 表示细节阶段未分类、在最终阶段已推断。分数与多视角一致性都不是语义准确率，不能用来宣称达到某个准确率。

0.2.3 最终步骤会依据已有独立相机实例支持和叶面掩膜重新评估历史不确定标记，原始值另存为 `uncertain_before_reassessment`，解除位置记为 `uncertainty_released`。仅靠高分或邻域传播不解除标记，语义标签、分数、坐标与拓扑不变。原植物复核版位于 `outputs/plant_semantics_complete_v3_reviewed`：标记从 477,072（30.43%）降至 391,087（24.95%），基质从 13,167 降至 6,901。该变化是复核元数据更新，不代表精度提高，也不意味着解除标记的区域可以直接作为机械臂安全落点。

### 可选：增加视角继续复核

对上述已有 `complete` 结果运行以下命令；新植物替换路径即可。使用与原 v3 相同的三个模型，保留语义标签、分数和全部几何。该步骤是额外推理，未强制加入默认完整流程。

```powershell
& 'F:\植物建模\tools\plant_semantics\run_extra_views.ps1' `
  -SourceDirectory 'F:\植物建模\outputs\plant_semantics_complete_v3_reviewed' `
  -OutputDirectory 'F:\植物建模\outputs\plant_semantics_v3_extra_views' `
  -Views 12
```

`Views` 是覆盖选择预算；有基质时还会补足最多 3 个相隔较远的俯视角，本株实际新增 14 个视角。新结果是输出目录下的 `complete/semantic_mesh.ply`。不要用已完成本步骤的结果再次运行；改变预算时，从同一源结果开始并使用新的输出目录。

本株标记从 391,087（24.95%）降到 323,052（20.61%）：新增证据解除 75,475 个，也新发现 7,440 个冲突标记。改善主要来自花瓣；基质标记从 6,901 增至 7,271，不能认为各类都会随视角数增加而改善。`extra_view_report.json`、`extra_view_audit.json`、`extra_view_config.json` 和 `uncertainty_comparison.jpg` 记录本步骤，早期 `uncertainty_report.json` 是上一步的历史结果。

这里的“新视角”是从已有模型重新渲染，不是补拍真实照片。它不能恢复扫描缺失的表面，模型的相似错误也可能跨视角重复。详细规则、保存字段和限制见 [UNCERTAINTY.md](docs/UNCERTAINTY.md)。

## 自动流程

1. 用 Open3D 射线渲染 14 个全局视角，保存像素对应的三角面与重心坐标。
2. GroundingDINO-Tiny 检测器官候选框，CLIPSeg 生成文本语义分数，MobileSAM 细化候选掩膜。
3. 投影每个原顶点并做射线遮挡检查，融合可见视角的证据。
4. 自动寻找细茎区域及可用盆口，生成局部放大视角；双尺度 PCA 提供线性/平面形状辅助。同一相机的裁剪共享一次投票。
5. 重提实例证据、保护花苞，用自动伪标签拟合当前场景的 Lab 颜色 GMM。沿实际网格边传播与细化，给所有顶点和面分配五类标签。
6. 自动提取可靠叶片连通区域，用内部点提示 MobileSAM 重新勾画完整叶面。补充 12 个斜下方方向；至少 3 个不同相机支持且可见视角同意率达到 70%，才将邻近的茎/花误分纠正为叶片。只沿网格边连接、限定距离；保护花苞、盆和基质，不按固定 RGB 颜色阈值修改。
7. 补全残留的茎色窄叶缘：从已有可靠叶片区域沿茎/叶网格边搜索，距离不超过模型对角线的 0.4%；周围均匀采样表面中叶片的加权支持达到 40% 才改为叶片。种子固定、只执行一轮，避免递归向枝干生长。此步不读取 RGB，不要求卷曲边缘是平面；真实叶柄连接处仍可能有歧义。
8. 输出模型、预览、报告并验证导出。`boundary_records.json`、`boundary_config.json` 记录自动点提示、掩膜质量、相机及版本；`labels.npz` 中 `leaf_boundary_refined`、`leaf_rim_refined` 分别记录两阶段改动。窄叶缘新改动点保留不确定标记。

0.2.3 在导出最终结果之前增加不确定标记复核，详见 [UNCERTAINTY.md](docs/UNCERTAINTY.md)。新增中间目录 `complete_before_uncertainty/` 保存原 v3 标记，最终 `complete/` 使用复核标记。新输出里的历史阶段预览依然描述相同的分类结果，不是当前不确定性图；查看 `uncertainty_comparison.jpg` 了解标记变化。

基础模型冻结，无人工框点标注或微调；颜色分布会对每个输入自动拟合。算法不依赖当前会话的人工操作，但提示词与阈值曾根据原植物的试验结果进行工程选择，泛化精度仍需独立验证。此默认流程不使用 SAM3；目录中保留独立的 `run_sam3.ps1` 试验入口，配置要求见 `docs/SAM3.md`。

## 在另一台 Windows 电脑部署

源码包 `plant_semantics_source.zip` 不含 Python、vendor 依赖或约 1.33 GB 的模型权重。复制整个当前工具目录可带上现有权重；本机的 `runtime.local.json` 路径不可直接迁移，应在新机器重新配置。

推荐安装 Python 3.10–3.12，并在解压目录中创建虚拟环境。先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装相互匹配的 torch/torchvision（CPU 或对应 CUDA），再执行：

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& '.\setup_local.ps1' -PythonExecutable '.\.venv\Scripts\python.exe' -InstallDependencies -DownloadModels
```

下载公开 GroundingDINO、CLIPSeg 权重及 MobileSAM 官方源码/权重。大文件当前使用 hf-mirror 公开文件镜像，需可访问镜像与 GitHub；不下载受限 SAM3。也可把现有 `models/` 和 `third_party/MobileSAM/` 复制过去，再省略 `-DownloadModels`。下载器会复用缓存；这不是离线安装包。

`setup_local.ps1` 通过检查后保存本机 Python 路径。入口按显式 `-PythonExecutable`、`PLANTSEG_PYTHON`、工具内/项目内虚拟环境、本机配置、PATH/Windows Python launcher 查找。也可手动指定 `-SamCheckpoint 'D:\models\mobile_sam.pt'`。

如果本机策略拦截 `.ps1`，可使用 `segment.cmd`，或使用进程级参数：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File '.\run_auto.ps1' -InputMesh 'D:\plants\new.ply' -OutputDirectory 'D:\results\new' -UpAxis y
```

## 验证记录与限制

- 原始植物：1,567,628 顶点、3,136,830 面，最新 V3 结果在 `F:\植物建模\outputs\plant_semantics_v3_extra_views\complete`；顶点/面未知数均为 0，几何及导出校验通过。历史 v2/v3 仍保留，尚无人工真值 IoU。
- 叶缘优化结果在 `F:\植物建模\outputs\plant_semantics_complete_v2`：11,324 个顶点依据新增的叶面实例证据改为叶片；盆、基质和已保护花苞标签不变。修复主要针对叶缘黄带与叶背粉色杂点，仍有极细边缘、基部和重建粘连处误差。该数字是改动数量，不是已证明正确的数量或准确率。
- 最新补全结果在 `F:\植物建模\outputs\plant_semantics_complete_v3`：在 v2 基础上，将 2,534 个茎标签顶点改为叶片，约占总表面积 0.1706%；原有叶、花、盆、基质及受保护花苞标签不变。多视角局部对比显示橙色叶缘减少，尚无人工真值精度评价。
- 小型网格冒烟测试：`Plant_nomip_crop_attempt.ply`，1,437 顶点、2,473 面，完整自动流程通过，输出在 `outputs/plant_auto_smoke2/complete`。该粗糙模型有 99.93% 顶点标为不确定，因此只用于验证流程与无盆口分支，不证明语义效果或跨物种泛化。
- 共 17 项回归：12 项几何/边界、4 项窄叶缘行为、1 项源码包测试。覆盖遮挡、投影、导出、图传播、缺失盆口/细节预览、叶缘补全、不连通枝干和其他器官保护、缩放平移一致性及分发完整性；这些测试不衡量语义准确率。
- 0.2.3 新增 4 项不确定复核回归，共 21 项；验证多相机确认、类别冲突保留、保护花苞、几何修改需独立掩膜确认及相机数据身份。详见 `docs/UNCERTAINTY.md`。
- 0.2.4 新增 8 项视角复核测试，0.2.5 新增 V3 分发检查，共 30 项；新的一键入口通过调用链路验证，详见 `docs/VALIDATION.md`。

新植物可直接运行；不同物种、结构、重建质量下，尤其细茎、绿色花苞、叶柄和花梗边界、盆内基质，仍可能错分。跨植物高精度保证需要独立样本及逐类 IoU/召回、边界质量评估。旧试验记录在 `docs/archive/EXPERIMENT_HISTORY.md`，其中旧命令和路径仅作历史记录。

## 模型来源与源码包

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) / [Tiny 权重](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
- [CLIPSeg](https://github.com/timojl/clipseg) / [rd64-refined 权重](https://huggingface.co/CIDAS/clipseg-rd64-refined)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)，许可证随 `third_party/MobileSAM/LICENSE` 保存
- [Open3D](https://github.com/isl-org/Open3D)

各代码及权重的使用条件以对应项目许可为准。[SAM3 独立入口与部署说明](docs/SAM3.md) 已补充原植物的 CPU、CUDA 和完整自动流程验证。`run_sam3.ps1 -RefineLeafBoundary` 可运行 SAM3 + MobileSAM 叶面修复组合；修复结果另存于 `complete_leaf_refined/`。本株可视对比中，直接 SAM3 融合仍有叶背粉色误分，加入自动叶面修复后大幅减少；尚无人工真值和跨植物精度验证。默认 `run_auto.ps1` 不因此切换模型。

更新源码包：在已配置的 Python 中运行 `python build_package.py`。源码包包含独立入口、部署/下载/检查脚本、测试、算法/维护说明及 MobileSAM 源码，不含本机路径配置；同时生成 `plant_semantics_source.zip.sha256` 和 ZIP 内 `PACKAGE_MANIFEST.json`。

仅分发 V3 时运行 `python build_package.py --profile v3`，生成 `plant_semantics_v3_source.zip` 及对应 SHA256。该包排除独立 SAM3 程序/说明与历史实验目录，使用 `run_v3.ps1`；其余必要源码、MobileSAM 许可证和维护测试均保留。
