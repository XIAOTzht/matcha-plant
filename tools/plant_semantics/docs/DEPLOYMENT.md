# 部署、运行与排错

## 本机

本机 `models/`、MobileSAM 源码及 Python 配置已就绪。无需保持 Codex 开启。

推荐先看根目录 `START_HERE_V3.md`，使用 `run_v3.ps1` 或将新模型拖到 `segment_v3.cmd`。该入口串联分割和新增视角复核，最终位置在 `result_paths.txt/json` 中；只做原分割可设 `-ExtraViews 0`。原入口的路径布局保持下面的用法。

把新的彩色网格 PLY 拖到根目录 `segment.cmd`，结果在输入旁的 `原文件名_semantic/complete/`。命令行调用见根目录 README。`auto` 只按坐标范围最大的轴估计竖直方向，不旋转模型、不判断正负；旋转或很宽的植物应先摆正并指定轴。原始 `Plant_cropped.ply` 使用 `-UpAxis y`。

## 另一台 Windows 电脑

源码包不包含 Python、约 1.33 GB 的三个模型权重或 `vendor/` 二进制依赖。推荐 Python 3.10–3.12。解压源码后在项目目录运行：

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& '.\setup_local.ps1' -PythonExecutable '.\.venv\Scripts\python.exe' -InstallDependencies -DownloadModels
```

示例安装 CPU PyTorch；GPU 环境先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 选用配套 torch/torchvision 和 CUDA，再执行 setup。`requirements.txt` 保存项目依赖约束，并非跨电脑完全锁定的二进制环境。

下载器获取公开 GroundingDINO-Tiny、CLIPSeg 权重和 MobileSAM 官方源码/权重，不需要 SAM3 授权。目前大文件使用 hf-mirror，需能访问该镜像及 GitHub；下载中断可重跑并复用分块。也可复制当前工具目录的 `models/` 和 `third_party/MobileSAM/`，再省略 `-DownloadModels`。

安装脚本检查通过才保存新电脑的 `runtime.local.json`。不要把旧电脑的该配置当作可移植配置。当前 `vendor/` 含 Windows Python 3.12 编译扩展，不建议复制到不同 Python 环境。

## 环境定位

Python 搜索顺序：显式 `-PythonExecutable` → `PLANTSEG_PYTHON` → 工具目录/项目目录虚拟环境 → `runtime.local.json` → PATH 和 Windows py launcher。MobileSAM 默认路径为 `models/mobile_sam_parallel.pt`，也可使用 `-SamCheckpoint` 指定。

```powershell
& '.\.venv\Scripts\python.exe' '.\check_install.py' --input 'D:\plants\new.ply'
& '.\run_auto.ps1' -InputMesh 'D:\plants\new.ply' -OutputDirectory 'D:\results\new' -UpAxis y -PythonExecutable '.\.venv\Scripts\python.exe'
```

首次运行的时间与内存/磁盘占用取决于网格大小、视角及 CPU/GPU。本机当前使用 CPU。每个新输入使用独立输出目录；同一输入与同一配置可以复用已有二维推理缓存，完整五类与边缘阶段仍可能重新计算。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| PowerShell 禁止运行 ps1 | 用 `segment.cmd`，或对本次进程加 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File ...` |
| 找不到 Python | 用 `-PythonExecutable` 显式传入 Python 3.10–3.12，并运行 setup |
| 找不到权重/模型配置 | `setup_local.ps1 -DownloadModels`；如网络不可用则复制完整模型目录 |
| 缓存配置不一致 | 新建输出目录，不混用不同输入/视角/模型的中间文件 |
| 中间模型还有灰色 | 使用最终 `complete/semantic_mesh.ply`；原 RGB 的灰色不代表没有标签 |
| 没有检测到花盆 | 盆口局部细化可跳过，流程仍可完成；需要检查最终语义是否合理 |
| 几何精度校验失败 | 导出使用 float32，输入若需要 float64 精度须先评估；入口不会把校验失败当成功 |
| 跨物种或特殊颜色效果不好 | 检查预览及不确定性；当前无跨植物精度保证，见算法和验证说明 |

## 开源来源

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) / [Tiny 权重](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
- [CLIPSeg](https://github.com/timojl/clipseg) / [rd64-refined 权重](https://huggingface.co/CIDAS/clipseg-rd64-refined)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)，许可证随源码分发
- [Open3D](https://github.com/isl-org/Open3D)

各模型权重及依赖的使用条件以对应项目许可为准。默认三个模型的权重 SHA256 保存在 `models/weight_hashes.json`。SAM3 独立试验入口有单独的环境与权重要求，见 [SAM3.md](SAM3.md)；本页安装步骤针对默认流程。
