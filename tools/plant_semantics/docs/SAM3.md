# SAM3 分割与对比入口

`run_auto.ps1` 保持原 v2 流程。`run_sam3.ps1` 是独立的 SAM3 候选流程，仍需人工复核输出；没有人工标注准确率与跨植物精度验证。

SAM3 运行环境与原有 `vendor/` 分开。`sam3_predict.py` 仅使用本地 Hugging Face SAM3 权重文件夹：`config.json`、`model.safetensors` 和 tokenizer 文件。若配置为视频模型，从 safetensors 提取 `detector_model.` 部分，用严格键匹配加载图像模型，丢弃视频追踪器。不能把 SAM3.1 multiplex 的 PT 文件直接当作该 HF 图像权重文件夹使用。

本机 CPU 环境为 `.venv-sam3/Scripts/python.exe`；GPU 环境为 `.venv-sam3-gpu/Scripts/python.exe`，指定 `-Device cuda` 自动选择。新环境需安装支持 SAM3 的 Transformers、匹配版本的 torch/torchvision、Pillow、numpy、safetensors。不要向旧 vendor 目录覆盖安装新 Transformers。GPU 模式使用 bfloat16，CPU 为 float32。

```powershell
& 'F:\植物建模\tools\plant_semantics\run_sam3.ps1' `
  -InputMesh 'D:\plants\new.ply' `
  -OutputDirectory 'D:\results\new_sam3' `
  -Sam3Weights 'F:\sam3\权重' `
  -Device cpu -UpAxis auto
```

本机 RTX 4060 8 GB 已完成 CUDA 推理验证，建议改用 `-Device cuda`。可追加 `-RefineLeafBoundary`，使用已有 MobileSAM 对自动识别的叶片补充斜下方视角与点提示修复；所有提示由程序生成，保留 SAM3 原结果，另输出 `complete_leaf_refined/semantic_mesh.ply`。这是 SAM3 + MobileSAM 组合，不能称为纯 SAM3 结果。移动到其他电脑时也需提供 MobileSAM 权重，可使用 `-MobileSamCheckpoint` 指定。

新建隔离的 SAM3 环境示例（Python 3.12；基础渲染环境仍按 README 安装）：

```powershell
python -m venv .venv-sam3-gpu
& '.\.venv-sam3-gpu\Scripts\python.exe' -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
& '.\.venv-sam3-gpu\Scripts\python.exe' -m pip install transformers==5.17.0 huggingface-hub==1.33.0 numpy Pillow safetensors
```

CPU 环境名称改成 `.venv-sam3`，PyTorch 索引改为 `https://download.pytorch.org/whl/cpu`。环境可以单独放置，通过 `-Sam3Python` 指向其 Python。源码 ZIP 不包含这些运行环境或 SAM3 权重。

入口依次执行：全局 RGB 渲染 → SAM3 11 个通用文本提示 → 均匀局部裁剪及自动盆口裁剪 → 局部 SAM3 → 射线深度检查与三维多视角融合 → 自动外观伪种子与网格边扩散 → 五类完整导出 → 原几何/标签一致性校验。不需要人工点、框、涂色或微调。不依据花/叶的固定颜色设定类别。

输出为 `complete/semantic_mesh.ply` 和 `complete/labeled_original_rgb.ply`。所有顶点和三角面归入 1–5 类，低证据区域保留 `uncertain`/`inferred` 标志，不显示灰色。

当前原植物对照实验复用了现有 14 个全局和 28 个自动局部视角，以便比较相同输入。旧标签没有参与 SAM3 三维补全，仅用于对比图。独立入口对新植物通过均匀裁剪和 SAM3 的盆掩膜自动选择局部相机，不需要先运行 v2。

v2 还包含额外的叶面修复视角；独立 SAM3 对本株生成 35 个局部视角。因此这些结果是完整流程的案例比较，不是只替换一个模型、其余因素完全相同的准确率基准。原植物的比较结论和图片见 `F:\植物建模\outputs\sam3_comparison\对比说明.md`。

盆约束依然采用单个紧凑近圆容器假设；自动竖直轴只取坐标跨度最大轴。不能保证任意形态、姿态、花色、叶色和严重重建粘连均准确。三维补全、二维覆盖率及两算法一致率均不是语义准确率。
