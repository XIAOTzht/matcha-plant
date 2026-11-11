# 自动提取植物及花盆

**只输入三维 PLY、裁剪旧的完整场景模型：请看 [三维自动裁剪说明.md](三维自动裁剪说明.md)，运行 `auto_crop_mesh.py` 或 `run_crop.ps1`。下面是另一种需要照片遮罩和位姿的流程。**

工具位于 `F:\植物建模\tools\plant_extract`。运行在 CPU，无需重新训练，不修改 MAtCha 和 `mip_filter`。

输入必须配套：原始三角网格 PLY、同一次重建导出的 `mast3r_sfm`、每个视角的植物和花盆遮罩。
程序将三维顶点和三角形中心投影到照片上，默认要求至少 3 个视角命中前景，且在有效画面内的命中比例达到 70%。
遮罩轻微扩张以容忍位姿和边缘误差。默认保留所有通过前景判断的连通块，避免误删分离的叶片。

这是一种多视图轮廓筛选，不是能够从任意 PLY 理解植物的语义模型。同样投影进前景轮廓的环境表面可能残留；
遮罩漏掉的叶子或位姿不准可能导致误删。遮罩需同时包含花、叶、茎和花盆。
拍摄覆盖越完整越可靠。它不能修复没有重建出来的表面，也不能修复旋转植物导致的错误相机位姿。

## 当前实验：上传脚本和已有遮罩

在本机 Windows PowerShell 执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "F:\植物建模\tools\plant_extract\upload_segmented.ps1"
```

这只上传代码和 12 张遮罩，不重传照片，不启动新训练。SSH/SCP 可能提示多次输入密码。

## 在服务器启动自动提取

```bash
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
"$ROOT/envs/matcha/bin/python" -m pip install -r "$ROOT/tools/plant_extract/requirements.txt"
tmux new-session -d -s plant_auto_extract "bash '$ROOT/tools/plant_extract/extract_after_segmented.sh'"
```

如果已有同名提取会话，请先查看其进度；不要重复启动。此脚本等待当前 `matcha_plant_segmented` 会话结束，
确认日志出现 `PLANT_SEGMENTED_MODEL_SUCCESS` 后才开始提取。训练失败则停止，原因写入日志。

```bash
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
tail -n 60 "$ROOT/logs/plant_segmented_auto_extract.log"
```

## 对其他新植物复用

已有配套照片遮罩时：

```bash
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
bash "$ROOT/tools/plant_extract/run_extract.sh" \
  "$ROOT/outputs/你的新实验" \
  "$ROOT/data/你的新实验/masks"
```

`run_extract.sh` 默认读取该实验的 `tetra_meshes/tetra_mesh_binary_search_7.ply` 和 `mast3r_sfm`。
也可以传第三个参数，指定一个新的输出目录。

新照片没有遮罩时，可以自动抠图再提取：

```bash
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
"$ROOT/envs/matcha/bin/python" -m pip install 'rembg[cpu]'
export U2NET_HOME="$ROOT/checkpoints/rembg"
"$ROOT/envs/matcha/bin/python" "$ROOT/tools/plant_extract/auto_extract_plant.py" \
  --mesh "$ROOT/outputs/你的新实验/tetra_meshes/tetra_mesh_binary_search_7.ply" \
  --sfm-dir "$ROOT/outputs/你的新实验/mast3r_sfm" \
  --images-dir "$ROOT/data/你的新实验/images" \
  --output-dir "$ROOT/outputs/你的新实验/plant_auto_01"
```

首次需要下载 rembg 的 u2netp 模型；当前实验使用已有遮罩，没有这一步。
生成遮罩以 COLMAP 图像名匹配，原图与重建照片必须具有相同裁切和朝向。
自动遮罩不保证每张照片正确，建议检查 `generated_masks` 和最终预览。

## 输出与查看

每次输出到新的 `实验目录/plant_auto_日期_时间/`：

- `plant.ply`：提取后的植物和花盆，MeshLab 打开。
- `plant_preview.jpg`：最多四个视角的前后对照，灰色为前景遮罩；上排绿点保留、红点删除，下排保留点的原颜色。这是抽样投影，不是完整网格渲染。
- `report.json`：输入和输出规模、参数、警告、耗时。
- `extract.log`：通过 shell 包装器运行时的日志。

保留原始世界坐标、顶点颜色和其他顶点属性。输入网格请用原始导出文件，不能用之前旋转/裁剪后更换坐标的文件。
不输出面属性、UV 纹理，不补洞、不平滑、不生成叶子。适合当前 MAtCha 的顶点着色网格。

清理参数可直接传给 Python：

- `--min-ratio 0.85`：更严格，但位姿和遮罩误差大时可能误删。
- `--min-ratio 0.6 --dilate 5`：较宽松，更有利于保护边缘，也可能留更多碎片。
- `--min-component-faces 20`：额外删除少于 20 个三角形的连通块；谨慎用于细小叶片。

支持 COLMAP 二进制/文本 `PINHOLE` 和 `SIMPLE_PINHOLE` 相机，畸变相机需要先去畸变。
不支持将高斯 `point_cloud.ply` 作为三角网格输入。

## 验证情况

已运行 CPU 端到端测试，覆盖：COLMAP 二进制和文本位姿、去除较大的环境模型、
保留三个分离的前景连通块（花盆、绿色叶片、非绿色花朵）、原坐标和颜色一致、
相机后方及画面外的点不投票、遮罩缺失时报错、禁止覆盖旧结果。

当前新实验仍在服务器，尚未验证这次真实植物模型的提取效果。结果需以输出预览和 MeshLab 为准。

```bash
python -m unittest discover -s tools/plant_extract -p 'test_*.py' -v
```
