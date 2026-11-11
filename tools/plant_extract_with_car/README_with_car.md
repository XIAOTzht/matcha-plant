# 植物与小车自动裁剪 0.2.0

新入口 `auto_crop_with_car.py` / `run_crop_with_car.ps1` / `run_crop_with_car.sh`。
原有 `auto_crop_mesh.py`、`run_crop.ps1` 的植物单独裁剪功能不变。

输入是带顶点 RGB 的三角网格 PLY。CPU 运行，依赖 Python >=3.10、numpy、scipy、Pillow、plyfile；无需 GPU 或模型权重。
同目录必须包含原有 `auto_crop_mesh.py` 和 `auto_extract_plant.py`。

## 方法与适用范围

1. 聚类固定蓝色车体，拟合底盘平面，分别验证法线两个方向的上层蓝板。
2. 以 135×110 mm 上层板的长宽比例、可见边界和平面性验证目标；从板子附近向上搜索花盆/植物几何，**不使用植物颜色筛选**。在底盘下方寻找轮子接触的桌面/凳面。
3. 按车体和植物范围生成有余量的裁剪框，移除支撑面及外部环境。
4. 只清除很小的孤立碎片，不使用“只留最大连通块”，避免丢失断开的叶片、花朵和车部件。

适用于固定蓝色小车、同一上层支撑板和可见平坦支撑面的拍摄布置。黄叶、橙花、白花、灰色植物等都不再因颜色被排除。位置、旋转、整体比例由输入推断，没有硬编码某株植物的坐标。它是通用于该布置的几何裁剪器，不能保证任意植物、任意背景都正确：遮挡严重、车体重建缺失、多辆蓝色车、植物与背景粘连或支撑面不可见时仍可能失败或需要人工复核。

默认向上搜索范围为底盘长度的 4 倍、水平半径为底盘长度的 2 倍；触及搜索上限会报错，避免静默裁断。可使用 `--max-height-cart-lengths`、`--max-radius-cart-lengths` 显式调整。断裂很大或离主体很远的叶片无法仅靠邻近几何可靠判断归属，需检查预览。裁剪保留目标包围范围内网格，并非精细语义分割；处于同一范围的背景仍可能保留。`--plate-length-mm`、`--plate-width-mm` 默认 135、110，须匹配标定步骤。

轮胎与凳面接触处可能裁掉薄层；不补洞。原始坐标和顶点属性保留，面属性/UV 不导出。裁剪不会修复重建本身的缺失细节。
本步骤不是语义分割，也尚未使用 **上层蓝板 135×110 mm** 标定实际尺寸。

## Windows

```powershell
& 'F:\植物建模\tools\plant_extract\run_crop_with_car.ps1' `
  -InputMesh 'F:\植物建模\outputs\with_car\tetra_mesh_binary_search_7.ply' `
  -OutputDirectory 'F:\植物建模\outputs\with_car\plant_car_new'
```

输出目录必须没有已有的 `plant_car.ply` 或 `report.json`，避免覆盖结果。

## 服务器

把本目录中的新脚本与两个 Python 辅助文件放在项目 `tools/plant_extract` 中，然后执行：

```bash
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
bash "$ROOT/tools/plant_extract/run_crop_with_car.sh" \
  "$ROOT/outputs/with_car_20261021/tetra_meshes_gpu1_ds05/tetra_mesh_binary_search_7.ply" \
  "$ROOT/outputs/with_car_20261021/plant_car_auto_v1"
```

默认使用已安装的 `envs/plant_semantics/bin/python`。本地已运行验证，服务器尚未执行本次新裁剪。

## 输出

- `plant_car.ply`：保留植物、花盆、蓝板、黑布、小车的网格。
- `plant_car_preview.jpg`：三个方向的抽样点预览；点的空隙不等于网格洞。
- `detection_preview.jpg`：红色表示被移除的几何。
- `source_vertex_indices.npy`：每个输出顶点在原网格中的索引。
- `report.json`：参数、检测框、支撑面、数量及限制说明。

2026-10-21 当前模型试跑：13,001,469 顶点 / 25,995,241 三角面，裁剪后 2,148,841 顶点 / 4,294,728 三角面。结果位于 `outputs/with_car/plant_car_auto_v1`。
