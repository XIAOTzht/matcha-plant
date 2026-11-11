# 固定小车的底面修复

`repair_cart_underside.py` 在**语义分类后、花叶实例与权重生成前**执行。输入是毫米单位的六类语义目录，必须包含真实 RGB 的 `labeled_original_rgb_mm.ply`、匹配的 `labels.npz` 和 `coordinate_system.json`。不要把权重色图当真实 RGB 输入。

该工具面向当前蓝色底盘、四个黄色轮毂的小车。没有植物坐标或植物种类的硬编码。它沿用上层板标定坐标，在下层蓝色底盘拟合平面，保护四个轮子的圆柱包络，裁掉其余向下悬垂面，并用有轮槽的平面补底。植物、花盆、泥土的顶点坐标和三角形必须保持不变，否则验证失败。

底面是**推定的规则几何，不是真实扫描或精确 CAD**。默认底盘厚度假设为 6 mm；轮胎保护半径按黄色轮毂半径 × 1.28 + 1 mm 估计、保护宽度为 33 mm。可以用实测数据修改导出的 `cart_template.json`，以后用 `--template` 指定。复用模板必须确认上层板坐标的轴向和正负方向一致；只知道板子长宽不能消除小车前后朝向歧义。

补面按约 2 mm 间距细分，避免仅用边缘顶点表示大片底面。原重建并不闭合，本工具不承诺整个网格闭合或水密，也不能根据补出的底面认定真实车底有安全通行空间。原结果保留在原目录。

## 执行

Windows PowerShell，`SOURCE` 设为已完成的语义目录，输出必须为新目录：

```powershell
$ROOT = 'F:\plant\matcha_plant'
$PY = 'F:\matcha_mujoco_simulation\envs\mujoco\Scripts\python.exe'
$SOURCE = "$ROOT\outputs\semantic_refinement\plant_20261106_new_rim_shape_v2"
$OUT = "$ROOT\outputs\semantic_refinement\my_cart_flat_revision"
& $PY "$ROOT\tools\plant_extract_with_car\repair_cart_underside.py" --source $SOURCE --output $OUT
if ($LASTEXITCODE -eq 0) {
    & $PY "$ROOT\scripts\plant_contact\run_pipeline.py" --input-dir $OUT --output-dir "${OUT}_contact" --no-parts
}
```

Linux 同一个脚本与依赖即可运行：

```bash
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
PY="$ROOT/envs/plant_semantics/bin/python"
# SOURCE 指向本次完整语义目录，OUT 指向新的结果目录。
"$PY" "$ROOT/tools/plant_extract_with_car/repair_cart_underside.py" --source "$SOURCE" --output "$OUT" &&
"$PY" "$ROOT/scripts/plant_contact/run_pipeline.py" --input-dir "$OUT" --output-dir "${OUT}_contact" --no-parts
```

只需要 numpy、scipy、Pillow、plyfile，CPU 可运行。`render_cart_repair.py` 是另一个可选的真实网格对照渲染工具，需要 Open3D；修复本身不需要 Open3D。

## 输出与下游

- `labeled_original_rgb_mm.ply`：原有区域保留真实 RGB；新增底面为中性灰，不能理解为拍摄恢复的纹理。
- `semantic_mesh_mm.ply`、`labels.npz`：匹配的新几何、顶点/面标签。新增面都属于 cart，小车继续是硬障碍。
- PLY 的 `geometry_inferred`、NPZ 的 `cart_geometry_inferred` / `cart_face_geometry_inferred`：指出补面与裁切产生的几何。
- `source_mapping.npz`：输出顶点/面到输入索引的映射；新增几何为 -1，裁切面记录其来源面。
- `cart_template.json`：本次估计的底盘与四轮参数，可检查、实测修订、保存。
- `validation.json`：坐标保留、标签一致、平面性和导出回读检查。它不是物理精度认证。
- `CART_UNDERSIDE_REPAIR_SUCCESS`：几何与索引导出成功；还需权重流程的 `PLANT_CONTACT_EXPORT_SUCCESS`。

网格拓扑已经变化，不能配用旧的 labels、器官索引、权重或旧路径。修复后的语义目录先进入 `scripts/plant_contact/run_pipeline.py`，再进入路径规划。

`configs/plant_pipeline.json` 的 `cart_underside.enabled` 为本固定小车开启修复，`assumed_chassis_thickness_mm` 控制厚度。新的照片流程会在 `08c_cart_underside` 生成新场景，权重和规划均使用此场景。服务器语义修订入口也会在权重前执行此步骤。其他小车应关闭该选项或提供经过核对的模板；不会在自动检测失败时静默删除车轮。

此次开发在 Windows 本地验证。上传新代码后服务器才有此功能；本地完成不代表服务器已经更新或新路径已经生成。
