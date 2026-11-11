# V3 植物 + 小车语义分割适配器 0.1.0

不改动原 `tools/plant_semantics`。依赖其现成的 `run_v3_server.sh`、环境及模型。
新增代码只依赖 numpy、scipy（测试）、Pillow、plyfile，实际 V3 使用现有 GPU 环境。

## 本次策略

1. 校验未缩放裁剪网格、毫米网格、标定报告之间的尺寸、三角面和颜色对应。
2. 使用上层蓝板中心和法向定位小车，板面以上2 mm以内及以下暂归小车；分界上下3 mm标记需复核。
3. 提取完全位于分界上方的植物/花盆三角面，按蓝板法向摆正为 +Y，保留原比例供 V3 推理。
4. 调用原五类 V3 和新增视角复核，将分类映射回原网格顶点顺序。
5. 合入小车第6类，输出原重建坐标方向下的毫米网格。完整网格的顶点、三角面都保留。

这是相对蓝板的几何分类，不是训练新小车检测器，也没有做小车模板配准。新模型会重新识别板位姿，不固定世界坐标框。
当前假设植物与花盆在上层板上方，小车/黑布/板在下方；下垂到板下的叶片、伸高的车体配件需要改进分离方法。
上层板螺钉、花盆底部和接触面附近标记 uncertain，不声称确定分类。
未进入植物三角面的孤立顶点保留为0类未知。不要把未知或不确定区域当作可通行空间。

## 已完成的本地检查

真实模型分离预览已生成并查看：762,681个顶点送入V3，1,386,143个暂归小车，17个未处理顶点保留未知。
接触复核带103,631点。本地没有完成这株植物的GPU语义推理；准备预览不是最终语义结果。
测试覆盖姿态/比例变换、接触不确定性、未知保留、合成V3结果合回、完整毫米几何/颜色一致性和错误缓存拒绝。

## 上传与服务器执行

本地 PowerShell：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\植物建模\tools\plant_semantics_with_car\upload.ps1"
```

服务器（GPU编号1只是参数；先确认该卡空闲）：

```bash
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
CROP="$ROOT/outputs/with_car_20261021/plant_car_auto_v1"
METRIC="$ROOT/outputs/with_car_20261021/plant_car_metric_v1"
OUT="$ROOT/outputs/with_car_20261021/plant_car_semantics_v1"
TOOL="$ROOT/tools/plant_semantics_with_car"
mkdir -p "$ROOT/logs"
sed -i 's/\r$//' "$TOOL/run_server.sh"
bash -n "$TOOL/run_server.sh" &&
tmux new-session -d -s semantics_with_car \
  "bash '$TOOL/run_server.sh' '$CROP/plant_car.ply' '$METRIC/plant_car_mm.ply' '$METRIC/scale_report.json' '$OUT' 1 12 > '$ROOT/logs/with_car_semantics.log' 2>&1"
tail -f "$ROOT/logs/with_car_semantics.log"
```

旧流程输出 `PLANTSEM_SERVER_SUCCESS` 后仍需合并；以 **WITH_CAR_SEMANTICS_SUCCESS** 为整条流程完成标记。
同输入同参数可复用 prepared 和原V3缓存；更换输入、视角参数或规则用新输出目录。有 complete 目录时不会覆盖。

## 最终输出

位于 `outputs/with_car_20261021/plant_car_semantics_v1/complete/`：

- `semantic_mesh_mm.ply`：毫米单位、六类颜色。
- `labeled_original_rgb_mm.ply`：毫米单位、原RGB与标签。
- `labels.npz`：顶点/面标签、不确定性、label_source、接触带、植物索引。
- `coordinate_system.json`：单位、原始/推理坐标变换、标尺中心和方向。
- `semantic_preview.jpg` / `uncertainty_preview.jpg`：语义图和黄色复核点。
- `classes.json` / `report.json` / `validation.json`：类别、统计及导出一致性。

类别：0未知、1茎枝、2叶片、3花/花苞、4花盆、5土壤/基质、6小车（青色，包括蓝板、黑布、轮子和附件）。
`class_scores` 有6列，对应1..6；小车区域为NaN且 `class_score_valid=False`，没有伪造100%模型置信度。V3分数也不是校准概率。
模型输出不确定性会保留，接触带追加 uncertain。三角面跨类别时 `face_boundary=True`；无多数类别且未由V3处理的面保留未知。
完整V3中间记录位于 `plant_v3/`，没有强行把它们伪装为六类数据。

## 路径规划衔接

最终模型与毫米裁剪模型的坐标/顶点顺序/面索引完全相同，分类只添加属性。后续路径点、安全距离也按毫米处理。
毫米标定只解决比例；模型到机器人基座/实际场地的位姿配准尚未完成，本步骤不生成或执行机器人路径。
原相机若继续用于毫米模型投影，平移/相机中心也需同尺度转换；原尺度模型与推理摆正模型留作中间文件。
