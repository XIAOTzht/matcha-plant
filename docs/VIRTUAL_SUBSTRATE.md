# 盆内虚拟基质面（明确的仿真假设）

“泥土／基质”指盆口内承载植物的上表面，颜色不构成判断依据。盆外壁、盆底和盆沿仍为花盆。只有缺少基质标签且用户选择假设土面时，才运行本工具；不会修改已有表面类别。

默认从毫米坐标系的盆壁截面估计内轮廓，向内缩 6 mm，在估计盆沿下方 10 mm 生成间距约 2 mm 的平面三角网格。适用单个紧凑、近似凸形的盆；实际填充高度和壁厚未经测量。各方向覆盖不足或盆形状不支持时拒绝。喷口在新增面上方 10 mm，现有浇水方向、5–15 mm 距离和碰撞检查继续生效；没有可行姿态仍会失败。

所有旧顶点、三角面和标签按原顺序保留，新几何只追加。`virtual_substrate.json`、PLY 的 `virtual_substrate` 字段和 NPZ 的 `virtual_substrate` / `face_virtual_substrate` 掩码记录假设来源；原小车底部推断标记保留。新增分类置信度为 NaN、有效位为 false，不虚构识别证据。基质是硬障碍，只允许喷口靠近，不能穿透。

Windows PowerShell 一次上传并启动服务器修订任务：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\plant\matcha_plant\maintenance\start_virtual_substrate.ps1"
```

默认复用 `plant_20261108_coordinated` 的成功建模结果，建立新任务 `plant_20261108_virtual_soil_v1`。原任务不修改，不重新跑照片建模。先检查重用网格、语义和标定的校验和，再生成基质、重算分花叶与权重、使用已选的 coordinated 规划、动态复查并渲染。新旧源码签名不通过强行修改原 state 来兼容；修订任务保存自己的输入签名和上游副本。

Linux 可直接启动（先完成代码上传）：

```bash
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
"$ROOT/envs/plant_semantics/bin/python" -u "$ROOT/maintenance/continue_with_virtual_substrate.py" --root "$ROOT" --run plant_20261108_coordinated --revision plant_20261108_virtual_soil_v1 --assume-pot-interior-substrate --submit
```

进度日志：`$ROOT/logs/pipeline_plant_20261108_virtual_soil_v1.log`。

成功条件：新目录 `$ROOT/outputs/pipeline/plant_20261108_virtual_soil_v1/PHOTO_TO_PATH_SUCCESS` 存在，且仿真报告的检查全部通过。`VIRTUAL_SUBSTRATE_SUBMITTED` 仅表示 tmux 提交；不是规划完成。成功标志和结果索引均标记 `assumed_virtual_substrate`，不表示真实土面已恢复。

完成后下载：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\plant\matcha_plant\maintenance\fetch_plant_results.ps1" -RunName "plant_20261108_virtual_soil_v1" -OutputDirectory "F:\plant_results_virtual_soil_v1" -Extract
```

`--resume` 只复用新修订任务中同一输入/代码/参数的已完成阶段。更改填充高度、轮廓、源码或配置后另选 `--revision`。`--prepare-only` 只用于场景检查，不会输出完整成功标志。重建有孔洞的地方仍不是已知自由空间，虚拟基质不保证真实隐藏内容、实物可达性或真实浇水安全。

如果旧版在 `02_scene` 因 `at least 4 vertices required` 失败，单个三角形的颜色分块会在原平面内细分成三个三角形。显示表面不变，原始模型、标签、权重、规划碰撞网格不变。修复上传和恢复：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\plant\matcha_plant\maintenance\start_virtual_substrate.ps1" -Resume -RepairDisplayMesh
```

恢复助手只接受这一项经过校验的显示代码变更，并检查所有已完成文件的 SHA256、输入、配置和原启动器。先备份两个状态文件，再更新代码签名；保留失败场景目录，重新执行场景构建及后续规划，不重算已经完成的土面和权重。无关代码/配置变更会拒绝复用。
