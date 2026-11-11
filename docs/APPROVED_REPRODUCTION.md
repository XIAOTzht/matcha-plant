# 复刻 11 月 7 日认可版本

当前默认器官识别恢复为 `organ-rgb-shape-review-2.0`，使用原来的 24 视角证据、形状规则、表面一致性和叶缘处理。`organ_recovery.py` 实验规则及测试已移入备份目录。MAtCha 建模的四个主入口及对应配置与旧上传包一致。

本次复刻固定旧 RGB 网格、初始标签、坐标系、保存的识别证据和代码版本；重新计算分类、小车底修整、分器官及权重，再用认可的协同轨迹重新执行 MuJoCo 动态复查。它没有重新执行照片建模、神经网络推理或路径搜索。这几个阶段不能混称为重新从照片跑完一遍。

本地已验证：分类的 22 项 NPZ 数组、小车 RGB/语义网格与映射、器官 8 项数组、权重 24 项数组均与旧结果逐项相同。3422 个 5 ms 仿真状态、18 项动态检查通过，时长 `17.10500000000088` 秒，接触惩罚 `21.628808532733345`，目标距旧泥土标签面 10 mm。旧泥土标签是原网格上的稀疏片面，复刻保留原样，不将它表述为已确认的完整真实土面。

本地结果：

- `outputs/reproduction/approved_20261107_check_v1/cart/semantic_mesh_mm.ply`
- `outputs/reproduction/approved_20261107_check_v1/contact/weights/contact_risk_heatmap.ply`
- `outputs/reproduction/approved_20261107_check_v1/contact/weights/contact_risk_heatmap_preview.jpg`
- `simulation/outputs/reproduction/approved_20261107_check_v1/replay/trajectory_replay.gif`
- `simulation/outputs/reproduction/approved_20261107_check_v1/replay/validation_report.json`

原认可结果始终保留。最近实验源码备份在 `backups/local_code/restore_approved_20261109`。

## 在服务器复现

Windows PowerShell 执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\plant\matcha_plant\maintenance\start_approved_reproduction.ps1"
```

此命令先上传恢复后的默认代码，再上传独立的**输入与代码包**（约 201 MiB），含旧 RGB 网格、证据、小车模板、认可路径及其仿真场景。不覆盖服务器已有模型，不启动照片重建。需输入服务器密码。

服务器查看：

```bash
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
tail -n 30 "$ROOT/logs/reproduction_approved_20261107_reproduced_v1.log"
cat "$ROOT/outputs/reproduction/approved_20261107_reproduced_v1/result_paths.json"
```

只有 `APPROVED_REPRODUCTION_SUCCESS` 才代表全部复现检查通过。失败会输出 `REPRODUCTION_FAILED.json`，不会把不一致结果标为成功。不同系统的数值若不一致会明确停止，不悄悄放宽阈值。

`upload_plant_pipeline.ps1` 仍是纯代码更新，会带上恢复的旧分类代码与冻结版本；它不会附带本次模型和识别证据。后续全新照片使用同一方法，但不声称不同输入或重建一定得到相同网格。若要复核上游建模，可从服务器旧 run `plant_20261106_new` 的 `04_mesh/attempt_001`、`05_crop/attempt_002`、`06_metric/attempt_001` 和对应状态记录追溯。
