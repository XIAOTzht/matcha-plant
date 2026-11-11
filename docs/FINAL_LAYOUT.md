# 最终工作目录

目录不重命名，保留现有源码依赖的相对路径。

| 路径 | 用途与保留原因 |
| --- | --- |
| `MAtCha/` | 重建、网格提取、模型参数和已编译扩展，整体保留 |
| `tools/plant_extract/` | 旧名称下仍有裁剪辅助代码，属于运行依赖 |
| `tools/plant_extract_with_car/` | 植物和车裁剪、上层蓝板 135×110 mm 标定 |
| `tools/plant_semantics/` | 植物语义代码及其模型、第三方依赖 |
| `tools/plant_semantics_with_car/` | 小车分类及螺母修正 |
| `tools/plant_organ_instances/` | 通用花叶实例候选、中心与面积 |
| `tools/contact_weights/` | 最新权重计算、距离和持续接触代价、配色 |
| `scripts/plant_contact/` | 器官与权重流水线入口 |
| `maintenance/run_reconstruction.sh` | 新照片重建通用入口 |
| `maintenance/finish_server.sh` | 本次更新结果和最终整理总入口 |
| `data/with_car_20261021/` | 当前照片及输入 |
| `outputs/with_car_20261021/` | 当前场景和最新权重，详见下表 |
| `envs/` | 服务器现有环境，保持原位 |
| `cache/`, `weights/`, `third_party/` | 可能被流程加载的模型及运行依赖，整体保留 |
| `backups/` | 历史备份、旧代码、此次清理备份和总入口日志 |
| `docs/` | 操作说明、公式、整理后生成的路径索引 |

当前场景保留：

| 子目录 | 内容 |
| --- | --- |
| `mast3r_sfm/` | 相机与几何重建状态 |
| `free_gaussians/` | 训练结果，可重新提取网格 |
| `tetra_meshes_gpu1_ds05/` | 成功提取的完整场景网格 |
| `plant_car_auto_v1/` | 植物与车裁剪网格及报告 |
| `plant_car_metric_v1/` | 毫米模型和标定报告 |
| `plant_car_semantics_v2/` | 螺母已归车的最终语义 |
| `contact_latest_<本次成功运行>/` | 最新花叶拆分、权重数值、模型、预览及运行日志 |
| `latest_contact.json`, `latest_contact_paths.txt` | 最新权重路径指针 |

其余场景、当前场景中更早的权重、语义 v1、失败提取、旧实验输入、
一次性 `scripts/with_car_stage1` 和 `scripts/other_stage`、旧整理和测试脚本、
旧导航说明、旧源码导出、`incoming` 上传包、工作日志和临时文件，
先归档到一个 TAR 并校验，再删除原目录内容。
`tools/` 的源码和资产不按名称判断新旧，只清理可重建字节码。
未知根目录资产保留，避免误删尚未确认的运行依赖。
已有 `backups/` 不递归重新打包，也不删除。
语义 v2 的历史报告可能仍引用 v1，历史文件可以从清理备份查回；当前权重直接读 v2。
未来新增其他需要长期保留的场景时，请先更新 `final_organize.py` 的保留规则。

## 恢复旧内容

清理备份中 `files.json` 记录原项目相对路径和每个文件的 SHA256。
`manifest.json` 记录计划、已删除项目和执行状态。
不要直接向正在使用的项目根解包，先恢复到单独目录：

```bash
# 将下方路径换成本次实际输出的备份目录
BACKUP="$ROOT/backups/final_cleanup_实际时间"
(cd "$BACKUP" && sha256sum -c old_content.tar.sha256)
RESTORE="$ROOT/backups/restored_$(date +%Y%m%d_%H%M%S)"
mkdir "$RESTORE"
tar -xf "$BACKUP/old_content.tar" -C "$RESTORE"
```

恢复目录中的链接保留原链接目标；不会将目标文件复制进归档。
比较所需文件后再拷回原位置。环境和大模型没有从活动目录移走，不需要从此次清理备份恢复。
