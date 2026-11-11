# 植物建模与接触风险权重

照片重建、植物与小车裁剪、真实尺寸标定、语义分类、花叶实例候选和接触风险权重。
MuJoCo 仿真项目暂时独立，本次不在服务器更新包内。

## 从哪里开始

- [服务器部署、整理与 GitHub](docs/SERVER_MAINTENANCE.md)
- [最终目录与旧内容恢复](docs/FINAL_LAYOUT.md)
- [权重公式、文件和通用运行方法](docs/PLANT_CONTACT_GUIDE.md)
- [花叶候选拆分](tools/plant_organ_instances/README.md)
- [MAtCha 上游项目](MAtCha/README.md)

## 目录

| 路径 | 用途 |
| --- | --- |
| MAtCha/ | 重建源码、第三方模块与服务器已有模型资产 |
| tools/plant_extract_with_car/ | 植物与小车裁剪、上层蓝板 135×110 mm 尺度标定 |
| tools/plant_semantics/ | 植物语义推理 |
| tools/plant_semantics_with_car/ | 小车类别、螺母归车修正 |
| tools/plant_organ_instances/ | 通用花叶几何实例候选与中心/面积估计 |
| tools/contact_weights/ | 权重、距离和持续接触评分、共享配色 |
| scripts/plant_contact/ | 通用流水线和当前场景 Linux 入口 |
| data/ | 原始照片、输入数据，Git 排除 |
| outputs/ | 重建、语义、实例、权重结果，Git 排除 |
| envs/ | 服务器环境，Git 排除，不能直接移作 Windows 环境 |
| maintenance/ | 上传、校验、整理、源码快照导出 |
| docs/ | 操作说明及运行生成的结果索引 |
| backups/ | 更新备份和整理归档，Git 排除 |
| exports/ | 生成的独立 GitHub 源码目录，Git 排除 |

服务器保留现有两个环境：`envs/matcha` 用于重建，`envs/plant_semantics` 用于语义与权重。
本次更新复用它们，不安装或覆盖依赖。重新搭建语义环境请查看
`tools/plant_semantics/setup_server.sh`；重建还需要对应版本的 PyTorch/CUDA 与原生扩展。
仅下载 GitHub 源码并不包含大模型、照片、结果或已经编译的环境。

## 服务器计算当前场景的新权重

```bash
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
bash "$ROOT/scripts/plant_contact/run_latest_server.sh" --check-only
bash "$ROOT/scripts/plant_contact/run_latest_server.sh"
cat "$ROOT/outputs/with_car_20261021/latest_contact_paths.txt"
```

成功才更新最新结果指针；历史模型不覆盖。数据导出完成后仍需检查自动器官分区。
配色是权重预览，不代表测得植物损伤、接触力或路径安全。

## 代码测试

在具备 NumPy、SciPy、Pillow、plyfile 的 Python 3.10+ 环境中运行：

```bash
python tools/contact_weights/test_organ_weights.py
python scripts/plant_contact/test_deployment.py
python maintenance/test_server_project.py
python maintenance/test_final_organize.py
```

第三方源码遵循各自 LICENSE；发布时保留上游许可证。
