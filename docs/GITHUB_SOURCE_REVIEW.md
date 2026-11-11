# 本次 GitHub 源码整理说明

本目录来自 2026-11-11 的服务器源码快照。服务器原始 ZIP、下载校验记录保留在本地 github_review 目录；本发布目录单独管理，不修改服务器模型、环境或结果。

## 已检查

- 原始 ZIP 的 SHA256 与服务器导出输出一致，2,223 个文件逐项匹配清单。
- 500 个 Python 文件语法、28 个 JSON、3 个 XML 解析通过。
- 默认配置引用的场景、时间参数与协同配置存在。
- SO-101 XML 引用的 18 个网格文件完整。
- 认可版本的 50 个冻结源码与其原始校验清单一致。
- 流水线 11 项、权重 6 项、轨迹/规划 14 项、关节协同 7 项测试通过，共 38 项。

上述检查使用本机现有环境；没有在新 Linux 机器安装依赖、下载模型并重新跑照片全流程。不能把源码完整性检查写成从零部署成功。

## 运行入口

从根目录 README.md 和 PHOTO_TO_PATH_GUIDE.md 开始。旧的 SERVER_MAINTENANCE.md 与 LEGACY_SERVER_README.md 记录的是早期“不包含仿真”的部署范围，不代表当前发布范围。当前版本包含 simulation/ 和 configs/。

现有 maintenance/server_project.py 的 export-github 子命令属于旧导出实现，会漏掉仿真与顶层配置；新的源码导出使用 maintenance/export_github_source.py。

## 两条不同的流程

1. 新照片：scripts/plant_pipeline/photos.py 及 run_pipeline.py，执行重建到规划。
2. 认可结果复现：maintenance/approved_reproduction.py，使用保存的原始 RGB 几何、推理证据、冻结源码与认可路径。其输入胶囊没有纳入 Git，不能只克隆源码就声称复现了原始数值。

models/ 下的预训练模型、重建网格、照片和历史结果均单独存放。权重公式配置 JSON 是算法参数，会提交；模型检查点权重不会提交。

## 主机与依赖

部分 Windows 客户端保留原部署的服务器地址、账号及项目目录默认值，使用时通过参数覆盖；这些默认值不是密码或认证信息。Linux 主流水线从当前项目根目录解析路径。旧的手动修复脚本及场景示例需要结合文档理解，不能作为通用安装器。

MAtCha 与各子项目保留原始许可证，不能将整棵第三方代码统一标为 MIT。CUDA 扩展需在目标环境重新编译；仓库不包含服务器的 .so 文件。

## 后续版本管理

SOURCE_EXPORT_MANIFEST.json 记录的是导出当时的服务器源码，不是此后每次 Git 提交的全部状态。发布目录中的辅助脚本和文档变化另记在 PUBLICATION_CHANGES.json。后续修改以 Git 历史为准。
