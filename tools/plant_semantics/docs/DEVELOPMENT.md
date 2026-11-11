# 代码维护与复现

## 目录职责

| 文件/目录 | 职责 |
| --- | --- |
| `run_v3.ps1`、`segment_v3.cmd` | 推荐完整入口，串联原分割及可选新增视角，写实际结果路径表 |
| `START_HERE_V3.md` | 新任务运行、最终文件、路径布局与标签读取说明 |
| `segment.cmd` | 原 Windows 拖放启动，输出路径兼容 |
| `run_auto.ps1` | 原完整分割入口，解析环境及竖直轴，不追加新增视角 |
| `run_all_five.ps1` | 串行调度全部阶段，任一阶段失败即停止 |
| `plantseg.py` | 网格读取、渲染、基础二维推理、可见性融合、通用导出 |
| `refine_detail.py` | 自动局部视角、PCA 形状辅助、局部融合与盆口约束 |
| `complete_five.py` | 实例证据、伪标签颜色分布、网格边传播及完整赋类 |
| `refine_leaf_boundary.py` | 自动叶片核心、SAM 掩膜、多视角投票与叶缘修复 |
| `refine_leaf_rim.py` | 固定种子、窄网格距离和叶面邻域支持的残余叶缘补全 |
| `reassess_uncertainty.py` | 用既有多相机证据复核历史不确定标记，保持语义与几何不变 |
| `extra_view_review.py`、`run_extra_views.ps1` | 可选新增相机、原 v3 二维推理与冻结标签的不确定标记复核 |
| `validate_complete.py` | 核对原几何、PLY 标签/颜色和元数据 |
| `estimate_axis.py` | 正竖直轴的简单估计 |
| `runtime.ps1`、`setup_local.ps1` | Python 定位与依赖/模型准备 |
| `fetch_models.py`、`check_install.py` | 下载公开资产及检查环境 |
| `tests/`、`check_code.ps1` | 回归检查 |
| `build_package.py` | 从明确清单生成源码 ZIP 及校验文件 |
| `sam3_*.py`、`run_sam3.ps1` | 已有独立 SAM3 试验流程，不被默认入口调用 |
| `docs/` | 算法、部署、验证与维护文档 |
| `experiments/archive/` | 不参与正式流程的历史诊断脚本 |
| `models/`、`third_party/MobileSAM/` | 模型资产与第三方源码 |
| `vendor/`、`runtime.local.json` | 本机依赖与本机 Python 路径，源码包不含这些内容 |

核心模块留在项目根目录，避免改变 `ROOT/models`、第三方导入路径及已生成缓存中的身份记录。根目录 `run.ps1`、`run_detail.ps1`、`run_complete.ps1` 保留旧阶段命令兼容性；普通新任务使用 `run_v3.ps1`，原 `run_auto.ps1` 继续可用。

SAM3 源码一并保留在源码包中，使用说明见 [SAM3.md](SAM3.md)。本次回归与完整入口复跑针对默认流程及新增窄叶缘阶段，不代表完成 SAM3 的模型推理或语义精度验证。

## 输出目录

```text
输出目录/
  baseline/                  全局视图、二维分数和基础融合
  detail/                    局部视图、二维分数和局部融合
  complete_before_boundary/  叶缘修复前的完整五类
  complete_before_rim/       多视角叶缘修复后、窄叶缘补全前
  complete_before_uncertainty/  原 v3 分类及历史不确定标记
  complete/                  最终模型、标签、预览、报告和校验
```

最终 `labels.npz` 包括顶点/面标签、分数、不确定性、边界、推断及叶缘修复标记。各阶段 `manifest.json`、`inference_config.json` 与边缘阶段 `boundary_config.json`、`boundary_records.json` 保存输入、模型、相机与方法记录。分数是启发式证据，不是校准概率。

新 `run_v3.ps1` 将上面的输出树放在 `segmentation/`，追加相机结果放在 `extra_views/`；最终目录默认为 `extra_views/complete/`，`-ExtraViews 0` 时为 `segmentation/complete/`。两步成功后在总输出根目录写 `result_paths.txt/json`，其中 `uncertainty_report` 自动指向实际最终阶段的报告。旧入口的输出布局不变。

窄叶缘阶段另存 `rim_config.json`、`rim_diagnostics.npz` 和 `leaf_rim_refined` 顶点标记。该阶段要求独立输出目录，拒绝重复补全已有 `leaf_rim_refined` 的结果；可对原先的 v2 结果单独运行 `refine_leaf_rim.py --source <v2目录> --output <新目录>`，随后运行 `validate_complete.py`。

## 检查代码

在已配置环境中执行：

```powershell
& '.\check_code.ps1'
# 或显式指定 Python
& '.\check_code.ps1' -PythonExecutable 'C:\path\python.exe'
```

该入口解析项目 PowerShell 脚本并运行 `tests/test_*.py`。12 个几何/边界回归检查覆盖遮挡、拒识、投影、坐标/索引导出、图传播、缺失盆口预览、多相机与器官保护。另有源码分发检查覆盖清单完整性、本机资产排除、逐文件校验及重复构建一致性。单元测试不下载模型，也不衡量语义准确率。

0.2.2 新增 `tests/test_leaf_rim.py` 的 4 项行为测试，共 17 项：合成叶面的误分窄边得到补全，邻近但不连通的管状枝干保持原类，连接的枝干主体不因递归生长被吞入；验证其他类别/保护点及尺度平移一致性、无叶片安全跳过。合成测试使用更粗采样及相应尺度参数，不替代真实植物评价。

0.2.3 新增 `tests/test_uncertainty.py` 的 4 项检查，共 21 项：独立多相机支持才解除标记、分数/语义冲突保留、叶面证据及花苞保护、几何叶缘需掩膜确认、无效相机计数拒绝。器官支持数相减前转为有符号数，避免 uint16 溢出导致错误解除。

0.2.4 新增 `tests/test_extra_views.py` 的 8 项检查，共 29 项：覆盖权重、相机间距、重复覆盖收益递减、空覆盖、仅新增多相机共识才能确认、弃权计入分母、竞争类新增标记、有符号减法、输入字段不变及无效标签/分数/票数拒绝。默认完整分割入口保持原有阶段；新增可选入口的输出在独立目录，保留全部历史字段。

## 打包

```powershell
python build_package.py
```

生成 `plant_semantics_source.zip` 和 `.zip.sha256`。ZIP 中 `PACKAGE_MANIFEST.json` 列出源码版本、文件字节数与 SHA256。模型权重、Python、vendor、输出模型及本机路径不进入包。构建固定 ZIP 元数据；在相同 Python 压缩环境、相同源文件下重复构建应产生相同字节。

`python build_package.py --profile v3` 生成 `plant_semantics_v3_source.zip`，排除独立 SAM3 文件/目录、历史实验和归档；默认 `full` 包继续保留这些来源。两种包使用相同 V3 入口，均不包含权重。0.2.5 新增分发内容检查，回归共 30 项。

新增运行文件时更新 `build_package.py` 的 `ROOT_FILES` 或目录类型清单，并执行分发回归检查。归档文档中的旧绝对路径仅供追溯，不应直接复用为当前部署命令。

## 兼容性及修改范围

本次 0.2.1 整理保留四个核心算法源文件及默认参数；原 `outputs/plant_semantics_complete_v2` 是整理前已生成的结果，未改写其元数据。未来修改提示词、模型或相机时使用新输出目录，并保留版本、输入和权重哈希用于追溯。缓存不是所有源码更改的完整依赖追踪器，不能假设旧缓存一定适用于新算法。

0.2.2 保留四个既有核心模块，增加独立窄叶缘阶段并调整自动调度。最新原植物结果为 `outputs/plant_semantics_complete_v3`，v2 保持可复查。

0.2.3 在 v3 上增加独立的不确定元数据复核阶段，结果为 `outputs/plant_semantics_complete_v3_reviewed`。旧 v3 未改写，新 NPZ 保留旧标记；各原有标签/分数/历史字段逐项核对，只有当前 `uncertain` 允许变化。配置与结果说明见 [UNCERTAINTY.md](UNCERTAINTY.md)。

结果保留输入顶点数量/顺序与三角面索引，导出 XYZ 为 float32；原始 alpha、法线和其他自定义属性未全部复制。若需要扩展格式，应同时更新导出与几何校验。
