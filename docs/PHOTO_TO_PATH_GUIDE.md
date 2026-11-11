# 从照片到浇水路径：服务器统一入口

本套脚本把现有流程连接起来：**照片 → 重建 → 植物和小车裁剪 → 毫米标定 → 六类语义分割 → 螺母归入小车 → 花叶实例 → 接触权重 → SO-101 路径 → 平滑运动验证 → 轨迹线和动画**。

服务端根目录默认：`/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant`。所有新缓存、任务和环境均在此目录下。

## 1. Windows 上传代码（只做一次，代码升级时重做）

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\plant\matcha_plant\maintenance\upload_plant_pipeline.ps1"
```

会打包、校验、上传并自动安装，无须手动解压。包含仿真代码、SO-101 网格资源和 Apache 许可证、新权重代码和完整流程入口；不包含历史模型、Python 环境、训练权重或旧结果。服务器被替换的代码保存到 `backups/code_updates/plant_pipeline_*`。安装不会开始建模。

仅检查本地部署包，不连接服务器：加 `-CheckOnly`。

连接 `10.7.88.77` 需要处于能访问服务器的校园网/VPN。脚本支持 `-ServerAddr`、`-ServerUser`、`-Port`、`-RemoteRoot`。密码由 OpenSSH 询问，不写入配置。

## 2. 服务器安装规划环境并检查（只做一次）

在 Linux 服务器终端执行：

```bash
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
bash "$ROOT/maintenance/setup_plant_pipeline.sh"
```

保留已工作的 `envs/matcha` 和 `envs/plant_semantics`；新增 Linux `envs/mujoco`，推荐 Python 3.12。不能复制 Windows 环境到 Linux 使用。入口会按步骤自动切换环境，因此用户只需运行一个任务命令。

脚本优先使用 `python3.12`，否则使用已有 conda 创建位于项目内的环境。也可指定：

```bash
BASE_PYTHON=/absolute/path/to/python3.12 bash "$ROOT/maintenance/setup_plant_pipeline.sh"
```

最后应出现 `OFFSCREEN_RENDER_OK`、`SIMULATION_RUNTIME_OK`、`PIPELINE_RUNTIME_OK`、`PLANT_PIPELINE_SETUP_OK`。它会检查旧重建 CUDA 扩展、分割依赖、已有模型权重和新规划环境。缺少原有预训练模型或 CUDA 扩展时会明确报出路径，不会冒充安装成功。

安装程序先测试 GPU EGL；失败后尝试 NVIDIA EGL vendor 和 CPU OSMesa。每个测试在独立进程中真实渲染图像，日志位于 `logs/render_setup/`。若系统缺少可用后端，程序会通过已有 conda 在项目内的 `envs/mujoco_render_libs` 安装 Mesa 运行库，再重试，不使用 sudo 或修改系统 NVIDIA 驱动。

渲染后端测试使用简单场景，默认每个后端允许 300 秒，不把机器人网格加载混入后端选择；后续完整检查仍加载 SO-101 和全部规划依赖。日志即时记录每个导入、场景编译和渲染步骤，长时间未结束时每 60 秒打印 Python 调用栈，超时也保留已产生的输出。NFS 首次读取很慢时可使用 `RENDER_PROBE_TIMEOUT=600 bash "$ROOT/maintenance/setup_plant_pipeline.sh"`。已安装 Mesa、只想复测时可加 `INSTALL_RENDER_LIBS=0`，复用已有环境且不触发 Mesa 安装。

通过的后端及运行库路径保存为 `configs/server_render_runtime.json`。后续从新的 SSH/tmux 窗口提交任务也会自动读取，因此无须每次手动 export。OSMesa 仅让预览和动画使用 CPU 渲染，重建和语义分割继续使用指定 GPU。设置 `INSTALL_RENDER_LIBS=0` 可只诊断、不安装备用运行库；全部后端仍失败时不会输出 setup 成功标记。渲染配置属于服务器本机配置，不会被上传包覆盖。

如果安装末尾出现 `Cannot initialize a EGL device display`，说明依赖包已装好，但渲染检查未通过，不能视为全流程就绪。更新代码后重跑相同 setup 命令即可复用现有环境。必须同时看到 `PLANT_PIPELINE_SETUP_OK` 才进入照片提交步骤。

## 3. 每次只上传照片并启动

在 Windows PowerShell 执行（RunName 每次用新的英文数字名称）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "F:\plant\matcha_plant\maintenance\submit_plant_photos.ps1" `
  -ImageDirectory "F:\植物建模\data\with car" `
  -RunName "plant_20261103_01" -GpuId 1
```

会验证 JPG/JPEG/PNG 是否能解码，用纯数字文件名避免中文路径问题，生成 SHA256 清单，上传后再次校验，再由 tmux 后台运行。照片数量自动识别，不固定为 16。照片应围绕整个植物和同一台小车拍摄，覆盖泥土、上层蓝板和植物各面；程序要求至少 3 张，实际建模应使用充分多视角照片。

`-CheckOnly` 只生成本地照片包；`-UploadOnly` 只上传并校验、不启动。上传 ZIP 本地保留在 `incoming/photo_uploads/`；项目不会清理历史照片和结果。

## 4. 查看进度与续跑

服务器：

```bash
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
RUN=plant_20261103_01
tail -f "$ROOT/logs/pipeline_${RUN}.log"
```

看到 `STAGE_START 01_sfm` 等表示进入该步骤，`STAGE_SUCCESS` 表示已检查产物。`10_planning` 内部还分场景、姿态、路线候选、加权规划、姿态优化、动力学和渲染。日志暂时不更新不能单独证明卡死。`Ctrl+C` 仅退出 tail，任务继续。

```bash
cat "$ROOT/outputs/pipeline/$RUN/pipeline_state.json"
cat "$ROOT/simulation/outputs/jobs/$RUN/pipeline_state.json"
```

只有出现 `outputs/pipeline/$RUN/PHOTO_TO_PATH_SUCCESS` 才表示整条链路完成。失败时查看 `PIPELINE_FAILED.json` 和当前步骤的 `console.log`。没有可行路径、碰撞检查失败、渲染失败均不会标记全流程成功。

修复环境/问题后，无须重传照片或重跑已经成功且校验一致的步骤：

```bash
"$ROOT/envs/plant_semantics/bin/python" "$ROOT/scripts/plant_pipeline/photos.py" start \
  --run "$RUN" --gpu 1 --resume
```

也可从 Windows 重新运行同一个照片提交命令并加 `-Resume`。完成阶段会校验其输出再复用；失败阶段写入新的 `attempt_002`，保留旧日志。输入照片、主配置或策略改变需要新的 RunName。一次运行一个全流程任务，避免共享显卡和全局场景指针竞争。

遇到 `OSError: AF_UNIX path too long` 时，旧流程可能仍在 tmux 中等待数据，并非正常运行。更新后的程序将 `TMPDIR`、`TEMP`、`TMP` 统一到项目内的短目录 `t/`；启动前用重建环境实际验证 Unix socket 和文件描述符共享，成功会显示 `PIPELINE_IPC_OK`。不使用项目外的 `/tmp`，不修改图片和算法参数。

先停止受影响的这一个任务及其子进程（保留所有结果和日志），更新代码后再续跑：

```bash
"$ROOT/envs/plant_semantics/bin/python" "$ROOT/maintenance/stop_plant_pipeline.py" --run "$RUN"
"$ROOT/envs/plant_semantics/bin/python" "$ROOT/scripts/plant_pipeline/photos.py" start --run "$RUN" --gpu 1 --resume
```

仅本次已审阅的临时路径修复允许迁移原代码校验签名，并在迁移前保存状态副本、校验已成功阶段的全部产物。其他代码、输入或配置变化仍拒绝复用。停止工具只匹配指定任务及其子进程；如果有其他任务占锁、进程仍等待 NFS I/O 或 tmux 中有无关子进程，将停止并保留现场。

### 裁剪失败的诊断包

如果 `04_mesh` 已成功但 `05_crop` 出现 `No blue cart with a plant above it found`，先检查网格证据，不重复建模。Windows 执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\plant\matcha_plant\maintenance\diagnose_plant_crop.ps1" -RunName "plant_20261106_new"
```

该命令上传独立的只读诊断工具到 `incoming/`，导出默认 30 万个顶点的 XYZ/RGB、场景预览及各候选的淘汰原因，校验后下载并解压到 `outputs/diagnostics/<RUN>_<ID>/`。它不改流程源码、状态、原网格或标签，也不重跑任何阶段；因此不会使断点续跑的代码签名失效。诊断包仅供定位问题，不作为裁剪或语义结果使用。

如果服务器已经出现 `CROP_DIAGNOSTICS_READY`，但下载失败，在相同 Windows 命令末尾加 `-DownloadOnly`，直接下载已有诊断包，无须上传代码或重新导出。脚本通过 SCP 获取 `latest.json` 并在本地解析，避免 Windows PowerShell 5.1 经过 SSH/Bash 执行嵌套引号命令时丢失引号。

### 固定小车裁剪升级

0.2.0 裁剪先定位蓝色小车与上层板，从板子附近向上扩展花盆、枝叶和花的几何，不再要求绿色/粉色植物。默认水平半径 2 倍底盘长度、板上高度 4 倍底盘长度；触及范围边界会拒绝输出。与背景连接、严重断裂或板子重建不足时仍需检查，不能把所有上方物体直接当作语义植物。下方小车仍完整保留，供下一步语义分割与硬障碍分类。

上传新版后，对已经完成 01–04、但 05_crop 失败的运行，在服务器执行：

```bash
"$ROOT/envs/plant_semantics/bin/python" "$ROOT/maintenance/resume_cart_crop.py" --run "$RUN" &&
"$ROOT/envs/plant_semantics/bin/python" "$ROOT/scripts/plant_pipeline/photos.py" start --run "$RUN" --gpu 1 --resume
```

该一次性迁移工具只允许已审阅的两个裁剪文件变化，校验照片、配置、策略和 01–04 所有产物后备份状态、迁移代码签名。任何其他源码变化、已有成功裁剪/后续阶段或产物不一致都拒绝迁移。随后正常 `--resume` 重新尝试 05，保留之前失败的日志，不重复建模。新 RunName 无须迁移。

## 5. 下载结果

Windows：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "F:\plant\matcha_plant\maintenance\fetch_plant_results.ps1" `
  -RunName "plant_20261103_01"
```

自动在服务器导出并下载 `results.zip`，校验 SHA256，默认保存到 `F:\plant\matcha_plant\outputs\downloads\plant_20261103_01`。下载包含语义模型、权重热力模型和数组、坐标信息、轨迹 CSV/NPZ、验证报告、动画和轨迹线图；不是整个训练项目备份。ZIP 内保留项目相对目录，详细文件位置见包内 `result_paths.json`。模型和数值结果可离线查看；交互式重新仿真仍需服务器场景及其资源。

也可用 WinSCP 下载 `exports/$RUN/results.zip`，或按 `outputs/pipeline/$RUN/result_paths.json` 查具体文件位置。

如果已出现 `RESULTS_EXPORTED`，但下载连接断开，加 `-DownloadOnly` 下载服务器已生成的包；无需重新导出或运行仿真。连接在 SCP 阶段关闭时可加 `-LegacyScp` 尝试传统 SCP 协议。下载脚本默认最多尝试两次，先取校验文件，再将压缩包传到独立暂存目录；只有 SHA256 通过才发布为 `results.zip`。断线后可以重复相同命令，未校验的文件不会被解压。同名但校验值不同的既有压缩包会保留并报错，此时换一个下载目录。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\plant\matcha_plant\maintenance\fetch_plant_results.ps1" -RunName "plant_20261108_virtual_soil_v1" -OutputDirectory "F:\plant_results\plant_20261108_virtual_soil_v1" -Extract -DownloadOnly -LegacyScp
```

## 6. 关键目录和参数

| 内容 | 项目内位置 |
|---|---|
| 全流程设置、显卡、板尺寸、规划预算 | `configs/plant_pipeline.json` |
| 新照片及校验清单 | `data/pipeline/<RUN>/` |
| 重建至权重的阶段结果 | `outputs/pipeline/<RUN>/01_sfm` 至 `09_weights` |
| 完整结果索引 | `outputs/pipeline/<RUN>/result_paths.json` |
| 规划和仿真结果 | `simulation/outputs/jobs/<RUN>/` |
| 喷嘴形状、机器人/植物虚拟布局和初始关节 | `simulation/configs/weighted_attitude_search_scene.json` |
| C3 七次曲线、关节速度/加速度、跟踪检查 | `simulation/configs/weighted_trajectory_validation_v5.json` |
| 类别和连续接触代价 | `tools/contact_weights/contact_policy.json` |
| 花叶中心与叶片大小权重 | `tools/contact_weights/organ_weight_policy.json` |
| 轨迹数值 | `job_report.json` 中 `line_csv`、`timed_trajectory` |
| 轨迹线图、动画 | `job_report.json` 中 `media.line`、`media.animation` |

## 7. 自动化的适用范围

使用现有同款蓝色小车、**上层支撑板 135×110 mm** 的拍摄布置。自动裁剪和上层板选择仍是启发式方法，标定失败会停止；植物裁剪不依赖花叶颜色。更换小车、板、布置，或出现严重遮挡、几何缺失/背景粘连时，应先检查裁剪/标定/语义预览。成功导出器官提案不代表人工标注精度。

规划沿用五关节加权 RRT*、侧向接近候选和腕部姿态搜索；最终按相同 C3 平滑时序计算接近风险和持续接触代价。硬障碍物为花盆、泥土、小车；枝干、花、叶按空间权重计分。喷嘴目标为真实泥土上方 10 mm，允许范围 5–15 mm。保留后端夹持的 120 mm 可调喷嘴。

有限搜索寻找的是已评估候选中的最低代价，**不能证明全局最优**。检查覆盖整个机械臂和喷嘴，但使用离散采样和几何代理，不模拟植物形变或真实接触损伤力。当前为仿真：机械臂基座位置和喷嘴尺寸仍需实测标定才能执行实机动作，结果不会自动控制真实机械臂。

本次本地验证不能替代服务器首次验收：先运行 setup 检查，再提交一组照片，确认 `PHOTO_TO_PATH_SUCCESS` 及预览。更新包不承担重建全部旧 Linux 环境；它使用服务器已安装并运行过的重建/分割环境和模型。

本地验收记录：`outputs/pipeline_validation/integration_validation.json`。使用现有完整语义模型重新生成权重，再运行低预算的全关节规划、姿态优化、C3 动力学检查和动画导出；18 项运动检查通过。泥土目标细化的独立接口测试也通过。服务器上的 GPU 重建、分割和 Linux EGL 尚需首次实跑验证，不能据此宣称服务器照片全流程已通过。
