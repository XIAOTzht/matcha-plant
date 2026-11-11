# SO-101、植物和占位喷嘴场景

本文记录场景导入、目标筛选和基准路径。当前已完成带时间的 MuJoCo 跟踪与外部采样距离/风险检查，最新说明见 `TIMED_REPLAY.md`。植物形变与流体仿真尚未接入。

## 打开

Windows PowerShell 中执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'F:\matcha_mujoco_simulation\scripts\open_watering_viewer.ps1'
```

鼠标左键拖动旋转，右键拖动平移，滚轮缩放；空格暂停/继续。界面中可以查看关节和调节 Controls 中的六个控制量，暂时仅作为场景检查。

## 输入来源

模型来自本机 `F:\植物建模\outputs\with_car`：修正螺母后的 `complete_hardware_v2` 语义结果、`contact_weights_v2` 权重和 `organ_instances_v2` 器官候选。未冒充刚刚下载的服务器 `contact_test_*` 结果，也没有重新运行语义推理。输入和数值已经通过匹配检查；本地与服务器如需逐文件一致，需另行下载服务器结果并对比。

原始 PLY、NPZ 和报告保存在 `assets/plant/`。`metadata/contact_weights_report.json` 中的原开发路径用于追溯；新脚本实际使用配置中的相对路径。

## 单位与摆放

原模型为毫米，仿真为米。蓝色上板中心为植物坐标原点，以原 `board_frame_axes` 的第二列为向上方向，转换后 Z 向上，且旋转行列式为 +1，不做镜像或非均匀缩放。小车最低顶点距地面 2 mm。

`configs/scene_preview.json` 保存可调配置：

- 植物板中心的 XY 坐标：`[0.28, 0.20]` m；朝向 0 度。机械臂底座位于世界原点。
- 占位喷嘴：直径 8 mm，长度 60 mm，质量暂定 15 g。固定在 `gripperframe`，沿该坐标系的负 Z 方向延伸，末端为 `nozzle_tip`。
- 蓝色小点：原始泥土三角面中心上方 10 mm 的候选喷口位置。生成场景之后必须运行 IK 和碰撞筛选。
- 工作空间：Sites 第 5 组控制预览边框显示；规划与回放检查整个运动碰撞体是否位于边框内。

喷嘴的固定方式和质量是仿真占位参数，不代表实际夹持结构或实物标定。

## 显示和后续规划

植物显示网格按类别简化并按数值权重分色，只用于查看。它们的 `contype=0`、`conaffinity=0`，不会把整株凸包错误当成碰撞物体。后续规划必须用完整几何、对应的 hard_collision 标志和局部数值权重建立碰撞查询。

官方 SO-101 碰撞形状和占位喷嘴的碰撞形状保留；当前不保证与植物碰撞时会被挡住。

`full_resolution_world_geometry.npz` 保存完整坐标和三角面，顶点/面顺序与原始权重一致。`scene_report.json` 保存双向坐标变换和检查结果，`input_manifest.json` 保存 SHA256。输入权重中的硬障碍软权重储存为 0，必须同时读取 hard_collision，不能当空气。

花叶实例仍为待复核的几何候选，成功导入不代表生物器官实例已人工核准。

## 修改配置并生成新场景

修改 `configs/scene_preview.json`，再运行：

```powershell
$ROOT = 'F:\matcha_mujoco_simulation'
& "$ROOT\envs\mujoco\Scripts\python.exe" "$ROOT\scripts\build_watering_scene.py"
```

自动使用新场景目录，不覆盖旧场景，并更新 `configs/latest_scene.json`。生成后重新运行查看入口。

## 已有验证

三项坐标/目标测试覆盖高度轴、135 mm 长度保持、旋转与反变换、错误坐标拒绝、目标与原泥土面的对应关系。联合场景还检查喷嘴实际编译尺寸、完整模型指纹、顶点和面标签、硬碰撞边界规则、器官和权重编号一致、1000 步无数值警告。

## 目标筛选（当前阶段）

运行：

```powershell
$ROOT = 'F:\matcha_mujoco_simulation'
$PY = "$ROOT\envs\mujoco\Scripts\python.exe"
& $PY "$ROOT\scripts\screen_watering_goals.py" --seed-count 24
```

脚本读取场景自身的配置快照，使用 `full_resolution_world_geometry.npz` 和原始面级权重。`libigl.AABB` 对三角形内部做最近距离查询，不把植物替换成凸包，因此叶片间隙仍然存在。SO-101 的五个手臂关节用多初值数值 IK，喷嘴要求到达目标并保持喷射轴朝下；夹爪固定为 0.35 rad。每个不同 IK 分支均用规划器相同的体积检查，优先选择通过的分支。10 mm 场景启用保守包围球与自适应原生体积细分，方法见 `TIMED_REPLAY.md`。

结果目录包括 `goal_screening_report.json`、`screened_poses.npz` 和 `GOAL_SCREENING_SUCCESS`。`accepted_screening` 只表示 IK、整个运动碰撞体的工作空间、硬障碍保守检查和 MuJoCo 自碰撞/地面检查通过；`risk_proxy` 仍需在后续姿态优化和连续轨迹中比较。几何距离是接触风险代理量，不是实测力或植物损伤。

查看姿态：

```powershell
& $PY "$ROOT\scripts\render_goal_screening.py" "$ROOT\outputs\watering_screen_YYYYMMDD_HHMMSS"
```

目标筛选输出是离散姿态；连续关节路径与带时间回放由后续独立脚本生成，见下文和 `TIMED_REPLAY.md`。

## 第一条连续路径

路径规划不需要训练神经网络。当前实现使用双向 RRT-Connect 在五个手臂关节的关节空间搜索，再用随机 shortcut 缩短路径；每条边按 25 mrad 的最大关节插值步长重新做硬障碍、工作空间和自碰撞检查。规划器命令：

```powershell
$ROOT = 'F:\matcha_mujoco_simulation'
$PY = "$ROOT\envs\mujoco\Scripts\python.exe"
& $PY "$ROOT\scripts\plan_watering_path.py"
```

规划结果包含 `trajectory.csv`、`trajectory_q.npz` 和 `path_report.json`。规划阶段的输出是离散关节路点；`validate_watering_trajectory.py` 会另行生成限速轨迹、MuJoCo 实际跟踪和每时间步的距离/风险记录。喷嘴流量与真实机器人标定没有接入；`real_robot_ready` 保持 false。带时间回放入口、公式和限制见 `TIMED_REPLAY.md`。

当前默认读取 `configs/latest_screening.json`，10 mm 场景的 5 个目标都未通过完整植物表面避碰；原 20 mm、2 mm 和软风险 10 mm 目录保留。更改高度要同时修改候选高度与 `watering_target_override`，再依次重建场景、筛选、规划、验证。覆盖项仅限目标高度，原泥土/花盆/小车硬碰撞标签保持不变。
