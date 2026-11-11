# SO-101 植物权重路径规划

所有路径都相对于本仿真目录。最新结果入口为 `configs/latest_path.json` 和 `configs/latest_replay.json`，旧结果保留。

## 算法与选择

采用五维关节空间的 RRT*，配合 RRT-Connect 可行初解、保留权重代价的 shortcut、多个泥土目标的比较以及包含连续接触惩罚的最终筛选。五个机械臂关节都可在中途自由调整；夹爪固定在夹持角度，最终喷口才约束向下。RRT* 适合机械臂的多关节空间、非凸障碍和自定义代价。RRT-Connect 只负责必要时寻找可行初解，不负责最低代价的结论。

这是本地可运行的有限预算实现。没有官方 Windows OMPL wheel，因此没有假称使用 OMPL。邻居数、采样密度与运行时间均有限，结果表示本次找到的最低评估代价，不证明全局最优。

新搜索将土面落点和末端姿态一起优化：先用 28 个真实土面三角面中心、每点 48 个初值搜索，位置约束满足时允许喷嘴在竖直向下 20° 圆锥内倾斜；再沿邻近真实土面细化位置，并优化自由姿态。每个姿态都按整臂和喷嘴的现有局部权重评分。腕部可以在中途自由旋转，夹爪开合保持夹持角；SO-101 是 5 个定位/姿态关节加 1 个夹爪开合关节，开合不能作为额外的自由旋转轴。

`scripts/build_weighted_corridors.py` 生成多个侧向进入候选作为 RRT* 初始路线；笛卡尔中间点仅用于提出关节姿态，连边仍进行原始三角网格硬约束检查。用相同权重和连续接触公式比较不同路线，不通过降低权重使结果变好。倾斜喷嘴还要求出水中心线的第一处原始三角面命中泥土，若先命中盆沿或花叶则剔除；该检查只描述中心射线，不代表完整水束或流体仿真。

## 代价与约束

- 花盆、泥土、小车及四个螺母：硬约束，不能接触；硬障碍面距离留 0.5 mm 仿真裕量。
- 枝干、花、叶、不确定区域：软代价，可接触，采用现有权重文件中的局部数值。花叶的中心、大小和不确定性已含在该数值内，不会再乘一遍。
- 距离倍率：`g(d)=clip((5-d_mm)/5, 0, 1)^2`。5 mm 外为 0，接触时为 1；负距离仍为 1，没有额外穿透倍率。
- 同一类别、同一器官实例：每时刻取机器人各部件造成的最大局部权重接近代价，避免夹爪多个碰撞代理重复计费；不同器官代价相加。
- 连续接触：同一器官两次接近之间的间隔不超过 0.15 s，合为同一事件。事件代价为 `积分(w*g(d),dt) * (1+0.5*min(T/1秒,3))`，最长倍率为 2.5。
- 最终候选目标：`J=各事件代价之和+0.05*轨迹秒数`，时间项用来避免无意义绕路；初始等待和终点 1 秒停留也计入。
- 树搜索用关节距离上的权重积分作快速代理；实际带时间和连续接触的代价决定最终候选选择。两者都保存于报告，不混称同一种量。
- 每条边的硬约束采样最大关节增量为 0.006 rad，最终动态回放每 5 ms 查询一次完整原始硬障碍网格。这是密集采样验证，不能称为数学上的连续扫掠体证明。

权重表达规划偏好，尚非实测植物损伤、接触力或变形模型。叶片和花仍可能出现视觉重叠，这是软接触被允许的结果；硬物体和机械臂自身的相交必须拒绝。

## 距离查询

硬物体使用完整 4294728 面原始模型中的 hard 子集与自适应机器人原生体积查询，不用显示简化网格判碰撞。

软代价覆盖每个原始软三角面，按 1 mm 单元及器官身份聚合，保存覆盖半径及单元内最大顶点权重。使用机器人 box/cylinder/capsule/convex mesh 到这些小覆盖球的距离。这会保守提前产生软代价，解决原先用整根连杆大包围球导致的虚假接触。`soft_contact` 报告保留半径统计。

## 2026-11-07 协同动作升级

当前集成项目位于 `F:\plant\matcha_plant`，仿真代码位于其 `simulation` 子目录。旧运行示例和历史成绩保留用于追溯，最新结果以 `configs/latest_path.json` 和 `configs/latest_replay.json` 为准。

加权 RRT* 和姿态优化后增加整条轨迹协同搜索，支持提前转腕、错开关节时序及保持喷口位置的肘腕姿态变化。默认在最低候选接触代价的 15% 范围内、且不增加原基准接触代价的前提下，偏好更均衡的关节参与，另有限时和行程约束。这是明确的动作偏好，并非全局最低惩罚证明；详见 `COORDINATED_MOTION.md`。

## 喷嘴夹持

`configs/weighted_planning_scene.json` 设置夹爪 -0.08 rad，初始 wrist_roll 为 90°。新喷嘴整体 120 mm；两指横向夹住后端 18 mm、直径 9.3 mm 的握持段，后接直径 8 mm 的延长管及最大直径 15 mm 的阶梯式喷头。视觉网格带浅滚花、环槽和凹陷出水孔，位于分段圆柱碰撞包络内。场景为 `scenes/watering_rear_grip_nozzle_v3`，配置副本为 `configs/weighted_stepped_nozzle_scene.json`；旧 60 mm 横柄配置保存在 `configs/weighted_planning_scene_legacy.json`。

尺寸和安装变换可调。喷嘴与夹爪采用刚性附着，夹爪不会沿路张开。7 个喷嘴碰撞段均参与硬物体、软器官、工作空间和机器人自身检查，补上 MuJoCo 对相邻或焊接 body 默认不检测的情况。仅允许握持段与命名的 fixed_jaw_sph_tip1、moving_jaw_sph_tip1 指尖球代理在固定夹持角附近有不超过 0.6 mm 的接触重叠；其他部位与任一连杆的相交仍拒绝。指尖球是保守代理，其外缘比视觉指面更突出。规划使用原生碰撞代理，不能将这些检查解释为实物加工公差或摩擦夹持稳定性验证。

这是夹持姿态占位模型，不包含软管、摩擦夹持稳定性、夹具加工公差。末端停在某个可达泥土目标上方 10 mm，允许范围 5–15 mm。

## 运行

```powershell
$ROOT = 'F:\matcha_mujoco_simulation'
$PY = "$ROOT\envs\mujoco\Scripts\python.exe"
& $PY "$ROOT\scripts\screen_watering_goals.py" --seed-count 36
& $PY "$ROOT\scripts\plan_watering_path.py" --seconds 120 --iterations 400 --optimize-goals 3
& $PY "$ROOT\scripts\validate_watering_trajectory.py" --config "$ROOT\configs\weighted_trajectory_validation.json"
& $PY "$ROOT\scripts\render_weighted_path.py"
& $PY "$ROOT\scripts\render_trajectory_replay.py" --fps 15
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$ROOT\scripts\open_trajectory_viewer.ps1"
```

更大搜索预算：增加 `--seconds`、`--iterations`、`--optimize-goals`，可用 `--seed` 改变采样。`--goal-id` 固定泥土位置。`--resume-candidates` 指向同场景之前的结果目录，保留旧候选参与比较。输出目录必须是新的，以免覆盖已检查结果。

## 关键文件

- `scripts/plan_watering_path.py`：关节空间规划、硬约束、候选代价与轨迹导出。
- `scripts/explore_watering_poses.py`：真实土面目标及末端姿态的联合搜索。
- `scripts/build_weighted_corridors.py`：不同腕姿和侧向通道候选。
- `scripts/spray_target.py`：出水中心射线的原始三角面首命中检查。
- `scripts/soft_contact_field.py`：软器官空间权重及机器人体积查询。
- `scripts/trajectory_tools.py`：距离倍率、连续接触、连续七次 Hermite 时间参数化和换向速度限制。
- `scripts/validate_watering_trajectory.py`：MuJoCo actuator 实际运动和密集检查。
- `configs/weighted_planning_scene.json`：场景、喷嘴和夹持参数。
- `configs/weighted_trajectory_validation.json`：速度、加速度、夹持角及回放参数。
- 输出 `nozzle_path_line.csv/npz`：以米为单位的喷口空间轨迹线，带时间。
- 输出 `trajectory_q.npz`：完整五关节关键点和夹爪角，喷口线不单独作为机器臂控制输入。
- 输出 `path_report.json`：算法、候选比较、目标、规则、有限搜索限制。
- 回放 `timed_trajectory.npz/csv`：指令和实际仿真关节、空间轨迹及软代价。
- `nozzle_path_line.png`、`trajectory_replay.gif`：静态线和实际运动动画。交互窗口也显示整条蓝色线，空格暂停、R 重播、T 切换指令/实际姿态。

当前时间参数化为 C3 连续七次 Hermite 曲线：所有关节同时运动，内部路点不再停稳；换向关节平滑减速，其他关节保持连续运动。它保持验证过的关节空间线段；没有用未经碰撞复查的笛卡尔曲线替代机械臂轨迹。


## 当前结果与验证入口

最新路径和复查目录以 `configs/latest_path.json`、`configs/latest_replay.json` 为准。旧 60 mm 喷嘴结果 `outputs/weighted_grasp_path_v3`、`outputs/weighted_grasp_replay_v4` 保留作为历史记录。更换喷嘴后必须重新运行目标筛选、路径规划和动态检查，不能将旧轨迹当作新工具的验证结果。终点喷口目标为指定土面上方 10 mm。

120 mm 后端夹持喷嘴的当前结果是 `outputs/rear_grip_nozzle_path_v3` 和 `outputs/rear_grip_nozzle_replay_v3`：5 个土面目标通过静态筛选，比较 12 条候选路径，选中目标 2。相同目标的初始路径最终目标函数为 621.58，优化后为 410.07，降低约 34%。这是当前有限预算候选中的最低评分，不是全局最优证明。

动态复查为 5.925 秒、1186 个状态，全部列出的检查通过；最小硬障碍保守间距 1.297 mm，终点竖直土面高度 10.00 mm、最近土面距离 7.887 mm，最大关节跟踪误差约 0.072°。实际仿真连续接触代理代价为 406.31；粗时间网格预筛选与 5 ms 动态检查有小幅数值差异。`scenes/watering_rear_grip_nozzle_v3/nozzle_grasp_geometry.json` 另外记录原生碰撞代理和原始指面三角网格的抽样距离检查。

`candidate_scores.json` 记录所有候选；`candidate_paths.npz` 保留对应路径。有限搜索本身无法排除其他更低代价路线。时间评分的 40 ms 预筛选及 5 ms 动态复查数值可能略有差别，以最新 `validation_report.json` 为仿真实际数值。

`configs/latest_replay.json` 指向当前复查结果，只有 `SAMPLED_REPLAY_CHECKS_PASSED` 才表示全部已列检查通过。失败输出保留诊断，不作为成功结果。夹持接触规则有专门测试，保证喷管或其他连杆碰撞不会被一并放行。
