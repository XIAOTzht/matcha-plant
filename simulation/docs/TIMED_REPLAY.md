# 带时间的权重路径与 MuJoCo 回放

当前算法、权重和喷嘴说明见 [WEIGHTED_PATH_PLANNING.md](WEIGHTED_PATH_PLANNING.md)。最新路径入口为 `configs/latest_path.json`，密集复查入口为 `configs/latest_replay.json`。只有状态 `SAMPLED_REPLAY_CHECKS_PASSED` 表示该目录全部检查通过。

喷口目标为所选真实泥土三角面中心上方 10 mm，允许范围 5–15 mm。报告同时检查最终停留全过程的高度和到泥土三角面的最近距离。花盆、泥土、小车是硬障碍，保留 0.5 mm 仿真裕量；枝干、花、叶按现有局部权重计软代价。空气代价为 0。

距离倍率为 `g(d)=clip((5-d_mm)/5,0,1)^2`，接触时为 1，负间隙不额外增加倍率。同类别同器官每时刻取所有机器人体积造成的最大局部权重代价，不同器官相加。每个连续事件的积分乘 `1+0.5*min(T/1秒,3)`；小于等于 0.15 秒的间隔连接为同一事件。最终候选目标是连续事件代价之和加 `0.05*轨迹秒数`。

当前回放使用 `configs/weighted_trajectory_validation_v5.json` 的七次 Hermite（C3 连续）时间参数化。内部路点共享非零速度；同一关节发生换向时使用单调调和速度限制，避免过冲，其他关节继续运动，起点和终点仍为零速度。MuJoCo 原生位置 actuator 跟踪带 3.5 ms 位置超前的指令，并加入重力等 bias 前馈；参考和实际状态每 5 ms 检查一次完整原始硬障碍三角网格、机器人体积、喷嘴全部分段、工作空间、自碰撞和关节限制。报告保存速度、加速度、跟踪误差、终点误差和风险事件。

硬物体采用原生 box/cylinder/capsule/convex mesh 的自适应保守距离下界。软代价使用覆盖完整软三角面的毫米级小球场及局部最大权重。这些是几何代价，尚未测得真实接触力或植物损伤。植物显示网格关闭 MuJoCo 物理接触，不模拟叶片弯曲、水流和摩擦抓取；允许的软重叠可能表现为视觉穿过叶片。采样验证不等于连续扫掠体证明，开放重建表面也不能证明封闭实体内部安全。

```powershell
$ROOT = 'F:\matcha_mujoco_simulation'
$PY = "$ROOT\envs\mujoco\Scripts\python.exe"
& $PY "$ROOT\scripts\validate_watering_trajectory.py" --path "$ROOT\outputs\watering_attitude_path_v7" --config "$ROOT\configs\weighted_trajectory_validation_v5.json" --output-dir "$ROOT\outputs\watering_attitude_replay_v11"
& $PY "$ROOT\scripts\render_trajectory_replay.py" --fps 15
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$ROOT\scripts\open_trajectory_viewer.ps1"
```

窗口显示完整蓝色喷口路径线。空格暂停，R 重播，T 切换指令/实际姿态。新结果写入新目录，旧结果保留；换喷嘴或场景后重新筛选、规划及验证。回放保存的是仿真状态，不向实物发送指令。

Python 核心脚本使用相对资源路径；Linux 重新建立环境、安装 `requirements.txt` 后调用相同脚本。PowerShell 只是 Windows 启动入口，Windows 虚拟环境不应复制到 Linux。
