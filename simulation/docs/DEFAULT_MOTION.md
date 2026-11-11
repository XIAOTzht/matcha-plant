# 当前默认：五关节协同版本

2026-11-08 按用户选择恢复 `coordinated_20261107/coordinated_v2`。

- 默认路径：`simulation/outputs/coordinated_20261107/coordinated_v2`
- 默认回放：`simulation/outputs/coordinated_20261107/replay_v2`
- 动画：`simulation/outputs/coordinated_20261107/replay_v2/trajectory_replay.gif`
- 配置：`simulation/configs/coordinated_motion.json`
- 默认索引：`simulation/configs/latest_path.json`、`latest_replay.json`

本地恢复的是原先保存并验证的路线与回放：时长 17.10500000000088 s，实际接触惩罚 21.628808532733345，肘部范围 33.22368056401868°，腕旋转范围 43.21390126621767°，3422 个状态、18 项检查全部通过，土面上方约 10 mm。没有重新搜索或改写原结果。

恢复时核对了原路径、场景和验证报告的哈希；当前旧版时间参数化产生的时间、关节角度、速度、加速度与原保存的五关节参考数组逐值一致。该轨迹不带自然模式固定时间文件，继续采用原控制和时间逻辑。

完整流水线 `configs/plant_pipeline.json` 保留 `simulation_coordination_config`，移除 `simulation_natural_config` 和 `simulation_demonstration_config`，以后运行默认止于协同搜索再执行动态验证和渲染。未来其他模型会重新规划，不能保证得到与该模型相同的路线。旧自然版和展示版的代码、配置和全部结果保留，需明确选择才启用。

本次仅修改本地默认设置，未上传服务器。代码包不包含本地历史模型、回放或 latest 索引；未来上传新构建的代码包会同步流水线默认模式，但不会自动替换服务器已有结果。部署和重跑使用新 RunName。

旧实验目录内 README 描述各自当时状态；当前默认以本文和两个 latest 索引为准。结果是有限搜索与采样几何代理，不包含真实植物接触力或全局最优保证。
