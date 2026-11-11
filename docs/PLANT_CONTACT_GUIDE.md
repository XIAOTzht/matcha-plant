# 花叶实例与接触权重

配色：`weight-colors-0.4.2-smooth-green-yellow`。
参数：`contact-policy-0.3.0`、`organ-contact-policy-0.3.0`。
部署、服务器整理和 GitHub 操作见 [SERVER_MAINTENANCE.md](SERVER_MAINTENANCE.md)。

## 文件位置

- 语义输入：`outputs/with_car_20261021/plant_car_semantics_v2/`，包括螺母归车修正。
- 花叶拆分：`tools/plant_organ_instances/`，通用几何候选，并非人工验证的器官数量。
- 权重代码及默认参数：`tools/contact_weights/`。
- 通用流水线：`scripts/plant_contact/run_pipeline.py`（Windows/Linux 相同代码）。
- Linux 入口：`scripts/plant_contact/run_server.sh`，使用现有 `envs/plant_semantics`。
- 当前场景便捷入口：`scripts/plant_contact/run_latest_server.sh`。
- 默认输出：`outputs/with_car_20261021/contact_latest_<时间>/organs,weights,logs`。
- 最新结果指针：`outputs/with_car_20261021/latest_contact.json`。
- 本地已确认配色的历史导出：`outputs/with_car_20261021/contact_weights_v5_smooth/`。
  此目录不通过小代码包上传；服务器入口会重新生成同策略的新结果。

完整流程：照片 → 重建 → 植物与小车裁剪 → 上层蓝板 135×110 mm 尺寸标定 →
语义分割和螺母归车 → 花叶候选拆分 → 权重。
此次新增入口从语义结果开始，不重新重建、标定或推理语义。

## 当前权重公式

c 为每个候选器官从轮廓边缘到内部的归一化中心分数，0≤c≤1。
A 为重建叶片表面积，单位 mm²。中心分数来自平滑轮廓距离及 smoothstep。
完整叶/花的 U = 1 + 0.2 × 器官内不确定表面积占比，整个器官使用同一倍率；
原始逐点标记保留。枝干/unknown/未解决碎片仍使用 U=1.2（不确定），否则 1。

```text
叶片中心系数 = 1 + 2c
叶片面积系数 = 1 + 2A/(A+2500)
叶权重 = 1 × (1+2c) × (1+2A/(A+2500)) × U
花权重 = 5 × (1+1.5c) × U
枝干与 unknown = 20 × U
花盆、泥土、小车（包括螺母）= 硬约束，不可接触
空气 = 0
```

叶片中心随面积增加通常可到约 9；保守处理未解决碎片时最大 10.8（含不确定）。
花边缘 5、中心 12.5，不确定最高 15；枝干 20/24。
硬约束的软权重字段写 0，但 hard_collision=true，必须优先判断硬约束。
泥土是浇水目标，但喷嘴本体仍不可接触，目标高度 5–15 mm，默认 10 mm。
面积只代表规划偏好，不是测出的真实硬度；中心是轮廓内部深度，不保证是花蕊或叶脉。
叶/花面直接使用同类别角点权重上界，不再叠加零散面标记的倍率，以免产生权重碎片。

## 距离与连续接触

```text
距离因子 = clip((5-d)/5, 0, 1)^2
d≥5 mm → 0；d≤0 mm → 1；不加穿透倍率
瞬时风险 = 局部有效权重 × 距离因子 × 接触/邻近占比
段暴露量 = ∫瞬时风险 dt（梯形积分）
段代价 = 段暴露量 × [1 + 0.5×min(持续秒数/1秒, 3)]
```

距离越近，代价增长越快。连续段要求同类别、同实例、采样间隔≤0.15 秒。
局部权重已包括中心、面积和不确定系数，评分时不能重复乘。
占比由后续几何查询提供，不是实际力。当前代码导出风险数据，不运行机械臂或仿真。

## 换一株植物

给通用入口传新植物语义目录和空的输出目录。要求同一顶点顺序的
`semantic_mesh_mm.ply` 和 `labels.npz`。默认 mm；非 mm 模型显式传 `--mm-per-unit`。
类别编号有变化时，复制并修改两份策略，用 `--policy` 和 `--geometry-policy` 指定。
不固定器官数量、坐标或小车模板；各模型的实例分区仍需人工检查。

```bash
bash scripts/plant_contact/run_server.sh /path/to/semantic /path/to/new_result --check-only
bash scripts/plant_contact/run_server.sh /path/to/semantic /path/to/new_result
```

已有非空输出会拒绝覆盖。可以用 `--no-parts` 跳过单部件网格导出，或
`--no-previews` 跳过图片预览。输出成功标记为 `PLANT_CONTACT_EXPORT_SUCCESS`。
