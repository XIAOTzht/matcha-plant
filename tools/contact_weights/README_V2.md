# 通用植物局部接触权重

当前代码参数：`contact-policy-0.3.0`、`organ-contact-policy-0.3.0`；
配色：`weight-colors-0.4.2-smooth-green-yellow`。
部署及服务器整理见 [SERVER_MAINTENANCE.md](../../docs/SERVER_MAINTENANCE.md)，
完整公式和调用见 [PLANT_CONTACT_GUIDE.md](../../docs/PLANT_CONTACT_GUIDE.md)。

## 输入和部件拆分

同一网格的 `semantic_mesh_mm.ply` 与 `labels.npz` 为输入。
先用 `tools/plant_organ_instances/split_plant_organs.py` 产生叶/花几何候选，
再用 `build_organ_contact_weights.py` 赋权。实例指纹校验顶点、连接、标签和尺度，
防止拿另一份模型的实例赋权。Windows/Linux 共用相同 Python 代码，无固定植物坐标、
器官数量或小车模板。自动候选需要人工复核，不是精确植物学实例保证。

## 当前公式

c 是每个候选内部到投影轮廓边界的归一化深度，范围 0–1。
它不是整株植物的中心，也不保证识别到花蕊或叶脉。
A 为重建表面积，单位 mm²。完整叶/花使用 U=1+0.2×该器官不确定表面积占比，
全器官统一；枝干及未解决碎片仍逐点使用 1/1.2。中心分数采用平滑距离场。

```text
叶 = 1 × (1+2c) × (1+2A/(A+2500)) × U
花 = 5 × (1+1.5c) × U
枝干及 unknown = 20 × U
花盆、泥土、小车及螺母 = hard_collision=true
空气 = 0
```

理论上叶普通中心随面积增大趋向 9，含不确定最高 10.8。
花边缘 5，中心 12.5，不确定中心 15。枝干 20/24。
未解决叶/花碎片采用该类最大中心系数；未解决叶还用最大面积系数。
面积只是规划偏好，未测量真实硬度、刚度或损伤。叶片的双面重建、缺面和粘连会影响面积。

硬约束的数值软权重字段设为 0，规划时必须先读取硬约束字段，不能把它当空气。
泥土保留浇水目标标记，喷嘴仍不可接触，距离土面 5–15 mm，默认 10 mm。
三角面只要原面或任一角点为硬约束，整面为硬约束。
叶/花软面用同类别角点权重的保守上界，不另叠加碎片面标记；其他软面仍补足面自身不确定因子。

## 文件

- `contact_weighted_mesh.ply`：保留输入颜色，附加数值权重、实例、中心/面积、硬约束、目标字段。
- `contact_risk_heatmap.ply`：相同几何和权重，用逐顶点连续色可视化。
- `contact_weights.npz`：顶点和三角面数组，规划读取它或 PLY 数值字段。
- `contact_weights_report.json`：来源、指纹、单位、范围、限制和复核状态。
- `contact_policy.json`、`organ_weight_policy.json`：本次参数快照。
- `contact_weight_colorbar.png`：独立颜色条；与网格共用 `weight_palette.py`。
- `CONTACT_WEIGHTS_EXPORTED_REVIEW_REQUIRED`：导出完成，器官分区仍需复核。

颜色由全局权重连续映射为浅绿、绿、黄绿、黄、橙、浅红。
枝干 20 为浅红，不确定 24 稍深；花盆、泥土和小车统一亮深红。
颜色不改变数值和碰撞权限。原位置、连接、语义和单位保持。
新导出直接使用最新配色；旧结果可用 `recolor_heatmap.py` 仅重新着色。

## 距离和连续触碰

读取局部 `contact_weight` 或 `vertex_effective_weight`，
作为 `ContactSample.local_effective_weight`，它已包含 U，不能重复乘。

```text
R(d) = clip((5-d)/5, 0, 1)^2
d≥5 mm: 0
d=4,3,2,1 mm: 0.04,0.16,0.36,0.64
d≤0 mm: 1；不加穿透倍率
瞬时风险 = 局部有效权重 × 接触/邻近占比 × R(d)
连续段因子 = 1 + 0.5×min(持续秒数/1秒,3)
```

`score_trace` 用梯形积分求全程暴露量，再对同类别、同实例且采样间隔≤0.15 s
的连续段增加持续惩罚。完整段费用为该段暴露量乘连续段因子。
无接触或距离≥5 mm 会中断连续段；实例编号应由几何查询传入，未传时只能按类别分段。
多个机械臂部位同时接触要分别记录、汇总，不能视为同一点。
接触/邻近占比是后续几何查询提供的代价代理，不是接触力。

本工具没有运行 SO-101、仿真或路径规划。输入实例、参数和坐标配准仍需验证。
