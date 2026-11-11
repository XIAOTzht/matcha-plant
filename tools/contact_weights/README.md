# 接触权重工具

新版增加独立的通用部件拆分、中心/叶面积权重，见 [README_V2.md](README_V2.md)。原有 v1 文件保留。新模型使用显式输入路径和可配置类别编号，不依赖当前植物坐标。

本工具以最终修正后的 `complete_hardware_v2/semantic_mesh_mm.ply` 和同目录
`labels.npz` 为输入，建立 SO-101 浇水任务的第一版接触风险数据。它不会覆盖
原模型，也不会改动语义分割结果。

## 运行

```powershell
$py = "C:\Users\Y9000p\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py "F:\植物建模\tools\contact_weights\build_contact_weights.py" `
  --input-dir "F:\植物建模\outputs\with_car\complete_hardware_v2" `
  --output-dir "F:\植物建模\outputs\with_car\contact_weights_v1"
```

## 输出

- `contact_policy.json`：类别权重、硬碰撞、目标区域和连续接触参数。
- `contact_weights.npz`：逐顶点和逐三角面的基础/有效权重、硬碰撞、目标和不确定标记。
- `contact_weighted_mesh.ply`：原语义颜色加规划字段。
- `contact_risk_heatmap.ply`：风险颜色预览，绿色→黄色→浅红表示软权重增加；亮深红统一表示硬碰撞。
- `contact_weights_report.json`：数量与一致性报告。

## 第一版规则

- 空气：0。空气不是网格类别，而是工作空间内没有被物体占据的空间。
- 叶子：1；花：5；枝干/茎：20。
- `unknown`：20；当前只有 9 个顶点。
- 花盆、泥土、小车（包括螺母）：硬碰撞，碰到后路径无效，不以很大的有限数字代替。
- 泥土是浇水目标类别，但喷头仍不能与泥土相交，只能停在泥土上方。
- 不确定的软类别有效权重乘 1.2；硬碰撞类别始终是硬约束。

热力图颜色按权重分段显示：绿色→黄色→浅红色，普通枝干(20)为浅红，不确定枝干(24)只稍微加深，和不可触碰的深红明显区分。花盆、泥土、小车和螺母统一使用同一种亮深红色，表示不可触碰硬约束。泥土仍保留浇水目标字段；目标身份不改变喷头不可碰撞的规则。模型和独立颜色条共用 `weight_palette.py`，颜色条单独输出，不叠加到模型预览中。不可触碰对应独立色块，不冒充有限权重。

## 接触强度与连续触碰

后续 MuJoCo/规划器沿时间采样喷头和机械臂，向 `contact_cost.py` 提供：

```text
time_s, class_id, clearance_mm, penetration_mm, contact_fraction, uncertain
```

距离风险：

```text
clearance >= 5 mm       risk = 0
0 < clearance < 5 mm    risk = ((5 - clearance) / 5)^2
clearance <= 0            risk = 1（不再额外增加穿透倍率）
```

瞬时风险：

```text
class_weight × uncertainty_multiplier × distance_risk × contact_fraction
```

路径暴露量使用梯形积分：

```text
exposure_integral = ∫ instantaneous_risk dt
```

同一类别连续接触段还会增加持续性惩罚：

```text
episode_cost = exposure_integral × (1 + 0.5 × min(duration_s / 1.0, 3.0))
```

相同强度连续接触 3 秒会比 0.1 秒代价高；间隔超过 0.15 秒或类别改变，会切成新的接触段。参数都在 `contact_policy.json` 中。

这些是路径规划代价，不是植物真实损伤、接触力或牛顿单位。真实接触力以后可从 MuJoCo 读取，真实损伤还需要实验标定。
