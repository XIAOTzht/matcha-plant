# 通用植物部件拆分（几何候选）

输入已完成语义分割的三角网格，把叶、花这两类继续拆成局部实例编号。代码不写死植物品种、坐标、叶片数量、小车形状，也不依赖板子的方向。原网格、语义标签和已有权重结果均不覆盖。

**本工具输出待复核的几何候选，不保证每个编号就是生物学上的一片叶子或一朵花。** 重叠花叶、重建粘连和错误的语义标签，会影响实例拆分；花朵可能按花瓣分区。不能只凭类别标签可靠恢复任意植物的真实器官。

## 新模型怎么运行

Python 3.10+，依赖 `requirements.txt` 中四个包；Windows / Linux 使用同一套 Python 文件。

```bash
python split_plant_organs.py --mesh /path/new/semantic_mesh_mm.ply --labels /path/new/labels.npz --output-dir /path/new/organ_instances_v1 --export-parts
```

PLY 顶点已经有 `label` 时，可以省去 `--labels`。默认叶=2、花=3；其他编码用 `--leaf-id`、`--flower-id` 调整。支持 plyfile 可读取的 ASCII / 二进制三角 PLY，不接受 Gaussian 点云作为表面。

**必须知道尺寸。** 默认输入坐标单位为 mm；米制模型传 `--mm-per-unit 1000`。没有标定的模型应先标定，不能比较真实叶片大小。输出总网格和拆分部件的坐标保持输入单位和位置，面积与中心计算统一用 mm。

当前模型的 PowerShell 命令：

```powershell
$py = 'C:\Users\Y9000p\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py 'F:\植物建模\tools\plant_organ_instances\split_plant_organs.py' `
  --mesh 'F:\植物建模\outputs\with_car\complete_hardware_v2\semantic_mesh_mm.ply' `
  --labels 'F:\植物建模\outputs\with_car\complete_hardware_v2\labels.npz' `
  --output-dir 'F:\植物建模\outputs\with_car\organ_instances_v2' --export-parts
```

输出目录已有结果时会停止，选择新的输出目录即可，不自动清理。

## 算法

中心分数使用平滑后的闭合投影轮廓距离，连续插值后作 smoothstep 归一化。
器官内部的小洞和细碎轮廓不会被直接当成新边缘；中心到边缘的变化更连续。
仍需先修正语义碎片：本工具不会靠平滑颜色掩盖错误类别。

1. 从原三角面提取真实网格边，按语义类别做连通分量分析；不把空间上挨着但网格不连通的叶子强行连接。
2. 面积加权 PCA 得到局部投影，把三角表面栅格化为轮廓。投影分辨率相对组件尺寸自动缩放。
3. 轮廓内部的距离峰值给出候选中心；峰间抑制距离由局部轮廓宽度决定，没有固定叶片数。稀疏小碎片不进行多中心拆分。
4. 在原表面沿网格边计算多源最短距离，把粘连分量分配给这些中心；不是按世界坐标划固定方框。
5. 每个区域独立计算轮廓边界到内部的深度，归一化成 `center_score∈[0,1]`，供后续“内部权重更高”使用。该分数是几何近似，不是叶脉或花蕊检测。
6. 叶片大小用三角面表面积分摊到顶点后求和，单位 mm²，不使用顶点数量替代面积。该值是**重建表面积**；背面、缺面和假厚度会影响估计，不能直接等同真实单面叶面积。

`--marker-separation` 默认 1.0，调小通常增加分区，调大通常减少分区。`--min-area-mm2` 默认 10，小于阈值的组件保留为未解决碎片（不删除）。极小叶、针叶等需要降低阈值；这属于通用参数，不是改代码适配某株植物。

## 输出和复核

- `organ_instances.ply`：原模型完整保留，实例颜色、编号、中心分数和面积；非花叶为灰，未解决碎片为灰黑。原有顶点/面字段保留。面跨实例时编号 -2。
- `organ_instances.npz`：顶点实例、面积、中心分数、待复核标记、语义类别和网格指纹。
- `organ_report.json`：每个区域的面积、中心、原顶点编号、非平面程度、方法与限制；状态 `REVIEW_REQUIRED`。
- `seed_markers.json`：自动中心的原顶点编号，可用来调整并复跑。
- `parts/`：加 `--export-parts` 后导出每个候选区域，带 `source_vertex_index`，方便定位。仅导出完全落在该区域的面，跨实例边界的面仍保留在总网格中。

编号 0 表示非目标类别，-1 表示未解决碎片，正数表示候选实例。不要把候选数量当真实器官数量。

```bash
python render_organ_preview.py --instances-dir /path/new/organ_instances_v1
```

可选 `--coordinate-system coordinate_system.json` 仅用于把预览摆正，完全不参与拆分。预览点采样，完整几何请打开 PLY。

## 拆错时怎么修正

可用 `--seed-file seeds.json` 提供中心点；同一连通分量中只要提供了中心，就用提供的中心替代自动候选，不混入自动中心。文件是 JSON 列表：

```json
[
  {"class_id": 2, "xyz_mm": [10.0, 30.0, 5.0]},
  {"vertex_index": 12345}
]
```

坐标必须在输入模型相同的原始坐标系下、以 mm 表示。也可从 `seed_markers.json` 开始编辑顶点编号。它只是实例修正，不会把花改成叶。所有实例仍标记为未经过生物学验证。

后续权重采用这个独立输出作为输入。先复核实例，再计算每个实例的中心加权和叶面积加权；不能把整株植物的中心当每片叶子的中心。若实例仍不可靠，保留旧权重作为可用版本，不用候选结果覆盖它。
