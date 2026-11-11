# 历史试验记录（归档）

本文件保留早期方法、旧命令与路径，不能视作当前使用说明。当前自动入口及 v2 结果见项目根目录 [README](../../README.md)。原诊断脚本已移至 `experiments/archive/`，测试已移至 `tests/`；下文保留旧引用用于追溯。

# 植物网格五类零样本分割（实验版）

输入为带顶点 RGB 的三角网格 PLY。输出类别：0 未知，1 枝干/茎/叶柄/花梗，2 叶片，3 花/花苞，4 花盆/托盘，5 土壤/基质。

**最新交付版本为 `outputs/plant_semantics_complete_v1/`。用户已明确输入全属于五类，因此最新版本将所有顶点和面归入 1–5，不再用灰色/类别 0 表示不确定性。** 下文零样本基线与局部试验目录保留用于对比。完整标签的生成和限制见文末。

## 方法

从原网格渲染 14 个正交视角；Open3D 射线追踪保存每个像素的可见三角面和重心坐标。CLIPSeg 产生文本提示的稠密语义分数，GroundingDINO 产生局部部件框，MobileSAM 提取候选区域。用 CLIPSeg 判断候选区域的语义支持，再融合回原顶点。默认对每个原顶点投影并单独射线检验深度，避免小三角面落在像素间隙而漏采样；`--no-vertex-rays` 可比较旧版像素/重心采样。每个视角对一个顶点最多计一次观测；要求多视角支持、类别差距和一致性。证据不足保留未知。

没有手动框/点提示、图像专用多边形、按粉色/绿色阈值分组、训练或微调。英文提示和融合阈值在本株的试验视角上作过工程选择，因此本株不是独立测试集。这仍然是零样本模型推理，不是经过标注验证的通用高精度算法。

## 本机执行

```powershell
$py='C:\Users\Y9000p\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py 'F:\植物建模\tools\plant_semantics\plantseg.py' run `
  --input 'F:\植物建模\outputs\Plant_cropped.ply' `
  --output 'F:\植物建模\outputs\plant_semantics_zero_shot'
```

对新植物使用新的输出目录：

```powershell
& .\run.ps1 -InputMesh 'D:\plants\new.ply' -OutputDirectory 'D:\results\new' -UpAxis y
```

`--up x/y/z` 必须与输入模型的竖直方向一致；本模型为 y。输入可有任意平移、正尺度和绕竖直轴旋转。尚未实现可靠的自动竖直方向估计。点云 PLY、无颜色模型、多个花盆的整场景不在此版本已验证范围。

三个阶段可以单独运行 `prepare`、`predict`、`fuse`。`predict --limit 2` 用于试验前两个视角；`--repredict` 显式替换二维预测。修改模型、提示词后，缓存签名不一致会拒绝混用。`fuse` 可对保存的分数调整拒识阈值，无需重新运行网络。默认不传播未知标签；可选 `--fill-steps` 只沿网格边缘对从未被像素观测的顶点作一致邻域填补，仍不代表真实语义证据。

## 文件

- `semantic_mesh.ply`：语义着色，保留原顶点坐标、顶点顺序和三角面顺序；另有 `label`、`evidence` 顶点属性。
- `labeled_original_rgb.ply`：原 RGB 加上述属性，alpha 未复制到输出。输入文件不变。
- `labels.npz`：逐顶点标签、逐面标签、分数、类别差距、视角数、一致性和是否传播。三角面跨越类别边界时面标签为 0。
- `review_contact_sheet.jpg`：每行依次为渲染原图、二维预测叠加、三维融合结果回渲染。
- `report.json`：类别数量/面积占比、未知/未观察比例、阈值、真实验证状态。
- `manifest.json`、`inference_config.json`：输入 SHA256、视角、模型 SHA256、提示与推理配置。
- `views/`：渲染图、可见面/重心坐标、二维分数、检测记录、标签和叠加图。

所有 evidence、CLIPSeg sigmoid、检测分数、SAM 质量分数、多视角一致性都不等于语义准确率。未知比例和覆盖率应与各类精度同时报告，不能通过把困难区域全部拒识来声称高准确率。

## 部署

运行代码为独立命令行接口，后续可封装 HTTP 服务/批处理/容器。为新环境创建独立 Python 3.10–3.12 环境，先从 PyTorch 官方安装相互匹配的 torch/torchvision（本机试验为 CPU），再 `pip install -r requirements.txt`。本地 `vendor/` 只为避开现有不完整 Python 环境准备，不修改现有重建环境；正式打包建议用固定版本环境并移除该目录的优先导入。

`fetch_models.py` 下载 GroundingDINO tiny 和 MobileSAM 官方源码；`fetch_models.py --clipseg` 下载 CLIPSeg refined。Hugging Face 在本网络超时，下载使用公开模型的 hf-mirror 文件镜像，不绕过受限访问。MobileSAM 权重当前复用 `F:\植物建模\tmp\segmentation_models\mobile_sam_parallel.pt`，迁移时用 `--sam` 指定。模型 SHA256 记录在推理配置中。网络下载可续用分片。

## 已验证及限制

`test_geometry.py` 九项测试检查像素及逐顶点遮挡、弱证据/冲突拒识、平移/尺度不变性、导出坐标及面索引、细节相机投影和盆口几何约束、完整五类面标签/元数据、图传播不跨越未连接表面、不可见区域沿已连接表面取得标签。它不检验植物语义准确率。当前仅本株三维模型执行端到端试验，不能据此宣称跨物种泛化已验证。

2026-10-20 实际结果：`outputs/plant_semantics_zero_shot/report.json` 中未知顶点 29.85%，未被任一视角观察到的顶点 9.28%；未知面积约 40.5%，所以只报顶点覆盖率会高估效果。仅 6,941 顶点被接受为茎、166 顶点为基质，明显漏分。结论：**端到端流程已运行，但这版五类准确性不达标，不应标为完成的高精度分割产品。** 原输入 1,567,628 顶点 / 3,136,830 面保持不变。

另有 `refine_appearance.py`：对自动伪标签种子拟合当前模型的 Lab 颜色高斯混合分布，不微调基础模型、不用人工标注。这是**逐场景拟合**，并非完全无需适配。运行输出在独立的 `outputs/plant_semantics_appearance_trial/`。可改善一些细茎与叶片，但已发现绿色花苞误分成茎、基质种子不足。只作诊断对照，不替换主流程结果。

```powershell
& $py 'F:\植物建模\tools\plant_semantics\refine_appearance.py' `
  --source 'F:\植物建模\outputs\plant_semantics_zero_shot' `
  --output 'F:\植物建模\outputs\plant_semantics_appearance_trial'
```

细花梗、茎/花接触边界、重建粘连、花瓣背面、叶片基部和盆内基质是已发现的难点。表面已被重建丢失时，本算法不能补回真实器官。不同视角可能重复相同模型错误，一致性不等于正确。高精度承诺仍需独立的跨植株/跨物种测试，以及逐类 IoU、茎召回、边界与覆盖率评估。

## 模型来源

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)，[预训练模型](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
- [CLIPSeg](https://github.com/timojl/clipseg)，[预训练模型](https://huggingface.co/CIDAS/clipseg-rd64-refined)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)，源码许可证保存在 `third_party/MobileSAM/LICENSE`
- [Open3D](https://github.com/isl-org/Open3D)

本实验未使用需要申请访问的 SAM3 权重。今后可用 SAM3/其他二维模型替换语义前端并复用几何融合；必须重新验证分数和阈值。分发模型与代码时应随附对应仓库及权重许可。

## 自动细节试验（独立结果目录）

`refine_detail.py` 自动定位粗分割的花盆和高线性度区域，重新射线渲染 28 张局部放大图。几何特征使用均匀空间采样的双尺度 PCA，避免直接把网格顶点密度当作器官形状。使用同样三个冻结模型推理；重叠裁剪与原相机共享一票，避免重复视角抬高一致性。

盆口约束只剔除外壁/盆外的基质候选，不能单独产生基质标签。约束假设一个紧凑、接近圆形的花盆，不能直接推广到长方盆、多盆和裸根植物。线性度只提供小幅茎证据加成，花瓣/叶片边缘同样可能具有高线性度，因此不作硬标签。所有尺寸相对模型大小归一化，没有手工指定这株的裁剪框或颜色规则。

```powershell
& 'F:\植物建模\tools\plant_semantics\run_detail.ps1' `
  -SourceDirectory 'F:\植物建模\outputs\plant_semantics_zero_shot' `
  -OutputDirectory 'F:\植物建模\outputs\plant_semantics_detail_trial'
```

输出 `review_contact_sheet.jpg` 为完整视角回看，`detail_review_contact_sheet.jpg` 为放大区域回看，`comparison_preview.jpg` 为前后对照。`geometry.npz` 保存每个原顶点的几何特征。缓存记录模型、提示词、相机和基线标签的 SHA256，修改实验配置应使用新输出目录。

本株结果：茎从 6,941 增至 15,458 顶点，基质从 166 增至 1,048 顶点；未知顶点从 29.85% 变为 30.18%，未知面积 41.10%。这些是输出分布，**不是准确率或召回率**。盆口单张图上改善较明显，三维多视角确认仍不足；叶片背面、细花梗与花瓣边界仍有错漏。结果保持实验状态，不自动取代基线。

SAM3 是值得优先比较的前端候选，尚无本任务实测结果证明它最佳。官方权重下载步骤见 [下载说明](models/sam3/下载说明.md)，本机预留路径 `models/sam3/sam3.pt`。获得权重后仍需单独准备推理环境，并沿用相同网格、相机和验收口径比较。

## 完整五类版本（当前交付）

`complete_five.py` 改用五类闭集判定：所有原顶点、所有原三角面都有 1–5 标签。不确定性独立保存，不生成灰色类别。

处理要点：重新提取全局/局部检测框的 MobileSAM 掩膜，允许被稠密模型误判为花盆的深色叶片和基质提供独立证据；只把各相机的一组裁剪计作一次观察；单独保留多视角花苞实例，减少绿色花苞被颜色模型归为茎；对空间均匀采样的 PCA 形状加以约束，抑制大范围 branch 掩膜吞并叶片；拟合自动伪标签的 Lab 颜色分布；沿真实网格边传播并细化标签。不可见表面优先沿其连接表面取得标签。与有观测区域不连接的小碎片使用外观/形状回退，不使用跨表面的欧氏最近邻。

这是场景内自动伪标签拟合，冻结基础模型，无人工标注或微调。盆口仍使用紧凑近圆单盆假设。全部赋类并不意味着准确率为 100%。深色叶片/茎接触处、花梗/花瓣边界、基质/根部仍需检查。

```powershell
& 'F:\植物建模\tools\plant_semantics\run_complete.ps1' `
  -BaseDirectory 'F:\植物建模\outputs\plant_semantics_zero_shot' `
  -DetailDirectory 'F:\植物建模\outputs\plant_semantics_detail_trial' `
  -OutputDirectory 'F:\植物建模\outputs\plant_semantics_complete_v1'
```

首次在新输入上应先运行基线及细节阶段，再调用完整标签阶段。`complete_five.py refine` 可复用实例投影缓存，仅重新细化；模型和相机缓存以 SHA256 检查。`inference_config.json` 记录本轮代码、输入标签和实例配置哈希。

完整版本文件：

- `semantic_mesh.ply`：五类纯语义颜色、逐顶点及逐面 label；不存在 label=0。
- `labeled_original_rgb.ply`：原 RGB 与相同标签；原色本身可能是灰色，不代表未分类。
- 两份 PLY 顶点另含 `uncertain`、`inferred` 属性。`inferred=1` 表示上一版本未分类、本版已自动推断；`uncertain` 是诊断标志，不代表该点已知分错。`evidence` 为非校准分数。
- `labels.npz`：五类分数、边界、不确定性、不可见区域传播标志、花苞保护等。
- `comparison_preview.jpg`、`comparison_compact.jpg`：完整与简版前后对比。

保留 1,567,628 个原顶点和 3,136,830 个原三角面，位置、顺序和索引不变。不修改原输入，不等待 SAM3。

对新模型从头自动运行全部阶段并检查导出：

```powershell
& 'F:\植物建模\tools\plant_semantics\run_all_five.ps1' `
  -InputMesh 'D:\plants\new.ply' -OutputDirectory 'D:\results\new' -UpAxis y
```

最终文件在指定输出目录的 `complete/semantic_mesh.ply`。可指定 `-PythonExecutable` 和 `-SamCheckpoint` 迁移环境；需先安装依赖、下载模型。这个入口保证执行五类全覆盖检查，不能据此保证新植物的分割准确率。
