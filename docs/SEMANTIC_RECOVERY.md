# 同照片分类退化修订

> 此实验已停用，未达到认可版本效果。当前默认已恢复旧方法，复现入口及验证结果见 [APPROVED_REPRODUCTION.md](APPROVED_REPRODUCTION.md)。以下仅保留实验记录。

2026-11-09：上传包中的器官修正代码与旧认可版本一致，但重新建模后的初始分类变化很大。新增恢复逻辑使用原始 RGB、独立多视角实例和几何限制，不迁移旧植物标签，不使用指定植物坐标。

修订内容：

- 将上板坐标系修正为右手系，与原始语义适配器一致，避免某些重建坐标系产生镜像推理图。
- 用局部长条几何自动选择细杆近景，即使原标签全部为花也会进入识别；局部图与主图共享一票。
- 用独立 RGB 枝干证据与双尺度形状共同恢复细杆，并在表面一致性和叶缘处理时保护这些证据。形状本身不能把叶缘改成枝干。
- 增加独立土壤提示词和盆口近景。只允许盆内已有软类表面在至少3个父视角、其中至少2个高视角支持时改为泥土；不填洞、不把盆壁改成土、不把虚拟土面伪装成实测。
- 保留小车和虚拟土面来源文件；原始硬类顶点和三角面保持不变。更新的类别仍需人工看图。
- 路径报告增加 `motion_style_review`，单独说明是否回退原路径、肘部是否达到30°、腕部是否达到40°。原有接触上限与已认可的协同选择算法不变。

旧模型有2,024个土面顶点和3,906个土面三角面，但单独查看是盆口附近稀疏窄片，不能据此确认完整土面。新模型排除虚拟面后，旧土面约72.8%的面积能在2 mm内找到对应表面，主要被标为花；这证明分类差异，并不证明旧标签一定正确。核查结果位于 `outputs/diagnostics/result_regression_20261109`。

## 服务器只重做分类和权重

先从 Windows 上传代码：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\plant\matcha_plant\maintenance\upload_plant_pipeline.ps1"
```

然后仍在 Windows PowerShell 执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\plant\matcha_plant\maintenance\start_semantic_review.ps1" -RunName "plant_20261108_virtual_soil_v1" -Revision "plant_20261109_recovery_v1"
```

此命令在服务器创建独立语义修订，不重建照片、不复用旧推理证据、不覆盖现有结果，不开始规划。保留已有假设土面作为仿真输入。报告会分别记录新恢复的原始表面和原来的虚拟面。

服务器日志：`$ROOT/logs/semantic_plant_20261109_recovery_v1.log`。完成标志：`$ROOT/outputs/semantic_refinement/plant_20261109_recovery_v1/SEMANTIC_REVISION_EXPORTED`；该标志只说明导出一致性通过，不说明语义准确。

检查修订目录中的 `01_semantics/attempt_001/result/semantic_surface_comparison.jpg`、`stem_recovery_report.json`、`substrate_recovery_report.json` 及 `02_weights/attempt_001/result/weights/contact_risk_heatmap_preview.jpg`。分类检查之后才重新规划；旧轨迹不能直接复用。

本地测试使用模型后端替身验证投票、几何限制、硬类保护和导出一致性。Windows 当前仿真环境缺少 torch/Open3D，未执行真实神经网络识别；服务器推理效果尚待本轮修订验证。
