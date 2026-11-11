# 已有验证与适用范围

## 当前植物

- 原模型：`F:\植物建模\outputs\Plant_cropped.ply`，1,567,628 个顶点，3,136,830 个三角面。
- 当前认可效果：`F:\植物建模\outputs\plant_semantics_complete_v2\semantic_mesh.ply`。
- 顶点、面未知标签均为 0 个。坐标、顶点顺序、三角面索引、标签、颜色及不确定性序列化检查通过，记录在同目录 `validation.json`。
- 叶缘修复相比 v1 将 11,324 个顶点改为叶片，其中茎→叶 7,379、花→叶 3,945。已保护的花苞、盆、基质以及原有叶片不被该步骤改写。52 次掩膜尝试接受 51 次。
- 以上改动数和掩膜质量不等于语义准确率。用户已进行视觉检查，尚无人工真值 IoU。

## 0.2.2 窄叶缘补全（v3）

- 新结果：`F:\植物建模\outputs\plant_semantics_complete_v3\semantic_mesh.ply`；v2 仍保留用于对照。
- 在 v2 上单独执行新增阶段，从 2,986 个近叶面候选顶点中将 2,534 个茎标签点改为叶片，约占总表面积 0.1706%。不使用人工标注、固定颜色门槛或新模型权重。
- `leaf_edge_side_comparison.jpg` 展示用户反馈区域的两侧对比；`leaf_rim_comparison.jpg` 提供自动选择的局部视角。可见橙色叶缘减少、附近主要枝干仍保留；连接处少量误差仍可能存在。
- `validation.json` 确认 1,567,628 个顶点、3,136,830 个面均有标签，坐标、顺序、索引、颜色及标签/不确定性序列化检查通过。
- `rim_audit.json` 确认仅茎→叶发生改变；原有叶、花、盆、基质与受保护花苞的顶点标签不变。新增改动点全部保留不确定标记。
- 此次视觉与程序检查不提供 IoU 或跨植物精度保证；2,534 是改动数量，不是已证明正确的数量。

## 0.2.3 v3 不确定标记复核

结果为 `outputs/plant_semantics_complete_v3_reviewed/`。原始 v3 的 477,072 个历史标记中，85,985 个通过独立相机证据与最终分数复核，剩余 391,087 个，占总顶点 24.95%。基质剩余 6,901 / 17,541 个。语义标签没有改变，降低标记数不能解释为精度提高。

`uncertainty_audit.json` 确认全部原 NPZ 字段除 `uncertain` 外逐项完全相同，原始标记完整保留。此步骤不加载任何模型，不调用或读取 SAM3 权重/预测。详细规则见 [UNCERTAINTY.md](UNCERTAINTY.md)。

复核版通过原模型几何、标签、颜色和元数据导出校验。21 项回归与 PowerShell 语法检查通过；完整 `run_auto.ps1 -UpAxis auto` 在同一小模型及既有二维缓存上复跑成功，包含新增 `complete_before_uncertainty/` 与最终复核阶段，日志为 `outputs/plant_auto_smoke2/uncertainty_integration.log`。这些检查不证明语义精度。

## 0.2.4 v3 新增视角复核

`outputs/plant_semantics_v3_extra_views/` 保存 14 个新增相机及原 v3 三模型的二维推理，最终模型位于其 `complete/`。24 个候选方向按实际不确定表面覆盖选择，并补足盆内俯视角；132,008 个原标记点能被至少 3 个所选新视角看到，但可见性本身不等于语义确认。

新增证据实际解除 75,475 个历史标记，并对原来未标记的 7,440 个点新增冲突标记，合计 391,087 → 323,052（24.95% → 20.61%）。基质 6,901 → 7,271；不能声称所有类别都有改善。`extra_view_audit.json` 全部原字段不变检查通过；几何、标签、颜色与当前标记导出检查通过。规则和逐类结果见 [UNCERTAINTY.md](UNCERTAINTY.md)。

本步骤使用 GroundingDINO-Tiny、CLIPSeg、MobileSAM，与原 v3 的权重哈希、提示和推理参数一致。未使用 SAM3，不改语义标签，没有新增真实扫描。29 项回归与 PowerShell 语法检查通过；这些检查不衡量语义准确率。

## 0.2.5 交付整理与入口

本次未改算法、阈值或既有模型。新增 `run_v3.ps1` 将原分割和加视角入口串联，并在成功后写实际结果路径表；`-ExtraViews 0` 跳过新增视角。原算法阶段已分别完成上面的真实推理与导出验证，此次不重复运行大模型推理。

入口验证在包含中文、空格的临时目录中，使用轻量阶段替身调用真实包装脚本，检查两种最终目录、参数转发、正确的最终不确定报告及上游失败即停止。三种情形通过，记录为 `outputs/v3_delivery_check/entrypoint_validation.json`；它验证调度和路径，不衡量语义质量。

30 项回归及 PowerShell 语法检查通过。V3 源码包检查包括入口/文档完整、排除独立 SAM3 与历史实验、逐文件哈希、无权重/本机环境以及解压后重新打包一致性。完整维护包继续保留独立历史入口。

## 独立自动流程验证记录

小网格 `cropped_models/Plant_nomip_crop_attempt.ply` 含 1,437 顶点、2,473 面，完整入口执行并通过最终导出检查，输出在 `outputs/plant_auto_smoke2/complete/`。该输入很粗糙，99.93% 顶点被记为不确定，仅用于验证调度、无盆口分支和序列化。

2026-10-20 整理后，用同一小模型和已有二维缓存复跑 Windows PowerShell 的 `run_auto.ps1 -UpAxis auto`，包含叶缘阶段与最终校验，执行成功。日志为 `outputs/plant_auto_smoke2/code_organization_smoke.log`。13 项回归测试、PowerShell 语法检查和解压后核心导入/帮助检查通过；这些检查均不衡量语义准确率，也不验证独立 SAM3 试验的推理效果。

0.2.2 再次执行同一完整入口，新增 `complete_before_rim/` 和窄叶缘阶段均成功，最终导出校验通过；日志为 `outputs/plant_auto_smoke2/rim_integration.log`。最终将支持门槛调整为 0.40 后，单独复跑新增阶段和导出校验，通过；日志为 `outputs/plant_auto_smoke2/rim_final_threshold.log`，该阶段改变 59 个顶点，仅说明执行链路可用。最终参数下 17 项回归及 PowerShell 语法检查通过。

## 仍需独立评价

目前没有跨物种、跨花色或枯黄叶片数据集评价。细茎、绿色花苞、叶柄/花梗交接、极细叶缘、盆内基质及重建粘连处可能错分。多盆、长方盆、任意姿态、无 RGB 或严重缺损输入不在当前已验证条件内。

后续评价应在独立模型上报告逐类 IoU/召回及边界质量，不能把“所有顶点已有标签”或低不确定比例作为准确性证明。完整流程与输入要求见 README 和 `ALGORITHM.md`。
