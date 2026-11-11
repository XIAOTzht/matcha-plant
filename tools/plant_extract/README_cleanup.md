# 清理当前 30° 实验的 plus 网格

不重跑训练，读取已经提取成功的 tetra_meshes_nointerp/tetra_mesh_binary_search_7.ply。
输入数据集 input_plus 的遮罩先按原始相机与 mast3r_sfm 导出相机的内参变换，
再做多视角投影筛选。这一步修正中心裁切和焦距重映射，不能修正错误的三维位姿。
仅适用于本次固定相机位姿、零畸变 PINHOLE 实验；位姿不匹配则停止。

本机 PowerShell 上传：

```powershell
$ToolDir = "F:\植物建模\tools\plant_extract"
$UploadFiles = @("auto_extract_plant.py", "clean_turntable_mesh.py", "run_clean_turntable.sh", "requirements.txt") | ForEach-Object { Join-Path $ToolDir $_ }
scp -C @UploadFiles "fyp24_haoteng_zhang@10.7.88.77:/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant/tools/plant_extract/"
```

服务器终端：

```bash
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
"$ROOT/envs/matcha/bin/python" -m pip install -r "$ROOT/tools/plant_extract/requirements.txt"
tmux new-session -d -s plant_cleanup "bash '$ROOT/tools/plant_extract/run_clean_turntable.sh'"
tail -f "$ROOT/logs/turntable30_cleanup.log"
```

输出位于 plus/plant_cleanup_日期_时间/，包含：

- plant.ply：筛选后的原始网格几何和顶点颜色。
- mask_alignment.jpg：12 张实际重建照片上的遮罩对齐图；红色区域为背景，植物及花盆应保留原色。
- plant_preview.jpg：三维顶点投影前后对照，抽样显示。
- report.json：规模、参数、耗时和警告。
- aligned_masks/ 和 mask_mapping.json：变换后的遮罩及其坐标变换。

若植物仍重叠/缺失，应检查近似相机位姿和前期深度，不要将裁剪结果等同于重建质量。
多视角轮廓筛选不能分离轮廓内的所有杂面，也不能恢复缺失叶片。

验证：三个针对性测试覆盖中心裁切、焦距重映射、投影一致性及位姿不匹配时报错。
当前真实 plus 网格仍在服务器；尚未验证本次真实模型清理效果。
