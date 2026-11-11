# 12 张照片、每隔 30° 的对照实验

对应关系：img_000.jpg=0°，img_001.jpg=30°，一直到 img_011.jpg=330°。
使用之前上传的白底抠图照片，保留 MAtCha、当前 mip_filter 实现和默认训练参数。
源代码仓库和旧模型不修改。输出放在新的时间戳目录。

## 必须理解的假设

角度来自用户提供，但内参和旋转轴没有实测标定。此次构造位姿采用：

- 方形像素、主点在图片中心、忽略镜头畸变。
- 焦距从之前 plant_smoke_segmented/mast3r_sfm 的 COLMAP 相机读取，按图像尺寸换算。
- 转轴在画面中近似竖直，摄影机向下俯视 15°，旋转轴方向无横向倾斜。
- 旋转轴通过画面中心射线上的点，轨道半径设为 1（任意尺度）。
- 植物刚性旋转，中心不平移，相机和焦距不变。

因此这叫“已知角度＋近似外参的对照实验”，不是完成了真实相机标定。
如果植物不在中心、花盆移动、相机俯角差异大或焦距估计错误，严格固定的位姿仍可能产生坏模型。
两种旋转方向都运行，结果分别在 plus 和 minus；并不根据训练 PSNR 自动判断哪一个几何正确。
实际相机/转轴标定可替换这些假设。改变俯角可编辑 run_turntable30.sh 中的 --assumed-elevation 15。

前景 SIFT 匹配试验未能可靠覆盖全部视角，因此没有把自动拟合结果作为已标定位姿。
保留可选的匹配拟合功能用于诊断，默认服务器脚本明确使用近似参数。

## 本机 PowerShell 上传

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "F:\植物建模\tools\plant_extract\upload_turntable30.ps1"
```

只上传新代码和已有遮罩；服务器照片仍位于 data/plant_smoke_segmented/images。
SSH/SCP 会要求密码。当前没有免密登录，因此代码准备好后需要用户执行上传、启动。

## 服务器启动

```bash
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
"$ROOT/envs/matcha/bin/python" -c "import cv2,numpy,scipy,PIL,torch; print('Dependencies OK')"
tmux new-session -d -s matcha_turntable30 "bash '$ROOT/tools/plant_extract/run_turntable30.sh'"
```

不要重复启动同名会话；脚本按顺序在物理 GPU 1 上运行两次完整流程。
请先确认 GPU 1 没有其他计算任务，输出盘有足够空间。原始大小不是耗时/内存的保证。

```bash
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
LATEST_LOG=$(ls -t "$ROOT"/logs/plant_turntable30_*.log | head -n 1)
tail -n 60 "$LATEST_LOG"
```

每个方向会输出成功标记；仅 tmux 会话消失不代表成功。

## 结果

```text
outputs/plant_turntable30_日期_时间/
  input_plus/diagnostics/camera_orbit.png
  input_plus/turntable_report.json
  input_minus/diagnostics/camera_orbit.png
  input_minus/turntable_report.json
  plus/mast3r_sfm/points.ply
  plus/mast3r_sfm/FIXED_POSES_VERIFIED.txt
  plus/tetra_meshes/tetra_mesh_binary_search_7.ply
  minus/mast3r_sfm/points.ply
  minus/tetra_meshes/tetra_mesh_binary_search_7.ply
```

可在每个方向的 SfM 完成后先查看 points.ply。camera_orbit.png 是输入的假定相机轨道示意，
不能作为位姿真实准确的证据；FIXED_POSES_VERIFIED.txt 只证明优化和导出遵守了输入位姿。
本次没有额外引入训练遮罩损失，白底可能继续产生表面，以便与上一轮保持可比。

## 为什么不直接使用官方 posed.yaml

当前代码实际会优化 translation 和 size，最后对相机及点云做近似对齐；
相机深度偏移与 size 耦合，所以仅设 opt_tran=False 也不足以固定完整位姿。
run_fixed_pose_sfm.py 仅在当前进程内调整 make_K_cam_depth，使生成点云、计算匹配损失和导出
都使用同一组输入 camera-to-world 位姿。深度及其尺度仍可优化。不进行最后的相机对齐。
这是针对这次实验的实现适配，不能宣称是未经修改的官方完整流程。

run_checked_stage.py 让官方包装脚本的子进程报错能够终止整个任务，避免前一阶段失败后继续跑。

## 已做验证和待验证项

测试了 30° 步长、固定相机下物体旋转和等效相机变换的数学一致性、轨道半径、COLMAP 二进制导出读取、
运行时改动位置和 Python/PowerShell 脚本语法。服务器 GPU 完整流程尚未运行，最终几何质量尚未验证。

```bash
python -m unittest discover -s tools/plant_extract -p 'test_turntable30.py' -v
```
