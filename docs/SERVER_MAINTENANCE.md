# 服务器最终更新和整理

以前的分步命令无需执行。仿真暂不上传。本地代码源为 `F:\plant\matcha_plant`。

## 第一步：Windows 上传

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "F:\plant\matcha_plant\maintenance\upload_plant_contact.ps1"
```

输入服务器密码。完成标记 `PLANT_CONTACT_UPLOAD_INSTALLED`。
上传脚本自动检查、解压和安装，并备份被替换的旧代码；无需手动解压。
不上传本地的 Linux 环境，不安装依赖，不上传仿真。

## 第二步：服务器执行

```bash
ROOT=/Data_PHD_Backup/FYP_Temp/fyp24_haoteng_zhang/matcha_plant
bash "$ROOT/maintenance/finish_server.sh" --apply
```

保持终端连接。脚本会依次检查环境和语义输入、生成新版花叶实例和权重、
备份旧内容、逐文件校验、从活动目录删除已备份的旧内容，最后再检查目录。
当前重建结果直接复用，不重新训练或语义推理。
只有完整新权重生成成功并且旧内容备份验证通过，才允许清理。
新权重文件夹也保存本次使用的两份权重配置。
自动器官候选仍需查看预览复核；本操作不运行机械臂仿真。

最终成功标记：`FINAL_SERVER_WORKFLOW_READY`。
总入口和整理程序使用原子目录锁，兼容 NFS，不依赖 `flock` 服务。
正常退出、任务失败时释放目录锁；强制杀进程或断电可能留下锁目录。
如提示目录锁已存在，请先查看其中的主机、PID 和启动时间，确认原任务已结束再清除。
执行日志在 `backups/finish_server_<时间>_<PID>.log`，清理不会删除它。
如中断或失败，查看该日志；不要手动把 `.partial` 改名当成功备份。
权重生成失败不会触发清理；备份或校验失败不会删除原文件。
如果删除阶段本身失败，完整已验证备份仍保留，逐项完成记录在 `manifest.json`。
修复失败原因后可以重跑同一入口；它会产生一份新的权重结果。

## 备份与保留范围

详见 [最终目录](FINAL_LAYOUT.md)。
旧内容备份位于 `backups/final_cleanup_<时间>/old_content.tar`，
附 `files.json`、`manifest.json`、SHA256 和 `BACKUP_VERIFIED`。
使用未压缩 TAR，减少重复压缩已有模型的 CPU 开销；需要足够备份空间。
生成清单、打包、逐文件核验和检查原文件均需读取磁盘，NFS 可能较慢。
扫描、打包、校验阶段会打印进度，不能用压缩包大小代替完成标记。

## 找到最新模型

```bash
cat "$ROOT/outputs/with_car_20261021/latest_contact_paths.txt"
cat "$ROOT/docs/SERVER_RESULT_INDEX.json"
```

`heatmap` 是彩色权重模型；`weight_arrays` 是实际数值与不可碰撞标记。
`weighted_mesh` 保留原颜色及权重字段。
最新结果的内部 `logs/` 会保留作为运行记录。

## 在服务器导出 GitHub 源码

可以从服务器直接上传 GitHub。先整理成功，再导出不含环境、模型和结果的源码快照：

```bash
"$ROOT/envs/plant_semantics/bin/python" "$ROOT/maintenance/server_project.py" export-github
```

输出为 `exports/github_source_<时间>/`。查看 `SOURCE_EXPORT_MANIFEST.json`，
保留上游许可证，检查是否有不应公开的内容。
GitHub 地址尚未提供，因此本次不创建远端、不提交、不推送。
源码快照不包含大模型与编译环境；完整迁移应使用服务器项目连同其运行资产。

仅查看整理计划（需要已经生成最新权重）可运行：

```bash
"$ROOT/envs/plant_semantics/bin/python" "$ROOT/maintenance/final_organize.py" --root "$ROOT"
```

旧的 `prepare_server_project.sh` 仅做保守临时文件整理，不能代替本次总入口。
