# 关系定位监督：静态重建与投影契约审计

日期：2026-10-08。**STATIC_RECONSTRUCTION_NOT_EXACT**。

只使用旧fit的13个场景，每场景按已有manifest顺序取首条分支的初始帧和root帧，共26帧。没有访问新开发或确认数据；没有执行控制动作或训练模型。

将保存qpos/qvel恢复到同任务环境并forward，只检验静态成像，不声称恢复controller/contact求解器历史。相机矩阵逐场景与已验证agentview契约比对；矩阵投影与独立几何公式互验，3D→128px→64px→3D闭合检查。

重建RGB逐位相同：0/26。可见grip site紧致标记：23帧；原1/1.5/2像素median/p95/max标准：True。

最大body位置重建差：0.414145 mm；最大grip位置重建差：0.422913 mm。

## 坐标与标签含义

保持原RGB方向，不翻图；只转换投影v为127−v_raw。2×2平均缩放后的像素中心为(u−0.5)/2，不是简单u/2。圆圈表示两个碗、盘的body原点及grip site，编号0/1/2/3；body原点不是可见区域质心，也不保证无遮挡。

投影正确不代表图像中可观察深度，更不证明body原点可直接通过分割质心学到。未来监督应分开投影原点、可见mask与深度/相对几何，不能混用标签。

静态重建未满足原图精确一致，不能将重建分割直接贴回原RGB作为像素真值。先定位forward/渲染状态差异；本轮不据此训练定位模型或宣称视觉失败。

投影示意：[projection_overlays.png](../results/relational_localization_contract/projection_overlays.png)。完整每帧记录：[evaluation.json](../results/relational_localization_contract/evaluation.json)。
