# Candidate Utility Oracle：执行状态

阶段1已启动。正式候选收集仍在运行，**尚无完整科学结论**。

协议和实现于首次仿真前冻结到提交 `4f5cf66`。10项针对性测试通过，包括数值访问边界、候选子集、T1选择规则、300步预算、policy输入白名单，以及使用独立合成数据核验完整analysis→decision→report流程。合成测试不计入科学实验结果。

技术preflight已通过：两个预定root各运行三次A0，起点RGB/proprio一致、sim-state满足容差，完整continuation的动作hash和success记录一致。仿真检查耗时106.37秒，见[preflight](../results/candidate_utility_oracle/preflight.json)。

本轮预注册10任务×init31–34，共40个root，每root64候选；H8后使用冻结C0继续至episode总300步。数据仅用于calibration诊断。未打开HDF5，也未训练encoder、value或WM。

收集进程由独立监督进程持续运行。每8条分支保存一个断点，每root复验A0与预声明的selected/rescue候选。若技术检查失败，记录blocker并停止；不会生成科学正负结论。全部完成后自动运行分析，生成metrics/decision，并以正式结果替换本文；下一阶段不自动启动。

- [实时运行状态](../results/candidate_utility_oracle/runtime.json)
- [日志](../results/candidate_utility_oracle/driver.log)
- [注册协议](../configs/candidate_utility_oracle_protocol.json)
- [Root清单](../configs/candidate_utility_oracle_roots.json)
- [访问记录](../results/candidate_utility_oracle/access_ledger.json)

发生进程中断时，可从项目目录续跑同一协议：

```bash
env -u PYTHONPATH LIBERO_CONFIG_PATH="$PWD/.libero" MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/run_candidate_utility_oracle.py
```

监督进程持有文件锁，避免重复并发收集。已完成分支不会被重算覆盖；续跑必须通过protocol、candidate及artifact hash校验。
