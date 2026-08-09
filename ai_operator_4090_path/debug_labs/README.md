# 故障驱动调试实验代码

这里的代码只服务于调试训练，不属于24个正式算子接口。

- `diagnose_extension.py`：检查实际加载的扩展、源码/二进制时间、导出符号、动态库和GPU架构。
- `build_fault_lab.py`：在临时目录中演示旧二进制、漏编译源码、缺失导出和架构配置的诊断方法。
- `stream_device_lab.py`：复现缺少Event的跨stream数据竞争，并验证current stream、Event和Device Guard。
- `pipeline_trace.py`：用NVTX区分输入准备、隐藏拷贝、隐藏同步、内部dtype转换和kernel调用。
- `fault_extension/`：与正式扩展隔离的故障kernel，只能通过独立子进程和Compute Sanitizer运行。
- `unknown_fault_lab.py`：从未知现象开始完成最小复现、分层、假设、工具证据、根因和回归测试。

4090服务器先执行预检，再按实验编号运行：

```bash
bash scripts/run_debug_experiment.sh preflight
bash scripts/run_debug_experiment.sh ENG-C01
bash scripts/run_debug_experiment.sh DBG-T01
bash scripts/run_debug_experiment.sh help
```

`run_debug_experiment.sh`覆盖第9节和第11节的24个实验编号，并把日志统一写到`reports/debug_labs/`。`50_debug_labs.sh`是它的底层工具入口，仍可用于单独运行stream case、NSYS scenario或Sanitizer fault。

`ENG-D01`中的异步报错对照使用独立的`illegal-address`故障，并分别在`CUDA_LAUNCH_BLOCKING=0/1`的新进程执行；`DBG-T01`继续使用单元素OOB供memcheck推导边界地址。两个故障不能混为同一项证据。

若`DBG-S02`返回`INCONCLUSIVE`，统一入口会非零退出。可增加producer延迟后重跑：

```bash
STREAM_SLEEP_CYCLES=400000000 bash scripts/run_debug_experiment.sh DBG-S02
```

无GPU的本地环境只能检查代码、入口和实验覆盖关系：

```bash
python debug_labs/preflight.py --host-only
```

故障扩展产生的`.so`和构建目录已忽略。运行OOB后不要在同一个Python进程继续其他实验。
