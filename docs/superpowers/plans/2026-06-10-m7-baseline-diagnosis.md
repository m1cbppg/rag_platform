# M7 全量基线评测与错误归因实施计划

> **执行要求：** 按任务顺序实施，新增行为必须先写失败测试，再写最小实现，最后执行全量验证。

**目标：** 在 M6 自动化评测执行器基础上，实现逐层错误归因、中文基线报告，并完成 180 道 DEVELOPMENT 题的 V0 全量评测。

**架构：** 使用评测适配器记录融合、精排和最终 Context 三个阶段的诊断数据；使用纯规则归因引擎定位主失败环节；使用独立报告生成器从数据库重建 Markdown 和 JSON 报告。归因和报告不修改业务 RAG 行为。

**技术栈：** Python 3.12、Pydantic、SQLAlchemy、MySQL 8、pytest、Markdown、JSON。

---

### 任务 1：增强评测观测数据

**文件：**
- 修改：`src/rag_platform/evaluation/rag_adapter.py`
- 修改：`tests/evaluation/test_rag_adapter.py`

- [ ] 编写失败测试，要求 Hybrid 融合 Hit 保存 `sources`、BM25/Vector 排名和原始分数。
- [ ] 验证失败原因是当前 Adapter 丢弃了 `WorkflowDocumentResponse.metadata`。
- [ ] 修改融合 Hit 的 metadata 合并逻辑。
- [ ] Hybrid 分数保存到 `fused_score`，单路检索分数保存到 `raw_score`。
- [ ] 运行 Adapter 测试并确认通过。

### 任务 2：实现纯规则错误归因引擎

**文件：**
- 新建：`src/rag_platform/evaluation/failure_attribution.py`
- 新建：`tests/evaluation/test_failure_attribution.py`

- [ ] 编写召回完全缺失测试。
- [ ] 编写多事实召回部分缺失测试。
- [ ] 编写精排淘汰和 Context 淘汰测试。
- [ ] 编写错误拒答、错误回答和缺少澄清能力测试。
- [ ] 编写引用、完整性、忠实度、相关性和冲突处理测试。
- [ ] 定义归因代码、中文名称、建议和逐题归因模型。
- [ ] 实现阶段 Chunk/Fact Coverage 计算。
- [ ] 实现上游优先的主因选择和次因收集。
- [ ] 运行归因测试并确认通过。

### 任务 3：增加诊断数据读取

**文件：**
- 修改：`src/rag_platform/evaluation/dataset_repository.py`
- 修改：`tests/evaluation/test_dataset_repository.py`

- [ ] 扩展逐题结果查询，读取问题、参考答案、Judge 理由。
- [ ] 增加按 Run 读取 Retrieval Hit 的方法并解析 metadata JSON。
- [ ] 增加按 Run 读取 Gold Evidence 的方法。
- [ ] 在 MySQL 集成测试中验证三类数据能够按 case_result_id/case_id 对齐。
- [ ] 运行 MySQL 集成测试并确认通过。

### 任务 4：实现基线报告构建器

**文件：**
- 新建：`src/rag_platform/evaluation/baseline_report.py`
- 新建：`tests/evaluation/test_baseline_report.py`

- [ ] 编写报告汇总测试。
- [ ] 汇总主归因数量和占比。
- [ ] 汇总融合、精排、Context 三阶段平均 Fact Coverage。
- [ ] 按题型和难度汇总通过率、Recall、Fact Coverage 和 Judge 通过率。
- [ ] 按失败数量生成 M8 优化优先级。
- [ ] 生成中文 Markdown，包含核心指标和典型失败题。
- [ ] 生成完整 JSON 数据结构。
- [ ] 运行报告测试并确认通过。

### 任务 5：实现报告命令行入口

**文件：**
- 新建：`scripts/generate_eval_report.py`
- 新建：`tests/evaluation/test_generate_eval_report_script.py`

- [ ] 编写 Run Code 校验和输出路径测试。
- [ ] 根据 Run Code 读取运行、逐题结果、Hit 和 Gold Evidence。
- [ ] 调用归因引擎和报告构建器。
- [ ] 原子写入 `evaluation/reports/<run_code>.json`。
- [ ] 原子写入 `evaluation/reports/<run_code>.md`。
- [ ] 输出报告路径和归因摘要。
- [ ] 运行命令行测试并确认通过。

### 任务 6：执行 V0 DEVELOPMENT 全量基线

**运行命令：**

```bash
.venv/bin/python -B scripts/run_rag_evaluation.py \
  --dataset rag_eval_ecommerce:v1 \
  --split development \
  --experiment-version V0 \
  --experiment-name baseline-hybrid-rrf-rerank \
  --run-code V0_DEVELOPMENT_20260610 \
  --top-k 20 \
  --concurrency 3
```

- [ ] 确认计划题数为 180。
- [ ] 运行实验；如外部服务或单题失败，使用相同 Run Code 续跑。
- [ ] 查询数据库确认 Run、Case Result、Retrieval Hit 和 Judge 数量。
- [ ] 确认成功题不会在续跑中重复执行。

### 任务 7：生成并核验中文报告

**运行命令：**

```bash
.venv/bin/python -B scripts/generate_eval_report.py \
  --run-code V0_DEVELOPMENT_20260610
```

- [ ] 生成 Markdown 和 JSON。
- [ ] 人工核查至少一条召回缺失、一条精排或 Context 问题、一条答案质量问题。
- [ ] 确认报告中的计数与数据库一致。
- [ ] 根据失败数量和上游优先原则输出 M8 优化顺序。

### 任务 8：最终验证

- [ ] 执行 M7 聚焦测试。
- [ ] 执行全量 pytest。
- [ ] 执行 MySQL 集成测试。
- [ ] 执行 Python 编译检查。
- [ ] 执行 `git diff --check`。
- [ ] 确认报告文件使用中文且不包含密钥或环境变量值。
