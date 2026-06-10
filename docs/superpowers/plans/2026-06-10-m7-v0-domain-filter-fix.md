# M7 V0 业务域过滤修复实施计划

> **执行要求：** 使用 TDD，先证明当前单值过滤无法处理顶层业务域，再实现最小修复。

**目标：** 修复顶层业务域导致的零召回问题，并重新建立正式 V0 DEVELOPMENT 基线。

**架构：** 使用独立业务域解析器统一 ES 与 Milvus 的过滤语义。顶层域解析为细分域集合，具体域保持精确过滤；报告诊断复用同一规则。

**技术栈：** Python 3.12、Elasticsearch、Milvus、Pydantic、pytest。

---

### 任务 1：业务域解析器

**文件：**
- 新建：`src/rag_platform/rag/retrieval/business_domain.py`
- 新建：`tests/rag/retrieval/test_business_domain.py`

- [ ] 测试顶层域展开为全部电商售后细分域。
- [ ] 测试具体细分域保持单值。
- [ ] 测试空值不产生过滤条件。
- [ ] 实现不可变映射和去重解析。

### 任务 2：Elasticsearch 多业务域过滤

**文件：**
- 修改：`src/rag_platform/infrastructure/elasticsearch_store.py`
- 新建：`tests/infrastructure/test_elasticsearch_store.py`

- [ ] 测试顶层域生成 `terms` 查询。
- [ ] 测试具体域仍生成单元素 `terms` 查询。
- [ ] 测试空业务域不添加业务域条件。
- [ ] 抽取可测试的 Filter 构建函数。

### 任务 3：Milvus 多业务域过滤

**文件：**
- 修改：`src/rag_platform/rag/retrieval/vector_retriever.py`
- 新建：`tests/rag/retrieval/test_vector_filter.py`

- [ ] 测试顶层域生成 Milvus `business_domain in [...]`。
- [ ] 测试具体域生成单值等式。
- [ ] 对字符串进行安全转义。
- [ ] 检索主流程调用统一表达式构建函数。

### 任务 4：报告诊断适配

**文件：**
- 修改：`src/rag_platform/evaluation/dataset_repository.py`
- 修改：`src/rag_platform/evaluation/baseline_report.py`
- 修改：相关测试

- [ ] 诊断数据增加解析后的 Case 业务域。
- [ ] 只有解析后的域与 Chunk 域无交集才报告冲突。
- [ ] 保留原始域和解析结果，便于复现。

### 任务 5：真实验证与正式 V0

- [ ] 运行一条真实 DEVELOPMENT 题，确认产生检索 Hit。
- [ ] 运行 5 条题的小批量 Smoke。
- [ ] 使用 `V0_DEVELOPMENT_BASELINE_20260610` 运行 180 条题。
- [ ] 生成 Markdown 和 JSON 报告。
- [ ] 核验 Case Result、Retrieval Hit 和 Judge 数量。
- [ ] 全量测试、MySQL 集成、编译和格式检查。
