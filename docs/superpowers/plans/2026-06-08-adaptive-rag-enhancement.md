# Adaptive RAG 项目增强实施计划

> **实施要求：** 按任务顺序执行。每项功能先写失败测试，再编写最小实现，
> 测试通过后才能进入下一项。建议使用 `superpowers:subagent-driven-development`
> 或 `superpowers:executing-plans` 按任务推进。

**目标：** 构建包含 40 篇电商售后文档和 300 道评测题的可复现评测体系，
并把现有 RAG 从 V0 基线逐步升级为具备自适应检索、多跳检索和 Claim 级
证据校验能力的 V3 系统。

**架构原则：** 保留当前 FastAPI、Application、Repository、RAG 的分层结构。
评测能力放在独立包中；Adaptive Retrieval 通过小型独立组件接入现有
LangGraph。每次改变系统行为前后，都必须在同一冻结数据集上运行实验。

**技术栈：** Python 3.12、FastAPI、Pydantic、SQLAlchemy、MySQL 8、
Elasticsearch、Milvus、LangGraph、DeepSeek、DashScope Qwen、pytest、
python-docx、reportlab。

---

## 一、不可违反的实施规则

1. V0 基线结果保存前，不修改现有检索逻辑。
2. V0、V1、V2、V3 必须使用同一份冻结测试集。
3. DeepSeek 负责生成，Qwen 负责独立审核和评分。
4. `reference_answer` 绝不能传给被测 RAG。
5. 新功能必须先写失败测试。
6. 每次实验必须把完整配置写入 `rag_eval_run.config_json`。
7. 对外报告效果时，必须同时给出基线值、最终值和提升量。
8. 开发集用于调试和阈值搜索，验证集用于选配置，测试集只用于最终报告。

## 二、文件落点总览

### 已准备的 SQL

- `src/sql/010_create_rag_evaluation_tables.sql`
  - 数据集、源文档、评测题、标准证据、实验运行和评测结果。
- `src/sql/011_create_rag_adaptive_trace_tables.sql`
  - Adaptive Retrieval 运行追踪和 Claim 证据校验。
- `src/sql/012_add_chunk_experiment_metadata.sql`
  - Chunking 消融实验使用的策略元数据。

### 需要新增的评测目录

```text
evaluation/
  blueprints/
    ecommerce_document_plan.json
    ecommerce_case_plan.json
  corpus/
    source/
    rendered/
    manifest.jsonl
    catalog.json
  datasets/
    rag_eval_v1.generated.jsonl
    rag_eval_v1.reviewed.jsonl
    rag_eval_v1.frozen.jsonl
  reports/
  prompts/
    document_generate.txt
    document_review.txt
    case_generate.txt
    case_review.txt
    answer_judge.txt

scripts/
  generate_eval_corpus.py
  review_eval_corpus.py
  render_eval_corpus.py
  ingest_eval_corpus.py
  generate_eval_cases.py
  review_eval_cases.py
  map_eval_evidence.py
  freeze_eval_dataset.py
  run_rag_evaluation.py
  compare_rag_runs.py

src/rag_platform/evaluation/
  __init__.py
  models.py
  dataset_repository.py
  dataset_validator.py
  retrieval_metrics.py
  action_metrics.py
  citation_metrics.py
  judge_client.py
  judge_service.py
  experiment_runner.py
  report_builder.py
```

### 需要新增的 Adaptive Retrieval 目录

```text
src/rag_platform/rag/adaptive/
  __init__.py
  models.py
  quality_features.py
  quality_policy.py
  query_rewriter.py
  query_decomposer.py
  multi_round_fusion.py

src/rag_platform/infrastructure/repositories/
  retrieval_trace_repository.py
```

### 需要新增的 Grounding 目录

```text
src/rag_platform/rag/grounding/
  __init__.py
  models.py
  claim_extractor.py
  evidence_validator.py
  answer_corrector.py
  grounding_service.py

src/rag_platform/infrastructure/
  dashscope_chat.py

src/rag_platform/infrastructure/repositories/
  grounding_repository.py
```

## 三、阶段 0：测试基础和可注入依赖

### 任务 0.1：增加开发及评测依赖

**修改文件：** `pyproject.toml`

新增：

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8,<9",
    "pytest-asyncio>=0.24,<1",
    "pytest-cov>=5,<6",
]
eval = [
    "reportlab>=4,<5",
]
```

第一版不要引入 pandas。JSONL 和 Python 标准库 `csv` 足够完成评测。

执行：

```bash
.venv/bin/pip install -e '.[dev,eval]'
.venv/bin/python -m pytest -q
```

**验收条件：**

- pytest 能正常收集测试。
- 测试收集阶段不连接 MySQL、Milvus、Elasticsearch 或外部模型。
- 导入 FastAPI 应用时不应因为 Milvus 不可用而失败。

### 任务 0.2：把核心服务改造成可注入依赖

**修改文件：**

- `src/rag_platform/application/rag_workflow_service.py`
- `src/rag_platform/application/chat_service.py`
- `src/rag_platform/rag/graph/rag_retrieval_graph.py`
- `src/rag_platform/application/query_understanding_service.py`
- `src/rag_platform/application/rerank_service.py`
- `src/rag_platform/application/context_build_service.py`

构造函数使用可选依赖：

```python
class RagWorkflowService:
    def __init__(self, graph=None) -> None:
        self.graph = graph or RagRetrievalGraphBuilder().build()
```

Repository、模型客户端和 Retriever 使用相同形式。不要额外引入大型
Dependency Injection 框架。

**新增测试：**

- `tests/application/test_rag_workflow_service.py`
- `tests/application/test_chat_service.py`

**验收条件：**

- Fake Graph 可以在没有任何外部基础设施的情况下运行。
- Fake Repository 可以验证写入行为。
- 导入路由不会立刻创建连接 Milvus 的 Service 实例。

## 四、阶段 1：创建评测数据库

### 任务 1.1：执行评测表 DDL

当前阶段只执行：

```bash
mysql -u <用户名> -p <数据库名> \
  < src/sql/010_create_rag_evaluation_tables.sql
```

检查：

```sql
SHOW TABLES LIKE 'rag_eval_%';
```

应存在：

```text
rag_eval_dataset
rag_eval_source_document
rag_eval_case
rag_eval_case_relevance
rag_eval_run
rag_eval_case_result
rag_eval_retrieval_hit
rag_eval_judge_result
```

此时不要执行 `011` 和 `012`。

### 任务 1.2：创建评测领域模型

**新增文件：** `src/rag_platform/evaluation/models.py`

定义枚举：

```python
class EvalCaseType(StrEnum):
    DIRECT = "DIRECT"
    PARAPHRASE = "PARAPHRASE"
    EXACT = "EXACT"
    MULTI_CONDITION = "MULTI_CONDITION"
    MULTI_HOP = "MULTI_HOP"
    CONFLICT = "CONFLICT"
    NO_ANSWER = "NO_ANSWER"


class ExpectedAction(StrEnum):
    ANSWER = "ANSWER"
    REFUSE = "REFUSE"
    CLARIFY = "CLARIFY"
```

定义以下 Pydantic 模型：

- `SourceDocumentSpec`
- `EvidenceSpec`
- `GeneratedEvalCase`
- `ReviewedEvalCase`
- `EvalRunConfig`
- `RetrievalMetricResult`
- `JudgeScore`

`EvidenceSpec` 至少包含：

```python
source_doc_code: str
evidence_quote: str
fact_key: str
relevance_grade: int
```

字段校验：

- `relevance_grade` 只能是 0、1、2、3。
- `MULTI_HOP` 的必要证据至少包含两个不同 `fact_key`。
- `NO_ANSWER` 不允许携带虚构的标准证据。
- `expected_action=ANSWER` 时必须有参考答案。

### 任务 1.3：创建评测 Repository

**新增文件：** `src/rag_platform/evaluation/dataset_repository.py`

实现方法：

```python
create_dataset(...)
save_source_document(...)
map_source_document(source_doc_code, doc_id)
save_eval_case(...)
save_case_evidence(...)
list_reviewed_cases(dataset_id, split)
create_run(config)
start_case_result(run_id, case_id)
save_retrieval_hits(case_result_id, hits)
finish_case_result(...)
save_judge_result(...)
finish_run(...)
```

**测试：**

- Repository 测试连接单独的 MySQL 测试库。
- 使用 `@pytest.mark.integration` 标记。
- 指标函数测试不能依赖数据库。

## 五、阶段 2：生成 40 篇受控电商语料

### 任务 2.1：定义文档蓝图

**新增文件：** `evaluation/blueprints/ecommerce_document_plan.json`

必须包含 40 条文档蓝图。

#### FAQ：12 篇

1. 订单状态与取消
2. 支付失败与重复支付
3. 退款进度
4. 退货运费
5. 优惠券使用
6. 优惠券退回
7. 发票开具
8. 收货地址修改
9. 物流延迟
10. 包裹丢失或破损
11. 会员权益
12. 售后进度

#### SOP：10 篇

1. 仅退款申请处理
2. 退货退款处理
3. 包裹丢失排查
4. 包裹破损处理
5. 错发商品处理
6. 退款失败排查
7. 优惠券投诉处理
8. 发票修改流程
9. 重复支付处理
10. 风控升级流程

#### RULE：12 篇

至少覆盖：

```text
R-ORDER-001
R-REFUND-001
R-REFUND-002
R-COUPON-001
R-COUPON-002
R-LOGISTICS-001
R-INVOICE-001
R-MEMBER-001
```

至少四个规则主题具有新旧两个版本。每个版本包含：

- `version`
- `effective_from`
- `effective_to`
- 被替代规则编码
- 明确例外条件
- 规则优先级

#### MANUAL：6 篇

1. 订单管理后台
2. 退款管理后台
3. 优惠券管理后台
4. 物流管理后台
5. 发票管理后台
6. 风控管理后台

蓝图示例：

```json
{
  "source_doc_code": "RULE_REFUND_001_V2",
  "doc_type": "RULE",
  "title": "售后退款规则 V2",
  "topic": "refund",
  "version": "2.0",
  "effective_from": "2026-01-01",
  "required_facts": [
    "已发货订单退款条件",
    "优惠券订单退款限制",
    "超过七天的例外条件"
  ],
  "required_identifiers": ["R-REFUND-001", "E-RF-1002"],
  "conflicts_with": ["RULE_REFUND_001_V1"]
}
```

### 任务 2.2：用 DeepSeek 生成结构化源文档

**新增文件：**

- `scripts/generate_eval_corpus.py`
- `evaluation/prompts/document_generate.txt`

每次模型请求只生成一篇文档，`temperature=0.4`。禁止一次请求生成 40 篇。

Prompt：

```text
你是企业电商售后知识库文档编写器。

目标文档规范：
{blueprint_json}

要求：
1. 只生成当前文档，不引用不存在的其他规则。
2. required_facts 必须全部出现并保持逻辑一致。
3. required_identifiers 必须逐字出现。
4. RULE 必须包含适用条件、例外、优先级和生效日期。
5. SOP 必须包含适用场景、前置检查、编号步骤、异常分支和升级条件。
6. FAQ 必须包含8到15组问答，每组包含2到4个同义问法。
7. MANUAL 必须包含菜单路径、字段、按钮、操作步骤和错误提示。
8. 不要出现“模拟数据”或生成过程说明。
9. 严格输出JSON，不要输出Markdown。

输出结构：
{
  "source_doc_code": "...",
  "title": "...",
  "doc_type": "...",
  "version": "...",
  "effective_from": "...",
  "effective_to": null,
  "sections": [
    {
      "section_code": "...",
      "heading": "...",
      "content": "...",
      "facts": [
        {"fact_key": "...", "fact_text": "..."}
      ]
    }
  ]
}
```

脚本要求：

1. 使用 `SourceDocumentSpec` 校验模型输出。
2. JSON 非法时最多重试两次。
3. 每篇文档单独保存到
   `evaluation/corpus/source/<source_doc_code>.json`。
4. 计算源内容 SHA256。
5. 文件已存在时默认跳过，仅 `--force` 时覆盖。
6. 写入可恢复执行的 `manifest.jsonl`。
7. 每完成一篇立即落盘，不能全部生成完成后才保存。

执行：

```bash
.venv/bin/python scripts/generate_eval_corpus.py \
  --blueprint evaluation/blueprints/ecommerce_document_plan.json \
  --output evaluation/corpus/source
```

**验收条件：**

- 正好生成 40 个合法 JSON。
- 每个 `required_fact` 和 `required_identifier` 都存在。
- 不存在完全重复的文档。

### 任务 2.3：使用 Qwen 独立审核文档

**新增文件：**

- `src/rag_platform/infrastructure/dashscope_chat.py`
- `scripts/review_eval_corpus.py`
- `evaluation/prompts/document_review.txt`

Qwen Chat 客户端配置：

```text
base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
endpoint: /chat/completions
模型配置键: QWEN_JUDGE_MODEL
默认模型别名: qwen-plus
temperature: 0
```

审核维度均为 0 到 1：

- 文档内部逻辑一致性
- 必要事实覆盖率
- 编号和错误码准确性
- 文档类型结构符合度
- 新旧版本一致性
- 内容歧义风险

通过条件：

```text
所有required_facts均存在
并且 internal_consistency >= 0.90
并且 structure_score >= 0.85
并且 overall_score >= 0.88
```

审核失败时，把 Qwen 的问题列表连同原蓝图发给 DeepSeek 重新生成，
最多允许两轮。每轮审核结果都保存到 `rag_eval_source_document`。

### 任务 2.4：渲染为现有解析器支持的文件

**新增文件：** `scripts/render_eval_corpus.py`

格式：

- FAQ 文档 -> DOCX
- RULE 规则文档 -> DOCX
- SOP 流程文档 -> PDF
- MANUAL 操作手册 -> PDF

使用：

- DOCX：`python-docx`
- PDF：`reportlab`

新增环境变量：

```text
EVAL_PDF_FONT_PATH=/绝对路径/NotoSansCJKsc-Regular.otf
```

渲染后必须调用现有 Parser 重新提取文本，并检查所有
`required_identifier` 是否完整保留。

输出目录：

```text
evaluation/corpus/rendered/<source_doc_code>.docx
evaluation/corpus/rendered/<source_doc_code>.pdf
```

## 六、阶段 3：导入文档并建立映射

### 任务 3.1：编写确定性导入脚本

**新增文件：** `scripts/ingest_eval_corpus.py`

不要通过 HTTP 接口导入，直接调用：

```text
DocumentIngestService
-> ChunkBuildService
-> EmbeddingService
-> SearchIndexService
```

每篇文档执行：

1. 上传、解析、清洗和质量检查。
2. 创建 Chunk。
3. 创建并执行 Embedding Task。
4. 创建并执行 Elasticsearch 索引 Task。
5. 把生成的 `doc_id` 写入
   `rag_eval_source_document.mapped_doc_id`。

`source_doc_code` 必须能够通过标题、扩展字段或文件 SHA256 反向查询到
`rag_document.id`。

**验收条件：**

- 40 篇文档都存在 `mapped_doc_id`。
- 全部完成 Chunk、Milvus 和 Elasticsearch 索引。
- Parser 和 Chunker 没有丢失必要规则编号。

### 任务 3.2：导出语料目录

**新增文件：** `evaluation/corpus/catalog.json`

每篇文档记录：

- `source_doc_code`
- `mapped_doc_id`
- 全部 Chunk ID
- Chunk 标题和正文摘要
- `fact_key`
- 内容 SHA256

该目录是后续评测题生成和证据映射的输入。

## 七、阶段 4：生成、审核并冻结 300 道评测题

### 任务 4.1：定义题型配额

**新增文件：** `evaluation/blueprints/ecommerce_case_plan.json`

```json
{
  "DIRECT": 90,
  "PARAPHRASE": 45,
  "EXACT": 30,
  "MULTI_CONDITION": 45,
  "MULTI_HOP": 30,
  "CONFLICT": 20,
  "NO_ANSWER": 40
}
```

40 道 `NO_ANSWER` 继续划分：

- 20 道 `REFUSE`：知识库确实没有答案。
- 10 道 `CLARIFY`：缺少订单状态、时间等必要条件。
- 10 道 `REFUSE`：问题不属于电商售后业务域。

### 任务 4.2：生成评测题

**新增文件：**

- `scripts/generate_eval_cases.py`
- `evaluation/prompts/case_generate.txt`

一次模型请求最多生成 5 道题，只传入选定源文档，不传入完整语料库。

输出示例：

```json
{
  "case_code": "CASE_MULTI_HOP_001",
  "question": "已发货且使用了满减券，签收九天后还能退款吗？",
  "case_type": "MULTI_HOP",
  "expected_action": "ANSWER",
  "difficulty": "HARD",
  "reference_answer": "...",
  "required_fact_count": 3,
  "target_doc_types": ["RULE"],
  "evidence": [
    {
      "source_doc_code": "RULE_REFUND_001_V2",
      "fact_key": "refund_after_delivery",
      "evidence_quote": "从源文档逐字复制的证据",
      "relevance_grade": 3
    }
  ]
}
```

约束：

1. `evidence_quote` 必须是源文档的逐字子串。
2. 多跳题至少需要两个不同 `fact_key`。
3. 冲突题至少需要两个不同版本或结论冲突的来源。
4. 精确题必须包含规则号、错误码、按钮名或字段名。
5. 无答案题不得生成虚假证据。
6. 问题中不得直接出现源文档标题。
7. 参考答案内部可以标记 `fact_key`，但不能使用生产引用编号 `C1`。

### 任务 4.3：确定性过滤和去重

**新增文件：** `src/rag_platform/evaluation/dataset_validator.py`

检查：

- 规范化问题是否重复。
- 标准证据是否确实存在于源文档。
- 问题是否直接泄露完整答案。
- 必要证据数量是否符合题型。
- `expected_action` 是否合法。
- 所有源文档编码是否存在。
- 问题长度是否在合理范围。
- 是否存在语义近重复题。

语义去重：

```text
余弦相似度 >= 0.92：直接判为重复，只保留一道
0.85 <= 相似度 < 0.92：交给Qwen判断是否重复
```

### 任务 4.4：使用 Qwen 审核评测题

**新增文件：**

- `scripts/review_eval_cases.py`
- `evaluation/prompts/case_review.txt`

Qwen 输入：

- 问题
- 参考答案
- 预期行为
- 相关源文档
- 候选标准证据

返回：

```json
{
  "answerable": true,
  "expected_action_correct": true,
  "reference_answer_supported": true,
  "evidence_complete": true,
  "ambiguity_score": 0.05,
  "difficulty": "HARD",
  "issues": [],
  "passed": true
}
```

通过条件：

```text
expected_action_correct为true
并且 reference_answer_supported为true
并且 evidence_complete为true
并且 ambiguity_score <= 0.15
```

持续生成替补题，直到正好有 300 道审核通过的题。

### 任务 4.5：把标准证据映射到 Chunk

**新增文件：** `scripts/map_eval_evidence.py`

映射算法：

1. 对证据文本和 Chunk 文本统一空白符。
2. 优先执行严格子串匹配。
3. 未命中时执行标点规范化后的子串匹配。
4. 如果证据跨越相邻 Chunk，则检查相邻 Chunk 拼接文本。
5. 确定性映射失败时，不能让 LLM 静默选择 Chunk。
6. 无法映射时标记为 `MISSING` 或 `AMBIGUOUS`。

只有所有 `relevance_grade=3` 的证据都为 `MAPPED`，题目才能被冻结。

### 任务 4.6：划分并冻结数据集

**新增文件：** `scripts/freeze_eval_dataset.py`

按照文档主题和版本组划分：

- `DEVELOPMENT`：60%
- `VALIDATION`：20%
- `TEST`：20%

写入 `rag_eval_case.dataset_split`。

冻结流程：

1. 按 `case_code` 排序。
2. 导出规范化 JSONL。
3. 计算整个文件的 SHA256。
4. 写入 `rag_eval_dataset.content_sha256`。
5. 状态修改为 `FROZEN`。
6. 冻结数据集禁止原地修改；需要修改时创建新版本。

## 八、阶段 5：实现确定性评测指标

### 任务 5.1：检索指标

**新增文件：** `src/rag_platform/evaluation/retrieval_metrics.py`

函数：

```python
recall_at_k(retrieved_ids, relevant_ids, k)
reciprocal_rank(retrieved_ids, relevant_ids)
ndcg_at_k(retrieved_ids, relevance_by_id, k)
fact_coverage(retrieved_ids, fact_keys_by_chunk)
```

nDCG Gain：

```text
grade 0 -> gain 0
grade 1 -> gain 1
grade 2 -> gain 3
grade 3 -> gain 7
```

**新增测试：** `tests/evaluation/test_retrieval_metrics.py`

至少覆盖：

- 相关结果位于第一名。
- 没有召回相关结果。
- 存在多个相关 Chunk。
- 召回结果中存在重复 Chunk ID。
- 分级相关度 nDCG。
- 无答案题没有 Gold Chunk。

### 任务 5.2：行为和引用指标

**新增文件：**

- `src/rag_platform/evaluation/action_metrics.py`
- `src/rag_platform/evaluation/citation_metrics.py`

行为混淆矩阵：

```text
ANSWER
REFUSE
CLARIFY
ERROR
```

引用指标：

```text
citation_precision = 引用中的Gold Chunk数量 / 全部引用Chunk数量
citation_recall = 引用中的Gold Chunk数量 / 全部Gold Chunk数量
```

无答案题不参与 Retrieval Recall 聚合，但必须参与行为指标。

## 九、阶段 6：实验执行器和 V0 基线

### 任务 6.1：实现实验执行器

**新增文件：** `src/rag_platform/evaluation/experiment_runner.py`

逐题执行：

1. 创建 `rag_eval_case_result`。
2. 构造 `ChatRequestV2`。
3. 调用 `ChatService.chat`。
4. 获取 Workflow 和 Trace 明细。
5. 保存最终有序 Chunk ID。
6. 计算确定性指标。
7. 调用 Qwen Judge。
8. 完成逐题结果。

并发规则：

- 首次运行并发数固定为 1。
- 正确性确认后才允许 `--concurrency 3`。
- 使用 `asyncio.Semaphore` 控制并发。
- 网络错误可以重试，模型返回非法评审结论不能无限重试。

### 任务 6.2：实现 Qwen Answer Judge

**新增文件：**

- `src/rag_platform/evaluation/judge_service.py`
- `evaluation/prompts/answer_judge.txt`

Judge 输入：

- 用户问题
- 预期行为
- 参考答案
- 系统答案
- 召回 Context
- 引用列表

输出：

```json
{
  "faithfulness_score": 0.0,
  "answer_relevance_score": 0.0,
  "completeness_score": 0.0,
  "citation_entailment_score": 0.0,
  "conflict_handling_score": 0.0,
  "refusal_correct": false,
  "clarification_correct": false,
  "passed": false,
  "reasons": {
    "unsupported_claims": [],
    "missing_facts": [],
    "citation_issues": []
  }
}
```

使用 `temperature=0`。如果某项分数距离通过阈值不超过 0.05，则把证据
顺序反转后再评审一次，最终取两次平均值，降低位置偏差。

### 任务 6.3：运行 V0

```bash
.venv/bin/python scripts/run_rag_evaluation.py \
  --dataset rag_eval_ecommerce:v1 \
  --split test \
  --experiment-version V0 \
  --experiment-name baseline-hybrid-rrf-rerank \
  --concurrency 1
```

保存：

- 当前 Git SHA
- 全部检索参数
- 模型名称
- Prompt 版本
- Milvus Collection（向量集合）
- Elasticsearch Index（检索索引）

V0 报告生成前，禁止实施 V1。

## 十、阶段 7：V1 Adaptive Retrieval

### 任务 7.1：扩展 RagState

**修改文件：** `src/rag_platform/domain/rag_state.py`

新增：

```python
retrieval_round: int
max_retrieval_rounds: int
retrieval_attempts: list[dict[str, Any]]
quality_features: dict[str, float]
quality_score: float
quality_level: str
retry_strategy: str | None
final_action: str
```

同时删除当前重复声明的 `context` 和 `citations`。

### 任务 7.2：提取质量特征

**新增文件：** `src/rag_platform/rag/adaptive/quality_features.py`

输入：

- BM25 Hits（关键词召回结果）
- Vector Hits（向量召回结果）
- 融合 Hits
- Rerank Hits（精排结果）
- Query Analysis（查询分析结果）

输出：

```python
@dataclass
class RetrievalQualityFeatures:
    candidate_count: int
    distinct_document_count: int
    channel_overlap_at_10: float
    rerank_top1: float
    rerank_top3_mean: float
    rerank_margin: float
    target_type_coverage: float
```

该文件只负责特征计算，不能硬编码最终路由阈值。

### 任务 7.3：实现质量决策策略

**新增文件：** `src/rag_platform/rag/adaptive/quality_policy.py`

```python
@dataclass
class QualityDecision:
    level: Literal["GOOD", "WEAK", "POOR"]
    score: float
    retry_strategy: str | None
    reasons: list[str]
```

初始规则：

- 没有候选：`POOR + RELAX_FILTER`
- Top1 Rerank 低于下限：`POOR + QUERY_REWRITE`
- 通道重合率或 Top1/Top2 分差较弱：`WEAK + QUERY_REWRITE`
- 其他情况：`GOOD`

配置字段加入 `Settings`：

```text
adaptive_max_rounds=2
adaptive_quality_good_threshold
adaptive_quality_poor_threshold
adaptive_rerank_top1_threshold
adaptive_rerank_margin_threshold
```

### 任务 7.4：实现 Query Rewrite

**新增文件：** `src/rag_platform/rag/adaptive/query_rewriter.py`

输入：

- 原问题
- 上一轮改写问题
- 失败候选的标题和短摘要
- 质量不足原因

输出：

```json
{
  "rewritten_query": "...",
  "expanded_queries": ["..."],
  "removed_filters": [],
  "reason": "..."
}
```

不要把全部失败文档正文传给改写模型，避免模型被错误候选锚定。

### 任务 7.5：实现多轮结果融合

**新增文件：** `src/rag_platform/rag/adaptive/multi_round_fusion.py`

使用加权 RRF：

```text
初始轮权重 = 1.0
Query Rewrite轮权重 = 0.9
放宽过滤轮权重 = 0.8
```

每条结果保留：

- 检索轮次
- Query 变体
- 召回通道
- 原始排名
- 原始分数

### 任务 7.6：修改 LangGraph

**修改文件：** `src/rag_platform/rag/graph/rag_retrieval_graph.py`

目标流程：

```text
analyze_query
-> retrieve
-> merge
-> rerank
-> judge_quality
   -> GOOD: build_context
   -> WEAK或POOR且未达到最大轮次: rewrite_or_relax
   -> 已达到最大轮次: build_context_or_refuse
-> finish
```

质量路由禁止继续只根据文档数量判断。

### 任务 7.7：校准质量阈值

**新增文件：** `scripts/calibrate_retrieval_quality.py`

只在开发集上搜索阈值：

```text
objective =
  Recall@5
  - 0.15 * 额外检索触发率
  - 0.10 * 归一化延迟增幅
```

使用验证集选最终配置，测试集运行前冻结阈值。

**V1 验收：**

- Recall@5 或 MRR 高于 V0。
- 二次检索触发率有明确数据。
- 报告 p95 延迟增幅。
- 没有使用测试集调参。

## 十一、阶段 8：V2 Query Decomposition 和多跳检索

### 任务 8.1：实现问题拆解器

**新增文件：** `src/rag_platform/rag/adaptive/query_decomposer.py`

仅在以下情况触发：

- 问题包含多个业务条件。
- 问题涉及多个实体或规则。
- 问题询问版本冲突。
- 单次检索无法覆盖全部必要事实。

输出：

```json
{
  "requires_decomposition": true,
  "sub_queries": [
    {
      "fact_key": "delivery_status_rule",
      "query": "已发货订单退款条件",
      "target_doc_types": ["RULE"]
    },
    {
      "fact_key": "coupon_refund_rule",
      "query": "使用满减券后退款的优惠券处理规则",
      "target_doc_types": ["RULE"]
    }
  ]
}
```

最多生成四个子问题。

### 任务 8.2：按事实独立检索

在 Graph 中增加：

```text
decompose_query
retrieve_sub_queries
judge_fact_coverage
```

每条召回结果携带对应 `fact_key`。

必要事实覆盖率：

```text
已覆盖必要事实数 / 全部必要事实数
```

达到最大轮次后仍不能覆盖必要事实时，系统必须澄清或拒答，禁止强行生成
综合结论。

### 任务 8.3：处理版本和规则冲突

生成文档和 Chunk 中保留版本、生效日期、失效日期。

冲突决策：

1. 当前生效版本优先于失效版本。
2. 只有文档明确声明优先级时，具体规则才能覆盖一般规则。
3. 无法确定优先级时，答案必须展示冲突并建议人工确认。

**V2 验收：**

- 30 道多跳题的必要事实覆盖率高于 V1。
- 无法解决的冲突题同时引用冲突双方。
- 简单直接题不执行问题拆解，不承担额外延迟。

## 十二、阶段 9：V3 Claim 级证据校验

### 任务 9.1：执行 Trace 与 Grounding DDL

```bash
mysql -u <用户名> -p <数据库名> \
  < src/sql/011_create_rag_adaptive_trace_tables.sql
```

### 任务 9.2：抽取原子 Claim

**新增文件：** `src/rag_platform/rag/grounding/claim_extractor.py`

输出：

```json
{
  "claims": [
    {
      "claim_no": 1,
      "claim_text": "...",
      "claim_type": "RULE",
      "citation_ids": ["C1"]
    }
  ]
}
```

只抽取可以被证据验证的事实、规则、步骤、数值和结论。问候语和纯表达性
文本不作为 Claim。

### 任务 9.3：校验 Claim 与证据

**新增文件：** `src/rag_platform/rag/grounding/evidence_validator.py`

必须使用 Qwen，不使用生成答案的 DeepSeek。

每个 Claim 只传入它引用的证据 Chunk：

```json
{
  "support_status": "SUPPORTED",
  "support_score": 0.94,
  "reason": "...",
  "conflicting_citation_ids": []
}
```

状态：

- `SUPPORTED`
- `PARTIAL`
- `UNSUPPORTED`
- `CONFLICT`

### 任务 9.4：实现 Grounding 决策

**新增文件：** `src/rag_platform/rag/grounding/grounding_service.py`

初始规则：

```text
存在UNSUPPORTED Claim -> CORRECT
存在关键CONFLICT Claim -> CORRECT
support_ratio < 0.8 -> CORRECT
第二次校验仍失败 -> REFUSE
其他情况 -> PASS
```

只允许修正一次，避免模型循环自检。

### 任务 9.5：接入 ChatService

**修改文件：**

- `src/rag_platform/application/chat_service.py`
- `src/rag_platform/schemas/chat_v2.py`

流程：

```text
生成答案
-> 抽取Claim
-> 校验证据
-> 必要时修正一次
-> 再校验一次
-> PASS或REFUSE
```

响应新增：

```python
answer_confidence: float | None
grounding_decision: str | None
claim_evidence: list[dict[str, Any]]
```

Grounding 不通过时禁止继续返回 `SUCCESS`。

**V3 验收：**

- Unsupported Claim 比例低于 V2。
- Citation Entailment 高于 V2。
- 报告修正触发率和修正后通过率。
- 报告新增延迟和费用。

## 十三、阶段 10：Chunking 消融实验

必须等 V0-V3 稳定后再进行。

### 任务 10.1：执行 Chunk 实验字段 DDL

先检查字段是否已经存在，再执行：

```bash
mysql -u <用户名> -p <数据库名> \
  < src/sql/012_add_chunk_experiment_metadata.sql
```

### 任务 10.2：增加切分策略

**新增文件：**

```text
src/rag_platform/rag/chunkers/fixed_chunker.py
src/rag_platform/rag/chunkers/recursive_chunker.py
src/rag_platform/rag/chunkers/semantic_chunker.py
```

当前按文档类型的 Chunker 标记为 `STRUCTURED`。

实验组合：

```text
fixed-500-50
fixed-800-80
recursive-800-80
structured-v1
parent-child-v1
semantic-v1
```

不同实验使用独立的 Milvus Collection 和 Elasticsearch Index：

```text
rag_chunk_vector__structured_v1
rag_chunk_bm25__structured_v1
```

禁止覆盖 V0 基线索引。

同一切分实验必须使用相同检索参数，并且按题型分别统计效果，不能只比较
总体平均值。

结论必须写成：

```text
SOP使用parent-child后，多步骤问题Recall@5提升X个百分点，
但平均Context Token增加Y%。
```

## 十四、阶段 11：检索诊断接口

**新增文件：**

- `src/rag_platform/api/evaluation_admin.py`
- `src/rag_platform/schemas/evaluation.py`

接口：

```text
POST /admin/evaluation/runs
GET  /admin/evaluation/runs/{run_code}
GET  /admin/evaluation/runs/{run_code}/report
GET  /admin/evaluation/traces/{trace_id}
```

Trace 返回：

- 原问题和全部改写问题
- 子问题
- 路由选择和原因
- 每一轮检索
- 每个通道的排名和分数
- Rerank 前后变化
- Context 选中和淘汰原因
- Claim 与引用证据
- 各阶段延迟和费用

## 十五、阶段 12：生成最终对比报告

**新增文件：**

- `src/rag_platform/evaluation/report_builder.py`
- `scripts/compare_rag_runs.py`

报告必须包含：

1. 数据集组成和 SHA256。
2. V0-V3 配置差异。
3. 总体检索指标。
4. 分题型检索指标。
5. 答案和 Grounding 指标。
6. 拒答、回答、澄清混淆矩阵。
7. 延迟和费用。
8. 10 个代表性成功案例。
9. 10 个代表性失败案例。
10. 工程结论和剩余限制。

执行：

```bash
.venv/bin/python scripts/compare_rag_runs.py \
  --runs <v0_run_code> <v1_run_code> <v2_run_code> <v3_run_code> \
  --output evaluation/reports/v0-v3-comparison.md
```

## 十六、最终实施顺序

严格按以下顺序：

1. 测试基础和依赖注入。
2. 执行评测 DDL，创建评测模型和 Repository。
3. 生成、审核并渲染 40 篇源文档。
4. 导入文档并导出 Chunk Catalog。
5. 生成、审核、映射并冻结 300 道评测题。
6. 实现确定性指标和 Qwen Judge。
7. 运行并保存 V0 基线。
8. 实现、校准并评测 V1。
9. 实现并评测 V2。
10. 实现并评测 V3。
11. 进行 Chunking 消融实验。
12. 增加 Trace API 并生成最终报告。

最关键的顺序约束是：**没有 V0 基线就不能开发 V1。**
否则项目只能展示实现了更多功能，无法证明新增复杂度确实改善了 RAG 效果。
