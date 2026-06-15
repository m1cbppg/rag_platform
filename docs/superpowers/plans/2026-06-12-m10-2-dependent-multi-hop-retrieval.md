# M10.2 顺序依赖多跳检索实施计划

> **实施方式：** 根据用户要求，本阶段不采用测试驱动。先完成最小功能实现，
> 再补必要回归测试和全量验证。

**目标：** 在现有 M9 两轮检索和 M10.1 查询分解基础上，实现有证据约束的固定两跳顺序依赖检索。

**架构：** `QueryDecomposer` 生成 `SQ1 + SQ2模板`，第一跳检索完成后由
`IntermediateFactExtractor` 从候选证据抽取中间事实，LangGraph 填充第二跳
查询并复用现有检索、RRF、Rerank 和 Context 链路。

**技术栈：** Python 3.12、Pydantic、LangGraph、DeepSeek、Qwen3-Rerank、pytest。

---

### 任务1：扩展查询分解契约

- [x] 扩展子问题依赖字段。
- [x] 更新分解 Prompt，要求 `DEPENDENT` 返回两跳模板。
- [x] 开启依赖型分解配置。

### 任务2：实现中间事实抽取

- [x] 新增抽取 Prompt。
- [x] 新增 `IntermediateFactExtractor`。
- [x] 校验支持 Chunk、证据原文和置信度。
- [x] 实现确定性失败结果。

### 任务3：接入 LangGraph

- [x] 新增 `prepare_dependent_hop` 节点。
- [x] 第一跳完成后优先进入依赖跳准备。
- [x] 填充第二跳模板并增加检索轮次。
- [x] 复用多轮 RRF 融合和统一 Rerank。
- [x] 抽取失败时使用原问题执行保守第二跳。

### 任务4：状态、响应和评测快照

- [x] 扩展 `RagState`。
- [x] 扩展 Workflow Response。
- [x] 在 Evaluation Adapter 中保留依赖跳诊断信息。
- [x] 保存新增配置和 Prompt 版本。

### 任务5：验证

- [x] 增加中间事实抽取测试。
- [x] 增加两跳 Graph 回归测试。
- [x] 验证简单问题和并行分解不退化。
- [x] 运行全量 pytest。
- [x] 运行 compileall 和 `git diff --check`。
