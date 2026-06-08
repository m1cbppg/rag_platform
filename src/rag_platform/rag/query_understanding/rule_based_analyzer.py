import re

from src.rag_platform.domain.query import QueryIntent, RetrievalMode
from src.rag_platform.schemas.query_analysis import QueryAnalysisResult


class RuleBasedQueryAnalyzer:
    """
    规则型 Query 分析器。

    作用：
    1. 快速判断问题类型；
    2. LLM 失败时兜底；
    3. 对明显关键词做稳定路由。

    注意：
    规则不追求完美，只负责稳定兜底。
    """

    def analyze(
        self,
        query: str,
        business_domain: str | None = None,
    ) -> QueryAnalysisResult:
        """
        分析用户问题。
        """

        normalized_query = query.strip()

        target_doc_types = self._guess_doc_types(normalized_query)
        retrieval_mode = self._guess_retrieval_mode(normalized_query, target_doc_types)
        rewritten_query = self._rewrite_by_rule(normalized_query)
        expanded_queries = self._expand_by_rule(normalized_query, rewritten_query)

        confidence = self._estimate_confidence(
            query=normalized_query,
            target_doc_types=target_doc_types,
            retrieval_mode=retrieval_mode,
        )

        reason = (
            f"规则分析：target_doc_types={target_doc_types}, "
            f"retrieval_mode={retrieval_mode}"
        )

        return QueryAnalysisResult(
            original_query=query,
            rewritten_query=rewritten_query,
            expanded_queries=expanded_queries,
            target_doc_types=target_doc_types,
            retrieval_mode=retrieval_mode,
            business_domain=business_domain,
            confidence=confidence,
            reason=reason,
            need_clarification=False,
            clarification_question=None,
            use_llm=False,
            fallback_used=True,
        )

    def _guess_doc_types(self, query: str) -> list[str]:
        """
        判断应该优先检索哪些文档类型。
        """

        doc_types: list[str] = []

        if self._looks_like_rule_query(query):
            doc_types.append("RULE")

        if self._looks_like_sop_query(query):
            doc_types.append("SOP")

        if self._looks_like_manual_query(query):
            doc_types.append("MANUAL")

        if self._looks_like_faq_query(query):
            doc_types.append("FAQ")

        if not doc_types:
            doc_types = ["FAQ", "SOP", "RULE", "MANUAL"]

        return doc_types

    def _guess_retrieval_mode(
        self,
        query: str,
        target_doc_types: list[str],
    ) -> str:
        """
        判断走 BM25 / Vector / Hybrid。

        精确关键词明显时偏 BM25；
        口语化表达偏 Vector；
        默认 Hybrid。
        """

        if self._contains_exact_signal(query):
            return RetrievalMode.BM25.value

        if self._looks_like_semantic_query(query):
            return RetrievalMode.VECTOR.value

        return RetrievalMode.HYBRID.value

    def _looks_like_rule_query(self, query: str) -> bool:
        keywords = [
            "规则", "能不能", "是否", "可不可以", "支持吗",
            "退款", "扣费", "条件", "限制", "例外", "生效",
            "条款", "第", "违约", "赔付"
        ]

        return any(word in query for word in keywords)

    def _looks_like_sop_query(self, query: str) -> bool:
        keywords = [
            "流程", "排查", "处理流程", "怎么处理", "如何处理",
            "步骤", "先做什么", "转人工", "异常处理"
        ]

        return any(word in query for word in keywords)

    def _looks_like_manual_query(self, query: str) -> bool:
        keywords = [
            "后台", "怎么操作", "在哪里", "入口", "菜单",
            "按钮", "点击", "配置", "新增", "编辑", "删除",
            "补发", "导出"
        ]

        return any(word in query for word in keywords)

    def _looks_like_faq_query(self, query: str) -> bool:
        keywords = [
            "怎么办", "为什么", "什么原因", "常见问题",
            "不刷新", "离线", "看不到", "失败", "异常"
        ]

        return any(word in query for word in keywords)

    def _contains_exact_signal(self, query: str) -> bool:
        """
        是否包含强精确匹配信号。

        例如：
        - 条款编号 2.3.1
        - 错误码 E1002
        - 按钮名【补发权益】
        - 接口路径 /api/order/refund
        """

        patterns = [
            r"\d+\.\d+(\.\d+)*",
            r"[A-Z]\d{3,}",
            r"【.+?】",
            r"/api/[a-zA-Z0-9_/\-]+",
            r"[a-zA-Z_]+_status",
        ]

        return any(re.search(pattern, query) for pattern in patterns)

    def _looks_like_semantic_query(self, query: str) -> bool:
        """
        判断是否偏口语化语义查询。
        """

        semantic_words = [
            "咋", "怎么回事", "一直", "好像", "不太对",
            "看不到", "不动了", "没反应"
        ]

        return any(word in query for word in semantic_words)

    def _rewrite_by_rule(self, query: str) -> str:
        """
        规则改写。

        这里只做轻量归一化。
        真正复杂的改写交给 LLM。
        """

        rewritten = query

        replacements = {
            "车位置": "车辆位置",
            "车在哪": "车辆位置在哪里",
            "不动了": "不刷新",
            "看不到车": "看不到车辆位置",
            "咋": "怎么",
        }

        for old, new in replacements.items():
            rewritten = rewritten.replace(old, new)

        return rewritten

    def _expand_by_rule(
        self,
        query: str,
        rewritten_query: str,
    ) -> list[str]:
        """
        简单 query expansion。

        注意：
        规则扩展不要太多，否则会引入噪声。
        """

        queries = [query]

        if rewritten_query != query:
            queries.append(rewritten_query)

        if "定位" in query or "车辆位置" in rewritten_query:
            queries.extend([
                "车辆定位不刷新怎么办",
                "车辆位置不更新怎么处理",
                "设备离线导致定位不刷新怎么排查",
            ])

        if "退款" in query:
            queries.extend([
                "退款规则",
                "已发货订单退款条件",
                "订单退款例外情况",
            ])

        # 去重，同时保持顺序
        return list(dict.fromkeys(queries))

    def _estimate_confidence(
        self,
        query: str,
        target_doc_types: list[str],
        retrieval_mode: str,
    ) -> float:
        """
        粗略估算规则分析置信度。
        """

        if target_doc_types == ["FAQ", "SOP", "RULE", "MANUAL"]:
            return 0.4

        if self._contains_exact_signal(query):
            return 0.85

        return 0.65