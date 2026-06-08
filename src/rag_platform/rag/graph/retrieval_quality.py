class RetrievalQualityJudge:
    """
    召回质量判断器。

    它不负责判断最终答案是否正确，只判断：
    当前候选文档是否足够进入下一阶段。
    """

    def judge(self, documents: list[dict]) -> dict:
        """
        判断召回质量。

        返回结构：
        {
            "quality": "GOOD/WEAK/POOR",
            "document_count": 10,
            "top_score": 0.92,
            "need_rewrite": false,
            "reason": "召回结果充足"
        }
        """

        if not documents:
            return {
                "quality": "POOR",
                "document_count": 0,
                "top_score": 0.0,
                "need_rewrite": True,
                "reason": "没有召回任何文档，需要改写 query 或扩大检索范围",
            }

        scores = [
            float(item.get("score") or 0.0)
            for item in documents
        ]

        top_score = max(scores) if scores else 0.0
        document_count = len(documents)

        if document_count < 2:
            return {
                "quality": "WEAK",
                "document_count": document_count,
                "top_score": top_score,
                "need_rewrite": True,
                "reason": "召回文档数量过少，可能需要二次检索",
            }

        return {
            "quality": "GOOD",
            "document_count": document_count,
            "top_score": top_score,
            "need_rewrite": False,
            "reason": "召回结果数量满足进入后续 Rerank 阶段",
        }