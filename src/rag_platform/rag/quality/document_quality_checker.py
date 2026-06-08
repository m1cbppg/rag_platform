from src.rag_platform.domain.document import DocumentType


class DocumentQualityChecker:
    """
    文档质量校验器。

    质量校验不是为了追求完美，而是为了防止明显错误的数据进入后续 chunk 切分。
    """

    def check(
        self,
        doc_type: DocumentType,
        clean_content: str,
        structure: dict,
    ) -> list[dict]:
        results: list[dict] = []

        if not clean_content.strip():
            results.append(self._fail("CONTENT_EMPTY", "清洗后正文为空"))

        if doc_type == DocumentType.FAQ:
            qa_pairs = structure.get("qa_pairs", [])

            if not qa_pairs:
                results.append(self._fail("FAQ_NO_QA", "FAQ 未解析到问答对"))

        elif doc_type == DocumentType.SOP:
            steps = structure.get("steps", [])

            if not steps:
                results.append(self._warn("SOP_NO_STEPS", "SOP 未识别到步骤"))

        elif doc_type == DocumentType.RULE:
            clauses = structure.get("clauses", [])

            if not clauses:
                results.append(self._warn("RULE_NO_CLAUSES", "业务规则未识别到条款编号"))

        elif doc_type == DocumentType.MANUAL:
            steps = structure.get("steps", [])

            if not steps:
                results.append(self._warn("MANUAL_NO_STEPS", "操作手册未识别到操作步骤"))

        if not results:
            results.append(self._pass("BASIC_CHECK", "基础质量校验通过"))

        return results

    def _pass(self, item: str, message: str) -> dict:
        return {
            "check_item": item,
            "check_result": "PASS",
            "message": message,
        }

    def _warn(self, item: str, message: str) -> dict:
        return {
            "check_item": item,
            "check_result": "WARN",
            "message": message,
        }

    def _fail(self, item: str, message: str) -> dict:
        return {
            "check_item": item,
            "check_result": "FAIL",
            "message": message,
        }