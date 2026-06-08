import re


class CitationValidator:
    """
    引用校验器。

    作用：
    1. 检查答案是否包含合法 citation；
    2. 防止模型编造不存在的 [C99]；
    3. 判断是否需要提醒答案引用不足。
    """

    CITATION_PATTERN = re.compile(r"\[(C\d+)\]")

    def validate(
        self,
        answer: str,
        citations: list[dict],
        require_citation: bool,
    ) -> dict:
        valid_ids = {
            item.get("citation_id")
            for item in citations
            if item.get("citation_id")
        }

        used_ids = set(self.CITATION_PATTERN.findall(answer or ""))

        unknown_ids = used_ids - valid_ids

        has_citation = len(used_ids) > 0

        passed = True

        if require_citation and not has_citation:
            passed = False

        if unknown_ids:
            passed = False

        return {
            "passed": passed,
            "has_citation": has_citation,
            "used_citation_ids": sorted(list(used_ids)),
            "unknown_citation_ids": sorted(list(unknown_ids)),
            "valid_citation_ids": sorted(list(valid_ids)),
        }