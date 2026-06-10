import re

from docx import Document

from src.rag_platform.rag.parsers.base import BaseDocumentParser, ParseResult


class RuleDocxParser(BaseDocumentParser):
    """
    业务规则 docx 解析器。

    业务规则的核心是条款。
    后续模块 3 会按照条款切 chunk。
    """

    CLAUSE_PATTERN = re.compile(
        r"^(\d+(\.\d+)*|第[一二三四五六七八九十]+条)[\.、\s]*(.+)",
        re.DOTALL,
    )

    def parse(self, file_path: str) -> ParseResult:
        doc = Document(file_path)

        lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        clauses: list[dict] = []
        current_title = ""

        for line in lines:
            if self._looks_like_title(line):
                current_title = line

            match = self.CLAUSE_PATTERN.match(line)

            if match:
                clauses.append({
                    "clause_no": match.group(1),
                    "title_path": current_title,
                    "content": match.group(3).strip(),
                    "raw_line": line,
                })

        raw_content = "\n".join(lines)

        return ParseResult(
            raw_content=raw_content,
            structure={
                "doc_type": "RULE",
                "title": lines[0] if lines else "",
                "clauses": clauses,
                "raw_lines": lines,
            },
            parser_type="RULE_DOCX",
        )

    def _looks_like_title(self, line: str) -> bool:
        """
        简单判断一行是否像标题。

        这个规则后续可以增强：
        例如读取 docx 的 heading 样式。
        """

        return (
            line.startswith(("一、", "二、", "三、", "四、"))
            or line.endswith(("规则", "说明", "规范", "流程"))
        )
