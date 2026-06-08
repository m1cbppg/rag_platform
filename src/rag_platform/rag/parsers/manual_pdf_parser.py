import re

from pypdf import PdfReader

from src.rag_platform.rag.parsers.base import BaseDocumentParser, ParseResult


class ManualPdfParser(BaseDocumentParser):
    """
    操作手册 PDF 解析器。

    操作手册的核心不是普通语义，而是：
    1. 去哪个页面；
    2. 点哪个按钮；
    3. 按什么步骤操作。
    """

    BUTTON_PATTERN = re.compile(r"[【\[](.+?)[】\]]")

    def parse(self, file_path: str) -> ParseResult:
        raw_content = self._read_pdf(file_path)
        lines = [line.strip() for line in raw_content.splitlines() if line.strip()]

        titles: list[str] = []
        steps: list[str] = []
        buttons: list[str] = []

        for line in lines:
            if self._looks_like_title(line):
                titles.append(line)

            if self._looks_like_step(line):
                steps.append(line)

            buttons.extend(self.BUTTON_PATTERN.findall(line))

        return ParseResult(
            raw_content=raw_content,
            structure={
                "doc_type": "MANUAL",
                "title": lines[0] if lines else "",
                "title_path": " > ".join(titles[-3:]),
                "steps": steps,
                "button_names": list(dict.fromkeys(buttons)),
                "raw_lines": lines,
            },
            parser_type="MANUAL_PDF",
        )

    def _read_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        texts: list[str] = []

        for page in reader.pages:
            texts.append(page.extract_text() or "")

        return "\n".join(texts)

    def _looks_like_title(self, line: str) -> bool:
        return (
            line.startswith(("一、", "二、", "三、", "四、"))
            or line.endswith(("操作", "说明", "配置", "管理"))
        )

    def _looks_like_step(self, line: str) -> bool:
        return bool(
            re.match(r"^(\d+[\.\、]|第[一二三四五六七八九十]+步)", line)
        )