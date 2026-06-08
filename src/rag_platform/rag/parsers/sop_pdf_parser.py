import re

from pypdf import PdfReader

from src.rag_platform.rag.parsers.base import BaseDocumentParser, ParseResult


class SopPdfParser(BaseDocumentParser):
    """
    SOP PDF 解析器。

    SOP 重点是：
    1. 流程标题；
    2. 适用场景；
    3. 处理步骤；
    4. 注意事项。
    """

    def parse(self, file_path: str) -> ParseResult:
        raw_content = self._read_pdf(file_path)
        lines = self._to_clean_lines(raw_content)

        title = lines[0] if lines else ""

        scenes: list[str] = []
        steps: list[str] = []
        notes: list[str] = []

        for line in lines:
            if self._is_scene_line(line):
                scenes.append(line)

            elif self._is_step_line(line):
                steps.append(line)

            elif self._is_note_line(line):
                notes.append(line)

        return ParseResult(
            raw_content=raw_content,
            structure={
                "doc_type": "SOP",
                "title": title,
                "scene": "\n".join(scenes),
                "steps": steps,
                "notes": notes,
                "raw_lines": lines,
            },
            parser_type="SOP_PDF",
        )

    def _read_pdf(self, file_path: str) -> str:
        """
        使用 pypdf 读取 PDF 文本。

        PdfReader(file_path) 会打开 PDF。
        reader.pages 是 PDF 的页列表。
        page.extract_text() 尝试提取当前页文本。
        """

        reader = PdfReader(file_path)
        page_texts: list[str] = []

        for page in reader.pages:
            text = page.extract_text() or ""
            page_texts.append(text)

        return "\n".join(page_texts)

    def _to_clean_lines(self, text: str) -> list[str]:
        """
        把大段文本拆成干净的行。

        splitlines()：
            按换行符切分字符串。

        strip()：
            去掉行首行尾空格。
        """

        return [line.strip() for line in text.splitlines() if line.strip()]

    def _is_scene_line(self, line: str) -> bool:
        return line.startswith(("适用场景", "场景", "适用范围"))

    def _is_step_line(self, line: str) -> bool:
        """
        判断是否是步骤行。

        支持：
        1. xxx
        1、xxx
        第一步 xxx
        """

        return bool(
            re.match(r"^(\d+[\.\、]|第[一二三四五六七八九十]+步)", line)
        )

    def _is_note_line(self, line: str) -> bool:
        return line.startswith(("注意", "备注", "说明"))