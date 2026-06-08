import re

from src.rag_platform.domain.document import DocumentType


class DocumentCleaner:
    """
    文档清洗器。

    清洗原则：
    1. 去掉页眉页脚、水印、乱码、过多空行；
    2. 保留条款编号、步骤编号、按钮名、金额、时间、状态条件；
    3. 不同文档类型使用不同清洗策略。
    """

    def clean(self, raw_content: str, doc_type: DocumentType) -> str:
        if not raw_content:
            return ""

        text = raw_content

        text = self._normalize_newline(text)
        text = self._remove_common_noise(text)

        if doc_type in [DocumentType.SOP, DocumentType.MANUAL]:
            text = self._fix_pdf_line_breaks(text)

        return text.strip()

    def _normalize_newline(self, text: str) -> str:
        """
        统一换行符，并压缩过多空行。
        """

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _remove_common_noise(self, text: str) -> str:
        """
        删除常见噪声。

        注意：
        这里只删除明确无意义的内容。
        不删除数字、条款编号、步骤编号。
        """

        result_lines: list[str] = []

        for line in text.splitlines():
            stripped = line.strip()

            if not stripped:
                result_lines.append("")
                continue

            if stripped in ["公司内部资料", "内部资料", "Confidential"]:
                continue

            if re.match(r"^第\s*\d+\s*页\s*/\s*共\s*\d+\s*页$", stripped):
                continue

            result_lines.append(stripped)

        return "\n".join(result_lines)

    def _fix_pdf_line_breaks(self, text: str) -> str:
        """
        修复 PDF 抽取时产生的断行。

        注意：
        不能无脑合并所有换行。
        SOP 和操作手册里的步骤编号必须保留独立行。
        """

        lines = text.splitlines()
        fixed_lines: list[str] = []

        for line in lines:
            if not fixed_lines:
                fixed_lines.append(line)
                continue

            previous = fixed_lines[-1]

            should_merge = (
                previous
                and line
                and not previous.endswith(("。", "；", "：", ":", "！", "？"))
                and not self._looks_like_numbered_line(line)
                and not self._looks_like_numbered_line(previous)
            )

            if should_merge:
                fixed_lines[-1] = previous + line
            else:
                fixed_lines.append(line)

        return "\n".join(fixed_lines)

    def _looks_like_numbered_line(self, line: str) -> bool:
        return bool(
            re.match(r"^(\d+[\.\、]|第[一二三四五六七八九十]+步|第[一二三四五六七八九十]+条)", line.strip())
        )