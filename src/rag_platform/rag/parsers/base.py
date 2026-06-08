from dataclasses import dataclass
from typing import Any


@dataclass
class ParseResult:
    """
    文档解析结果。

    dataclass 是 Python 的数据类。
    它可以自动生成 __init__ 方法，减少样板代码。

    raw_content：
        从 docx/pdf 中提取出来的原始文本。

    structure：
        结构化结果。
        FAQ 里可能是 qa_pairs；
        SOP 里可能是 title、scene、steps；
        RULE 里可能是 clauses；
        MANUAL 里可能是 title_path、steps、buttons。

    parser_type：
        标记使用了哪个解析器。
    """

    raw_content: str
    structure: dict[str, Any]
    parser_type: str


class BaseDocumentParser:
    """
    文档解析器基类。

    所有具体解析器都要实现 parse 方法。
    """

    def parse(self, file_path: str) -> ParseResult:
        """
        解析文件。

        子类必须重写这个方法。
        """
        raise NotImplementedError