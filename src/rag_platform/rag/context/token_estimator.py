class TokenEstimator:
    """
    粗略 token 估算器。

    当前为了工程简单，先用字符数粗估：
    - 中文：约 1 字符 ≈ 1 token；
    - 英文：粗略按字符数 / 4 估算。

    后续可以替换成具体模型 tokenizer。
    """

    def estimate(self, text: str) -> int:
        if not text:
            return 0

        chinese_chars = 0
        other_chars = 0

        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                chinese_chars += 1
            else:
                other_chars += 1

        return chinese_chars + max(1, other_chars // 4)