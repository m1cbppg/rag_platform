import logging
import sys


def setup_logging() -> None:
    """
    初始化项目日志配置。

    logging.basicConfig 用来设置日志输出格式。
    这里将日志输出到控制台，后续可以扩展到文件、ELK、OpenTelemetry。
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
    )