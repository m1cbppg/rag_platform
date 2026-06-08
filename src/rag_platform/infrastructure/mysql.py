from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.rag_platform.core.config import get_settings


def create_mysql_engine() -> Engine:
    """
    创建 MySQL 数据库连接引擎。

    SQLAlchemy 的 Engine 可以理解为数据库连接工厂。
    后续模块会用它执行 SQL、创建 Session、管理事务。
    """

    settings = get_settings()

    database_url = (
        f"mysql+pymysql://{settings.mysql_user}:"
        f"{settings.mysql_password}@"
        f"{settings.mysql_host}:"
        f"{settings.mysql_port}/"
        f"{settings.mysql_database}?charset=utf8mb4"
    )

    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=3600,
    )