import logging
import shutil
from pathlib import Path
from uuid import uuid4
import hashlib

from fastapi import UploadFile

from src.rag_platform.core.config import get_settings
from src.rag_platform.domain.document import DocumentStatus, DocumentType
from src.rag_platform.infrastructure.repositories.document_repository import DocumentRepository
from src.rag_platform.rag.cleaners.document_cleaner import DocumentCleaner
from src.rag_platform.rag.parsers.parser_factory import DocumentParserFactory
from src.rag_platform.rag.quality.document_quality_checker import DocumentQualityChecker
from src.rag_platform.schemas.document import DocumentUploadResponse

logger = logging.getLogger(__name__)


class DocumentIngestService:
    """
    文档入库应用服务。

    Application Service 的职责：
    编排一次完整业务流程。

    它不负责：
    1. 具体 SQL 怎么写；
    2. PDF/docx 怎么解析；
    3. 文本怎么清洗。

    这些分别交给 Repository、Parser、Cleaner。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.repository = DocumentRepository()
        self.parser_factory = DocumentParserFactory()
        self.cleaner = DocumentCleaner()
        self.quality_checker = DocumentQualityChecker()

    def ingest_upload_file(
            self,
            file: UploadFile,
            title: str,
            doc_type: DocumentType,
            business_domain: str | None,
            version: str | None,
            created_by: str | None,
    ) -> DocumentUploadResponse:
        """
        上传并处理文档。

        增加幂等控制：
        同一个文件内容重复上传时，不重复解析，不重复入库。
        """

        file_sha256 = self._calculate_upload_file_sha256(file)

        existed_document = self.repository.find_by_file_sha256(file_sha256)

        if existed_document is not None:
            return DocumentUploadResponse(
                doc_id=existed_document["id"],
                title=existed_document["title"],
                doc_type=existed_document["doc_type"],
                status=existed_document["status"],
                message="文件已存在，本次未重复解析入库",
            )

        saved_file_path = self._save_upload_file(file)

        doc_code = f"DOC_{uuid4().hex[:16]}"
        file_ext = self._get_file_ext(file.filename or "")

        doc_id = self.repository.create_document(
            doc_code=doc_code,
            title=title,
            doc_type=doc_type.value,
            file_name=file.filename or "",
            file_path=str(saved_file_path),
            file_ext=file_ext,
            file_sha256=file_sha256,
            business_domain=business_domain,
            version=version,
            created_by=created_by,
        )

        try:
            self.repository.update_status(doc_id, DocumentStatus.PARSING.value)

            parser = self.parser_factory.get_parser(doc_type)
            parse_result = parser.parse(str(saved_file_path))
            clean_content = self.cleaner.clean(
                raw_content=parse_result.raw_content,
                doc_type=doc_type,
            )

            quality_results = self.quality_checker.check(
                doc_type=doc_type,
                clean_content=clean_content,
                structure=parse_result.structure,
            )

            has_fail = any(item["check_result"] == "FAIL" for item in quality_results)

            parse_status = "NEED_REVIEW" if has_fail else "SUCCESS"
            final_status = DocumentStatus.NEED_REVIEW if has_fail else DocumentStatus.CLEANED

            self.repository.save_parse_result(
                doc_id=doc_id,
                parser_type=parse_result.parser_type,
                raw_content=parse_result.raw_content,
                clean_content=clean_content,
                structure=parse_result.structure,
                parse_status=parse_status,
            )

            self.repository.save_quality_results(doc_id, quality_results)
            self.repository.update_status(doc_id, final_status.value)

            return DocumentUploadResponse(
                doc_id=doc_id,
                title=title,
                doc_type=doc_type.value,
                status=final_status.value,
                message="文档解析完成",
            )

        except Exception as exc:
            self.repository.save_parse_result(
                doc_id=doc_id,
                parser_type="UNKNOWN",
                raw_content="",
                clean_content="",
                structure={},
                parse_status="FAILED",
                error_message=str(exc),
            )
            self.repository.update_status(doc_id, DocumentStatus.FAILED.value)

            raise

    def ingest_file_path(
        self,
        file_path: Path,
        title: str,
        doc_type: DocumentType,
        business_domain: str | None,
        version: str | None,
        created_by: str | None,
    ) -> DocumentUploadResponse:
        """
        从本地文件路径导入文档。

        评测语料已经由渲染脚本写入本地，不需要再经过 HTTP 上传。
        这里复用与接口上传完全相同的解析、清洗和质量检查流程。
        """

        with file_path.open("rb") as file_object:
            upload_file = UploadFile(
                file=file_object,
                filename=file_path.name,
            )
            return self.ingest_upload_file(
                file=upload_file,
                title=title,
                doc_type=doc_type,
                business_domain=business_domain,
                version=version,
                created_by=created_by,
            )

    def _save_upload_file(self, file: UploadFile) -> Path:
        """
        保存上传文件到本地目录。

        Path 是 pathlib 提供的路径对象，比字符串拼路径更安全。
        """

        upload_dir = Path(self.settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        original_name = file.filename or "unknown"
        file_ext = self._get_file_ext(original_name)

        saved_name = f"{uuid4().hex}.{file_ext}"
        saved_path = upload_dir / saved_name

        logger.info(
            "documents.ingest.save_file.start original_name=%s file_ext=%s saved_path=%s",
            original_name,
            file_ext,
            saved_path,
        )
        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(
            "documents.ingest.save_file.completed original_name=%s saved_path=%s",
            original_name,
            saved_path,
        )

        return saved_path

    def _get_file_ext(self, filename: str) -> str:
        """
        获取文件扩展名。

        例如：
        test.docx -> docx
        sop.pdf -> pdf
        """

        suffix = Path(filename).suffix

        if not suffix:
            return ""

        return suffix.replace(".", "").lower()

    def _calculate_upload_file_sha256(self, file: UploadFile) -> str:
        """
        计算上传文件的 SHA256。

        注意：
        UploadFile.file 是一个文件流。
        读取之后，文件指针会移动到末尾。
        所以计算完 hash 后，必须 file.file.seek(0)，把指针重置回开头。
        否则后面保存文件时会保存成空文件。
        """

        sha256 = hashlib.sha256()

        while True:
            chunk = file.file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

        file.file.seek(0)

        return sha256.hexdigest()
