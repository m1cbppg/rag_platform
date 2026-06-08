import logging

from fastapi import APIRouter, File, Form, UploadFile

from src.rag_platform.application.document_ingest_service import DocumentIngestService
from src.rag_platform.domain.document import DocumentType
from src.rag_platform.schemas.document import DocumentUploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

document_ingest_service = DocumentIngestService()


@router.post("/upload", response_model=DocumentUploadResponse)
def upload_document(
    file: UploadFile = File(..., description="上传的文档文件"),
    title: str = Form(..., description="文档标题"),
    doc_type: DocumentType = Form(..., description="文档类型：FAQ/SOP/RULE/MANUAL"),
    business_domain: str | None = Form(default=None, description="业务域"),
    version: str | None = Form(default=None, description="版本"),
    created_by: str | None = Form(default="system", description="上传人"),
) -> DocumentUploadResponse:
    """
    上传并解析文档。

    当前支持：
    1. FAQ：docx
    2. SOP：pdf
    3. 业务规则：docx
    4. 操作手册：pdf
    """

    logger.info(
        "documents.upload.request_received filename=%s title=%s doc_type=%s "
        "business_domain=%s version=%s created_by=%s",
        file.filename,
        title,
        doc_type.value,
        business_domain,
        version,
        created_by,
    )

    try:
        response = document_ingest_service.ingest_upload_file(
            file=file,
            title=title,
            doc_type=doc_type,
            business_domain=business_domain,
            version=version,
            created_by=created_by,
        )
    except Exception:
        logger.exception(
            "documents.upload.request_failed filename=%s title=%s doc_type=%s",
            file.filename,
            title,
            doc_type.value,
        )
        raise

    logger.info(
        "documents.upload.request_completed doc_id=%s status=%s",
        response.doc_id,
        response.status,
    )
    return response
