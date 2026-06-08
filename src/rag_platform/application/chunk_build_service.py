from uuid import uuid4

from src.rag_platform.domain.chunk import ChunkBuildItem, ChunkRelationType
from src.rag_platform.domain.document import DocumentStatus, DocumentType
from src.rag_platform.infrastructure.repositories.document_repository import DocumentRepository
from src.rag_platform.rag.chunkers.chunker_factory import ChunkerFactory
from src.rag_platform.schemas.chunk import ChunkBuildResponse


class ChunkBuildService:
    """
    chunk 构建服务。

    负责把模块 2 的结构化解析结果转换成 chunk，并保存到 MySQL。
    """

    def __init__(self) -> None:
        self.repository = DocumentRepository()
        self.chunker_factory = ChunkerFactory()

    def build_chunks_for_document(self, doc_id: int) -> ChunkBuildResponse:
        """
        为指定文档构建 chunk。
        """

        document_parse = self.repository.get_cleaned_document_parse(doc_id)

        if document_parse is None:
            return ChunkBuildResponse(
                doc_id=doc_id,
                chunk_count=0,
                relation_count=0,
                status="NOT_READY",
                message="文档不存在，或文档不是 CLEANED 状态，无法切分 chunk",
            )

        doc_type = DocumentType(document_parse["doc_type"])
        structure = document_parse["structure_json"]
        clean_content = document_parse["clean_content"] or ""

        chunker = self.chunker_factory.get_chunker(doc_type)

        chunk_items = chunker.build_chunks(
            structure=structure,
            clean_content=clean_content,
        )

        if not chunk_items:
            return ChunkBuildResponse(
                doc_id=doc_id,
                chunk_count=0,
                relation_count=0,
                status="NO_CHUNK",
                message="未生成任何 chunk，请检查文档解析结果",
            )

        # 幂等：先删除该文档已有 chunk，再重新生成。
        self.repository.delete_chunks_by_doc_id(doc_id)

        temp_key_to_chunk_id: dict[str, int] = {}
        saved_chunk_ids: list[int] = []

        # 第一轮：先保存所有 chunk。
        for item in chunk_items:
            parent_chunk_id = None

            # 如果当前 item 有 parent_temp_key，并且 parent 已经入库，
            # 就可以设置 parent_chunk_id。
            if item.parent_temp_key:
                parent_chunk_id = temp_key_to_chunk_id.get(item.parent_temp_key)

            chunk_id = self._save_chunk(
                item=item,
                doc_id=doc_id,
                parent_chunk_id=parent_chunk_id,
                document_parse=document_parse,
            )

            saved_chunk_ids.append(chunk_id)

            if item.temp_key:
                temp_key_to_chunk_id[item.temp_key] = chunk_id

        relation_count = 0

        # 第二轮：保存 parent-child 关系。
        for item in chunk_items:
            if not item.parent_temp_key or not item.temp_key:
                continue

            parent_id = temp_key_to_chunk_id.get(item.parent_temp_key)
            child_id = temp_key_to_chunk_id.get(item.temp_key)

            if parent_id and child_id:
                self.repository.create_chunk_relation(
                    from_chunk_id=parent_id,
                    to_chunk_id=child_id,
                    relation_type=ChunkRelationType.PARENT_CHILD.value,
                    sort_order=item.sort_order,
                )
                relation_count += 1

        # 第三轮：保存 previous-next 相邻关系。
        for index in range(len(saved_chunk_ids) - 1):
            self.repository.create_chunk_relation(
                from_chunk_id=saved_chunk_ids[index],
                to_chunk_id=saved_chunk_ids[index + 1],
                relation_type=ChunkRelationType.PREVIOUS_NEXT.value,
                sort_order=index + 1,
            )
            relation_count += 1

        self.repository.update_status(doc_id, DocumentStatus.CHUNKED.value)

        return ChunkBuildResponse(
            doc_id=doc_id,
            chunk_count=len(saved_chunk_ids),
            relation_count=relation_count,
            status=DocumentStatus.CHUNKED.value,
            message="chunk 构建完成",
        )

    def _save_chunk(
        self,
        item: ChunkBuildItem,
        doc_id: int,
        parent_chunk_id: int | None,
        document_parse: dict,
    ) -> int:
        """
        保存单个 chunk。
        """

        chunk_code = f"CHK_{uuid4().hex[:16]}"

        token_count = self._estimate_token_count(item.content)

        return self.repository.create_chunk(
            chunk_code=chunk_code,
            doc_id=doc_id,
            parent_chunk_id=parent_chunk_id,
            chunk_type=item.chunk_type.value,
            title=item.title,
            title_path=item.title_path,
            content=item.content,
            summary=item.summary,
            keywords=item.keywords,
            tags=item.tags,
            business_domain=document_parse.get("business_domain"),
            version=document_parse.get("version"),
            source_doc_title=document_parse.get("doc_title"),
            source_page=None,
            source_section=item.source_section,
            token_count=token_count,
            sort_order=item.sort_order,
        )

    def _estimate_token_count(self, text: str) -> int:
        """
        粗略估算 token 数量。

        中文场景下不能简单按英文单词数算。
        这里先用一个粗略估算：
        1 个中文字符大约 1 个 token；
        英文则偏粗略。

        后续如果需要精准，可以接入 tiktoken 或模型对应 tokenizer。
        """

        return len(text)