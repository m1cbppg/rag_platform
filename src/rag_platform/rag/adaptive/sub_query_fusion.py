from copy import deepcopy


class SubQueryFusion:
    def __init__(
        self,
        *,
        rank_constant: int = 60,
        min_candidates: int = 1,
    ) -> None:
        self.rank_constant = rank_constant
        self.min_candidates = max(1, min_candidates)

    def fuse(
        self,
        retrieval_tasks: list[dict],
        *,
        top_k: int,
    ) -> list[dict]:
        if top_k < 1:
            return []

        merged: dict[int, dict] = {}
        ranked_by_sub_query: dict[str, list[int]] = {}
        sub_query_order: list[str] = []

        for task in retrieval_tasks:
            sub_query_id = str(task.get("sub_query_id") or "").strip()
            question = str(task.get("question") or "").strip()
            if not sub_query_id:
                continue
            sub_query_order.append(sub_query_id)
            ranked_by_sub_query.setdefault(sub_query_id, [])
            seen_in_task: set[int] = set()

            for rank, document in enumerate(
                task.get("documents") or [],
                start=1,
            ):
                chunk_id = document.get("chunk_id")
                if chunk_id is None:
                    continue
                chunk_id = int(chunk_id)
                contribution = 1.0 / (
                    self.rank_constant + rank
                )
                if chunk_id not in seen_in_task:
                    ranked_by_sub_query[sub_query_id].append(chunk_id)
                    seen_in_task.add(chunk_id)

                if chunk_id not in merged:
                    item = deepcopy(document)
                    metadata = dict(item.get("metadata") or {})
                    metadata["sub_query_ids"] = [sub_query_id]
                    metadata["sub_query_texts"] = [question]
                    metadata["sub_query_sources"] = [
                        {
                            "sub_query_id": sub_query_id,
                            "question": question,
                            "rank": rank,
                            "raw_score": document.get("score"),
                            "rrf_contribution": contribution,
                        }
                    ]
                    metadata["sub_query_rrf_score"] = contribution
                    item["metadata"] = metadata
                    item["score"] = contribution
                    item["source"] = "sub_query_fusion"
                    merged[chunk_id] = item
                    continue

                existing = merged[chunk_id]
                existing["score"] = float(
                    existing.get("score") or 0.0
                ) + contribution
                metadata = existing.setdefault("metadata", {})
                self._append_unique(
                    metadata.setdefault("sub_query_ids", []),
                    sub_query_id,
                )
                self._append_unique(
                    metadata.setdefault("sub_query_texts", []),
                    question,
                )
                metadata.setdefault(
                    "sub_query_sources",
                    [],
                ).append(
                    {
                        "sub_query_id": sub_query_id,
                        "question": question,
                        "rank": rank,
                        "raw_score": document.get("score"),
                        "rrf_contribution": contribution,
                    }
                )
                metadata["sub_query_rrf_score"] = existing["score"]

        selected_ids: list[int] = []
        for sub_query_id in sub_query_order:
            added = 0
            for chunk_id in ranked_by_sub_query.get(
                sub_query_id,
                [],
            ):
                if chunk_id not in selected_ids:
                    selected_ids.append(chunk_id)
                added += 1
                if (
                    added >= self.min_candidates
                    or len(selected_ids) >= top_k
                ):
                    break
            if len(selected_ids) >= top_k:
                break

        ranked_merged = sorted(
            merged.values(),
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )
        for item in ranked_merged:
            chunk_id = int(item["chunk_id"])
            if chunk_id not in selected_ids:
                selected_ids.append(chunk_id)
            if len(selected_ids) >= top_k:
                break

        return [
            merged[chunk_id]
            for chunk_id in selected_ids[:top_k]
            if chunk_id in merged
        ]

    def restore_rerank_quota(
        self,
        *,
        reranked_documents: list[dict],
        candidate_documents: list[dict],
        sub_query_ids: list[str],
        top_n: int,
        quota: int,
    ) -> list[dict]:
        if top_n < 1:
            return []
        quota = max(1, quota)
        reranked = [deepcopy(item) for item in reranked_documents]
        selected = reranked[:top_n]
        selected_ids = {
            int(item["chunk_id"])
            for item in selected
            if item.get("chunk_id") is not None
        }

        for sub_query_id in sub_query_ids:
            while (
                self._coverage_count(selected, sub_query_id)
                < quota
            ):
                candidate = next(
                    (
                        deepcopy(item)
                        for item in candidate_documents
                        if (
                            item.get("chunk_id") is not None
                            and int(item["chunk_id"])
                            not in selected_ids
                            and sub_query_id
                            in self._sub_query_ids(item)
                        )
                    ),
                    None,
                )
                if candidate is None:
                    break
                metadata = dict(candidate.get("metadata") or {})
                metadata["quota_restored"] = True
                candidate["metadata"] = metadata
                candidate_id = int(candidate["chunk_id"])

                if len(selected) < top_n:
                    selected.append(candidate)
                    selected_ids.add(candidate_id)
                    continue

                replace_index = self._replacement_index(
                    selected=selected,
                    sub_query_ids=sub_query_ids,
                    quota=quota,
                )
                if replace_index is None:
                    break
                removed_id = int(
                    selected[replace_index]["chunk_id"]
                )
                selected[replace_index] = candidate
                selected_ids.discard(removed_id)
                selected_ids.add(candidate_id)

        return selected

    def calculate_coverage(
        self,
        *,
        sub_queries: list[dict],
        candidate_documents: list[dict],
        final_documents: list[dict],
    ) -> dict:
        items: dict[str, dict] = {}
        covered = 0
        for sub_query in sub_queries:
            sub_query_id = str(sub_query["sub_query_id"])
            candidate_count = sum(
                sub_query_id in self._sub_query_ids(document)
                for document in candidate_documents
            )
            final_count = sum(
                sub_query_id in self._sub_query_ids(document)
                for document in final_documents
            )
            is_covered = final_count > 0
            covered += int(is_covered)
            items[sub_query_id] = {
                "question": sub_query.get("question") or "",
                "candidate_count": candidate_count,
                "final_count": final_count,
                "covered": is_covered,
            }
        total = len(sub_queries)
        return {
            "total_sub_queries": total,
            "covered_sub_queries": covered,
            "coverage_rate": covered / total if total else 1.0,
            "items": items,
        }

    @staticmethod
    def _sub_query_ids(document: dict) -> list[str]:
        metadata = document.get("metadata") or {}
        return list(metadata.get("sub_query_ids") or [])

    @staticmethod
    def _append_unique(values: list, value) -> None:
        if value not in values:
            values.append(value)

    def _replacement_index(
        self,
        *,
        selected: list[dict],
        sub_query_ids: list[str],
        quota: int,
    ) -> int | None:
        for index in range(len(selected) - 1, -1, -1):
            supported = self._sub_query_ids(selected[index])
            if all(
                self._coverage_count(selected, sub_query_id)
                > quota
                for sub_query_id in supported
                if sub_query_id in sub_query_ids
            ):
                return index
        return None

    def _coverage_count(
        self,
        documents: list[dict],
        sub_query_id: str,
    ) -> int:
        return sum(
            sub_query_id in self._sub_query_ids(document)
            for document in documents
        )
