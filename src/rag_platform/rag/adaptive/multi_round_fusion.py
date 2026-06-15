from copy import deepcopy


_STRATEGY_WEIGHTS = {
    "INITIAL": 1.0,
    "QUERY_REWRITE": 0.9,
    "FORCE_BM25": 1.0,
    "RELAX_FILTER": 0.8,
}


class MultiRoundFusion:
    def __init__(self, *, rank_constant: int = 60) -> None:
        self.rank_constant = rank_constant

    def fuse(
        self,
        attempts: list[dict],
        *,
        top_k: int,
    ) -> list[dict]:
        merged: dict[int, dict] = {}
        for attempt in attempts:
            round_no = int(attempt.get("round_no") or 1)
            strategy = str(attempt.get("strategy") or "INITIAL")
            weight = float(
                attempt.get("weight")
                or _STRATEGY_WEIGHTS.get(strategy, 1.0)
            )
            queries = list(attempt.get("queries") or [])
            query_variant = str(
                attempt.get("query_variant") or "ORIGINAL"
            )
            for rank, document in enumerate(
                attempt.get("documents") or [],
                start=1,
            ):
                chunk_id = document.get("chunk_id")
                if chunk_id is None:
                    continue
                chunk_id = int(chunk_id)
                contribution = weight / (
                    self.rank_constant + rank
                )
                source_info = {
                    "round_no": round_no,
                    "strategy": strategy,
                    "query_variant": query_variant,
                    "queries": queries,
                    "rank": rank,
                    "raw_score": document.get("score"),
                    "source": document.get("source"),
                    "rrf_contribution": contribution,
                }
                if chunk_id not in merged:
                    item = deepcopy(document)
                    metadata = dict(item.get("metadata") or {})
                    metadata["retrieval_rounds"] = [round_no]
                    metadata["adaptive_sources"] = [source_info]
                    metadata["adaptive_rrf_score"] = contribution
                    item["metadata"] = metadata
                    item["score"] = contribution
                    item["source"] = "adaptive"
                    merged[chunk_id] = item
                    continue
                existing = merged[chunk_id]
                existing["score"] = float(
                    existing.get("score") or 0.0
                ) + contribution
                metadata = existing.setdefault("metadata", {})
                rounds = metadata.setdefault(
                    "retrieval_rounds",
                    [],
                )
                if round_no not in rounds:
                    rounds.append(round_no)
                metadata.setdefault(
                    "adaptive_sources",
                    [],
                ).append(source_info)
                incoming_metadata = document.get("metadata") or {}
                for key in ("sub_query_ids", "sub_query_texts"):
                    values = metadata.setdefault(key, [])
                    for value in incoming_metadata.get(key) or []:
                        if value not in values:
                            values.append(value)
                metadata.setdefault(
                    "sub_query_sources",
                    [],
                ).extend(
                    incoming_metadata.get("sub_query_sources") or []
                )
                metadata["adaptive_rrf_score"] = existing["score"]

        return sorted(
            merged.values(),
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )[:top_k]
