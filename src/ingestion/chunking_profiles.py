"""Explicit chunking profiles for reproducible corpus construction."""

from src.ingestion.chunk_models import (
    ChunkingConfig,
    OverlapPolicy,
    OversizedTextPolicy,
    SizeUnit,
    TableSplitPolicy,
)


# This is the reproducible baseline for the currently frozen 18-source
# corpus. Rejecting an oversized block is intentional fail-closed behavior,
# not a substitute for the approved future sentence/table splitters.
BASELINE_CHUNKING_CONFIG = ChunkingConfig(
    schema_version="1.0",
    algorithm_version="deterministic-grouping-baseline-v1",
    target_size=4_000,
    hard_max_size=6_000,
    size_unit=SizeUnit.UNICODE_CHARACTERS,
    overlap_policy=OverlapPolicy.NONE,
    table_split_policy=(
        TableSplitPolicy.RETAIN_WHOLE_OR_ROW_GROUPS_WITH_REPEATED_HEADERS
    ),
    oversized_text_policy=OversizedTextPolicy.REJECT,
    include_hierarchy_in_text=True,
)


__all__ = ["BASELINE_CHUNKING_CONFIG"]
