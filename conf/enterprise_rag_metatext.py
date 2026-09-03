# Metadata-in-text variant of "enterprise_rag" (arm B): the provenance header is written into every
# chunk and abstract before embedding / BM25 indexing, so the index files get a "_metatext" tag.
# Build with:  python index.py --config enterprise_rag_metatext --force_index_from_scratch true
from conf.enterprise_rag import conf as _base

conf = {
    **_base,
    "enterprise_chunk_metadata_prefix": True,
}
