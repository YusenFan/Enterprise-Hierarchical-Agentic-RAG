# Offline metadata-in-text smoke index (arm B on the smoke pipeline).
from conf.enterprise_rag_smoke import conf as _base

conf = {
    **_base,
    "enterprise_chunk_metadata_prefix": True,
}
