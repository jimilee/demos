"""Layer 1 - storage: persistence.

store (immutable experiment records, per-backend best pointers) and artifact_store
(content-addressed artifacts with retention policy). Depends only on core.
"""
