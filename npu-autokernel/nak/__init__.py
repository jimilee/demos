"""npu-autokernel (nak): vendor-neutral auto-optimization layer for mobile NPU compilers.

Common layers (this package): workload registry, candidate/result schema,
capability manifests, evaluator (correctness contract, measurement policy,
acceptance), experiment store and controller.
Vendor-specific code lives only in ``nak/backends/<backend>.py`` adapter
bundles registered in ``nak.registry``. Common code must not branch on vendor
names (checked by tests/test_no_vendor_branching.py).
"""
SCHEMA_VERSION = "nak-schema-0.1"
