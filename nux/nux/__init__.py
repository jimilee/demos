"""nux (nux): vendor-neutral auto-optimization layer for mobile NPU compilers.

Common layers (this package): workload registry, candidate/result schema,
capability manifests, evaluator (correctness contract, measurement policy,
acceptance), experiment store and controller.
Vendor-specific code lives only in ``nux/backends/<backend>.py`` adapter
bundles registered in ``nux.registry``. Common code must not branch on vendor
names (checked by tests/test_no_vendor_branching.py).
"""
SCHEMA_VERSION = "nux-schema-0.2"  # 0.2: backend_traits, paired measurement {base,cand}, no injected config keys
