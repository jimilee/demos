"""Layer 1 - toolchain: shared, target-neutral compiler building blocks used by backends.

mlir_host (low-level MLIR tools, host JIT), frontend (workload -> verified linalg IR), transforms (IR recipes),
aot (LLVM IR -> llc objects, ctypes loading), profilers (reusable profilers), onnx_export (vendor SDK input).
Depends only on core.
"""
