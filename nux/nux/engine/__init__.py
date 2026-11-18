"""Layer 2 - engine: the experiment loop.

registry (backend discovery), config (strict experiment config), measurement (protocols),
evaluator (correctness, admissibility, acceptance), stages (candidate pipeline), controller.
Depends on core, storage and toolchain; reaches backends only through the registry.
"""
