"""Evaluation-owned package for AutoKernel-on-Hexagon.

Everything in this package (accuracy contract, input generation, measurement
windows, adoption rule) is owned by the evaluator. The agent edits only
``kernel.py`` (and ``ops/linear_kernel.py`` for the HMX extension).
"""
