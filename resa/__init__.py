"""ReSA reconfigurable systolic array model, built on top of SCALE-Sim v3.

This is a reimplementation of Lee & Yeh, "ReSA: Reconfigurable Systolic Array
for Multiple Tiny DNN Tensors", ACM TACO 21(3), 2024. Numbers produced here are
based on our reimplementation and are not claimed to match the paper; the
modelling decisions that deviate from it are listed in docs/resa_spec.md
section 8.

Nothing in this package modifies scalesim/. The only module allowed to touch
scalesim.compute.* is resa.subarray_group.
"""
