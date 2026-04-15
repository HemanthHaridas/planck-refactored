"""
ccgen -- Coupled-Cluster equation generator.

Symbolic  derivation  of  spin-orbital CC residual equations via BCH expansion,
Wick contraction, and canonicalization.
"""

from .generate import generate_cc_contractions, generate_cc_equations
