"""RMP2 analytic-gradient debugging toolkit.

Consolidates the ad-hoc scripts that were used to track down the RMP2 gradient
bug (see ``RMP2_GRADIENT_FIX_SUMMARY.md``) into one reusable module.

Layout
------
``_runtime``   PySCF runtime bootstrap + repo paths (re-exec into the vendored venv).
``planck``     Run the ``hartree-fock`` binary with ``PLANCK_DEBUG_*`` env vars and
               parse the emitted matrix / term dumps.
``reference``  Canonical PySCF reproduction of every gradient intermediate
               (``part_dm2`` -> ``dm2buf`` -> ``Imat`` -> ``Xvo`` -> ``z`` ->
               relaxed density -> per-atom gradient terms).
``compare``    ``summarize_diff`` helper and the ``python -m ... .compare`` CLI
               with one subcommand per pipeline stage.

Invoke through the thin wrapper next to this package::

    python tests/benchmarks/mp2/pyscf_reference/rmp2_grad_debug.py all

The wrapper re-execs into the vendored PySCF venv when needed, so it works from
any interpreter.
"""

from ._runtime import REPO_ROOT, ensure_pyscf_runtime  # noqa: F401
