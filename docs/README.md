# Documentation Export

The main teaching document is:

- `docs/PLANCK_TEACHING_GUIDE.md`
- `docs/CCGEN_TEACHING_GUIDE.md` documents the Python coupled-cluster equation generator

## Conventions for this directory

Every file here **answers one architecture question** or is a teaching guide. The sole exception is
a doc scoping work that is still in progress, which may carry the scope and its steps — and which
must be rewritten into an architecture answer as soon as that work lands and is verified.

**Status does not live here.** What is done and what is open are recorded in
`vault/Status/Completion.md` and `vault/Status/Open Work.md`, which are canonical and regenerate
`CLAUDE.md`.

### Answered questions — dressed CC kernels

- `CCGEN_DRESSED_KERNEL_PIPELINE.md` — how does a dressed CC residual become a running C++ kernel?
- `CCGEN_SPIN_ADAPTER_CONTRACT.md` — what does the spin adapter guarantee, and how do you check it?
- `CCGEN_DRESSING_COST.md` — why does dressed-operator recognition cost what it costs?

### Answered questions — unrestricted (open-shell) CC

- `CCGEN_UNRESTRICTED_CC.md` — how does a spin-orbital CC manifold become a runnable
  open-shell kernel set, and what has to stay spin-resolved all the way down?
- `CCGEN_UCC_ERI_ANTISYMMETRY.md` — the equations want `<pq||rs>`, the cache stores `<pq|rs>`;
  where does the exchange get added?
- `CCGEN_UCC_NUMERIC_VALIDATION.md` — how do you check that a spin-block CC residual is right?
- `CCGEN_GCC_TO_UCC_BRIDGE.md` — how does a spin-orbital manifold become a spin-block-resolved
  one, and how does that differ from the spatial collapse?

To export it as a standalone static webpage:

```bash
python3 docs/build_teaching_site.py
```

This writes:

- `docs/site/index.html`

You can also use the CMake helper target:

```bash
cmake --build build --target teaching-site
```

The exporter is intentionally lightweight and repository-local. It does not
require MkDocs, Sphinx, or a JavaScript bundler. The generated page is a single
HTML file with embedded CSS, so it can be hosted from any static file server or
opened directly in a browser.
