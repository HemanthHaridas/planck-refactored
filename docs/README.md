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

### In-progress scopes

- `CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md` + `CCGEN_U1_UCC_ADAPT_SCOPE.md` — arbitrary-order UCC;
  U0 landed, U1 scoped (U1.0–U1.5), U2–U5 ahead.

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
