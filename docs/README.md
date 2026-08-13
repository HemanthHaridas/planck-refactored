# Documentation Export

The main teaching document is:

- `docs/PLANCK_TEACHING_GUIDE.md`
- `docs/CCGEN_TEACHING_GUIDE.md` documents the Python coupled-cluster equation generator

## ccgen status records

The `CCGEN_*_SCOPE.md` files are per-effort design history and are **not** reliable statements of
current state — several describe work that has since landed or been overturned. Read the
consolidated record first:

- `docs/CCGEN_DRESSED_KERNEL_COMPLETION.md` — dressed CC kernels (V1.0–V1.4, D0–D2). **Complete:**
  they generate from the build, compile, link, and run reproducing the undressed energy and
  iteration count, pinned by `dressed_kernel_equivalence_rccsdt`.

Live scope documents, i.e. work not yet finished:

- `docs/CCGEN_ARBITRARY_ORDER_UCC_SCOPE.md` + `docs/CCGEN_U1_UCC_ADAPT_SCOPE.md` — arbitrary-order
  UCC; U0 landed, U1 scoped (U1.0-U1.5), U2-U5 ahead.

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
