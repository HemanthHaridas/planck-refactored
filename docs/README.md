# Documentation Export

The main teaching document is:

- `docs/PLANCK_TEACHING_GUIDE.md`
- `docs/CCGEN_TEACHING_GUIDE.md` documents the Python coupled-cluster equation generator

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
