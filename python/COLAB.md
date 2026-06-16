# Running `ccgen` on Google Colab

`ccgen` is pure Python, so the easiest Colab workflow is to install the package
from the `python/` subdirectory and then use either the Python API or the new
`ccgen` CLI.

## Option 1: install from a local wheel you upload to Colab

Build the wheel locally:

```bash
cd python
python -m pip wheel . -w dist
```

That produces a file like:

```text
dist/ccgen-0.1.0-py3-none-any.whl
```

In Colab:

```python
from google.colab import files
uploaded = files.upload()  # upload ccgen-0.1.0-py3-none-any.whl
```

Then install it:

```python
!pip install /content/ccgen-0.1.0-py3-none-any.whl
```

For notebook-friendly extras:

```python
!pip install numpy opt_einsum
```

## Option 2: install directly from a Git repository subdirectory

If the repository is on GitHub, Colab can install straight from `python/`:

```python
!pip install "git+https://github.com/hemanthharidas/planck-refactored.git#subdirectory=python"
```

Or with the Colab extras:

```python
!pip install "git+https://github.com/hemanthharidas/planck-refactored.git#subdirectory=python[colab]"
```

## Quick sanity check

Python API:

```python
from ccgen import generate_cc_equations

eqs = generate_cc_equations("ccsd")
{name: len(terms) for name, terms in eqs.items()}
```

CLI:

```python
!ccgen ccsd --format counts --json
```

## Example notebook usage

```python
from ccgen import print_equations, print_einsum

print(print_equations("ccsd"))
print(print_einsum("ccsd"))
```

## Notes

- The core package has no mandatory third-party runtime dependencies.
- If a compiler toolchain is available, installation will also try to build the
  optional `ccgen._wickaccel` C++ extension automatically.
- `parallel_workers > 1` may fall back to serial execution in restricted
  environments. On Colab VMs, multiprocessing should generally be available.
- The Planck-specific C++ emitter is included, but compiling and running the
  emitted C++ kernels is separate from installing the Python package itself.
