# What does a Planck binary need before it will run?

Every Planck executable needs a valid license file, and — on a cold cache,
see [Caching](#caching) — network access at startup. That covers
`hartree-fock`, `planck-dft`, `planck-mpi` and every test binary built from
this tree. If the license is missing or invalid, the binary exits before it
reads its input. This page explains what it checks, where the license and
cache go, and what to do when the check fails.

## Why the check exists

Planck computes the Boys function with a tuned implementation that is not
part of this repository. The compiled library is distributed separately,
in encrypted form. A license contains the key that decrypts it. The source
code here is readable and buildable without a license, but running a
binary requires one.

The repository itself is licensed under PolyForm Noncommercial 1.0.0 (see
`LICENSE`). A runtime license is a separate thing: it lets you run the
binaries, and it does not change the terms under which you can use the
source.

## Getting a license

Licenses are issued by hand, one per person or group. To ask for one,
contact the maintainer listed in `CITATION.cff`, say what you intend to
use Planck for, and give the machines or cluster you will run it on. A
license is a single `license.json` file. Some licenses have an expiry
date; others don't.

Treat `license.json` like a credential. Don't commit it or pass it on to
anyone else. The repository's `.gitignore` already ignores it at any depth.

## Where the license goes

The binaries look in exactly one place:

```
<install prefix>/license.json
```

This is next to `bin/` and `share/` under the prefix given to
`-DCMAKE_INSTALL_PREFIX` at configure time. The prefix is compiled into the
binary.

| Configured with | License goes at |
|---|---|
| no `-DCMAKE_INSTALL_PREFIX` (the default) | `<source tree>/install/license.json` |
| `-DCMAKE_INSTALL_PREFIX=/opt/planck` | `/opt/planck/license.json` |

No environment variable overrides this location; this is deliberate. If
you run a binary straight from the build tree without installing it, it
still looks under the configured prefix.

Separately, basis sets are found under `<prefix>/share/basis-sets`, and
setting `BASIS_PATH` overrides that. The two are independent: setting
`BASIS_PATH` has no effect on where the license is looked for.

## What happens at startup

1. The license is read and its signature is checked against a public key
   built into the binary. A missing file, an edited file, or an expired
   license stops the run here.
2. The encrypted Boys-function library matching the license's
   `library_version` and your platform is obtained — from the local cache if
   a fresh-enough entry exists (see [Caching](#caching) below), otherwise
   downloaded over HTTPS using `curl`.
3. The library is decrypted with the key in the license, and loaded
   directly from memory. It is never written anywhere you'd find it later:
   Linux keeps it in an anonymous in-memory file, and macOS uses a
   temporary file that is deleted immediately.
4. The calculation starts.

This happens once per process start. A geometry optimization or frequency
run fetches once, not once per step.

## Caching

The encrypted library blob (step 2 above) is cached at
`<install prefix>/boys_cache/`, a sibling of `license.json`. This is public
ciphertext — the same bytes anyone can already download without a
license — so caching it doesn't weaken anything; only decrypting it needs
your license's key.

A cache entry is used as long as it's no more than a week old. Past that,
the next run re-fetches and refreshes it, even if your license itself is
still valid for much longer — this keeps some periodic network contact
happening rather than letting a long-lived license mean "never touches the
network again." If a cached entry fails to decrypt (a corrupted or
truncated file on disk), it's deleted and fetched again automatically,
once, before the run gives up.

There's no environment variable or flag for this — the cache always lives
at the fixed path above, the same non-overridable pattern as
`license.json` itself. If you need to clear it, delete the directory; it's
recreated on the next fetch.

Practical effect: after a first successful run, machines without reliable
outbound network access can keep working for up to a week (or until the
license itself would have expired anyway, whichever is sooner) as long as
the cache directory persists between runs.

### Under MPI

`planck-mpi` checks the license on every rank. If any rank's license is
missing or invalid, the whole job exits, cleanly and without hanging. Only
rank 0 downloads the library, and it passes the bytes to the other ranks
over MPI. Each rank then decrypts and loads its own copy. Every node in
the job therefore needs `license.json` at the configured prefix, but only
the node running rank 0 needs outbound network access.

## Requirements

- **Network access at startup, on a cold cache.** The binary has to be able
  to reach the library host over HTTPS the first time, and again roughly
  weekly thereafter (see [Caching](#caching)). Air-gapped machines and
  sandboxes with no outbound network can only run Planck by first warming
  the cache elsewhere and copying `<prefix>/boys_cache/` over.
- **`curl`** on your `PATH`.
- **OpenSSL 3** (`libcrypto`), found at configure time. On macOS, the
  Homebrew `openssl@3` works.
- **A supported platform:** `macos-arm64` or `linux-x86_64`. Building on
  any other OS/architecture combination fails at compile time with a
  clear error, rather than at runtime.

## Troubleshooting

Every failure is reported as a single `License Error` line naming what
went wrong, and the binary exits with a non-zero status.

| Message mentions | Likely cause | Fix |
|---|---|---|
| the license path / cannot open | No `license.json` at `<prefix>/license.json` | Put it there. Check which prefix the build was configured with in `CMakeCache.txt` (`CMAKE_INSTALL_PREFIX`) |
| signature, license JSON, base64 | The file was edited, truncated, or is not a Planck license | Use the file exactly as you received it |
| expired | The license's expiry date has passed | Ask for a renewal |
| fetch / curl | No network, a proxy blocking the host, or `curl` missing | Check outbound HTTPS from that machine (from rank 0's node under MPI) |
| decrypt | The license doesn't match this library version, or a fetched/cached blob was corrupted | A corrupted cache entry is retried automatically once (fetched fresh, cache entry replaced). If it still fails after that, the license and the library version don't match, so contact the maintainer |
| resolve `planck_boys` | The library loaded but has an unexpected layout | Report it; this should not happen with a matching license |

When you report a problem, include the full `License Error` line, the
platform, and `library_version` from your `license.json`. Don't send the
whole file: it contains the decryption key.

## For developers

- The test binaries go through the same check, so `ctest` and
  `tests/run_regressions.py` need the license at the configured prefix, and
  network access on a cold cache, just like a production run. There is
  deliberately no build option that skips the check.
- The loader is in `src/lookup/loader/`. `boys()` and `boys_vec()` in
  `HartreeFock::Lookup` are thin calls into the loaded library. Calling
  them before startup has loaded it is a programming error, and fails an
  `assert` rather than returning wrong integrals.
- `src/lookup/keys/` holds the public verification key.
  `src/lookup/loader/signing_key.cpp` is generated from it by
  `generate_embedded_key.py`; don't edit it by hand.
- `src/lookup/loader/cache.{h,cpp}` implement the blob cache mechanics
  (read/write/evict) as pure functions over a directory; `orchestrator.cpp`
  owns the policy (when a cached entry is fresh enough, the decrypt-failure
  retry). `cache_path.h.in`/`cache_path.h` follow the same
  `configure_file`-generated, gitignored pattern as `license_path.h`.
