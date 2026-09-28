# Signing key material

`planck_signing_public.pem` is the Ed25519 **public** key used to verify
license file signatures at startup. It is safe to commit and distribute —
it cannot be used to create new licenses, only to check ones already signed
with the matching private key.

The private key is never stored in this repository. It lives outside the
project tree, encrypted at rest, and is only used offline to sign new
license files.

Never compiled. The embedded copy the loader actually uses is
`src/lookup/loader/signing_key.cpp`, regenerated from this file by
`generate_embedded_key.py` whenever the key rotates.
