#ifndef PLANCK_TESTS_REQUIRE_BOYS_LIBRARY_H
#define PLANCK_TESTS_REQUIRE_BOYS_LIBRARY_H

// Shared license-gate init for standalone test binaries that link the Boys
// function through the loader (src/lookup/loader/). Every such test's main() must call this before
// touching HartreeFock::Lookup::boys()/boys_vec() — same requirement as
// hartree-fock/planck-dft's main(). Prints the failure reason and returns
// false on any gate failure (missing/invalid license, unreachable blob host,
// bad decrypt); the caller should return 1 from main() in that case.
//
// Deliberately NOT put behind a build option or test-only bypass: a test
// binary that skips this would leave an ungated Boys function in the tree.
bool require_boys_library();

#endif // !PLANCK_TESTS_REQUIRE_BOYS_LIBRARY_H
