#!/usr/bin/env python3
"""Gates for the statistical-assertion machinery (FCIQMC scope, G1-G2).

Run: python3 tests/statistical_gate.py

These test the GATE, not any physics. A statistical gate that cannot fail is
worse than no gate at all, so every check here asserts a failure as well as a
pass -- see docs/FCIQMC_RESEARCH_SCOPE.md.
"""
import importlib.util
import math
import random
import sys
from pathlib import Path

# Load run_regressions.py by path, with bytecode caching disabled. A same-second
# file restore (e.g. `cp` during a mutation test) can leave a stale .pyc that
# Python happily reuses, which makes this gate report on code that is no longer
# on disk -- in EITHER direction. Measured: it silently ran a mutated module.
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location(
    "run_regressions", Path(__file__).resolve().parent / "run_regressions.py"
)
_rr = importlib.util.module_from_spec(_spec)
sys.modules["run_regressions"] = _rr          # dataclass needs the module registered
_spec.loader.exec_module(_rr)
within_sigma_failure = _rr.within_sigma_failure

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blocking  # noqa: E402
import mc_estimator  # noqa: E402
import reproducibility  # noqa: E402


def check(cond, msg):
    if not cond:
        print(f"  [FAIL] {msg}")
        return False
    return True


def test_g1_within_sigma():
    """G1: the check passes at 3 sigma and FAILS at 0.1 sigma."""
    ok = True
    ref, sigma = -107.6529998854, 1.0e-4
    value = ref + 2.0 * sigma                 # deliberately 2 sigma out

    r = within_sigma_failure(value, sigma, ref, 3.0, "e", "e_sigma")
    ok &= check(r is None, f"2-sigma deviation should pass at 3 sigma, got: {r}")

    r = within_sigma_failure(value, sigma, ref, 0.1, "e", "e_sigma")
    ok &= check(r is not None, "2-sigma deviation must FAIL at 0.1 sigma")

    # Exactly on the boundary passes (<=), just outside does not.
    r = within_sigma_failure(ref + 3.0 * sigma, sigma, ref, 3.0, "e", "e_sigma")
    ok &= check(r is None, "exactly 3 sigma should pass")
    r = within_sigma_failure(ref + 3.0001 * sigma, sigma, ref, 3.0, "e", "e_sigma")
    ok &= check(r is not None, "just past 3 sigma must fail")

    # A run that reports sigma = 0 must not pass by dividing the gate away.
    r = within_sigma_failure(value, 0.0, ref, 3.0, "e", "e_sigma")
    ok &= check(r is not None, "sigma=0 with a nonzero deviation must fail")
    r = within_sigma_failure(ref, 0.0, ref, 3.0, "e", "e_sigma")
    ok &= check(r is None, "sigma=0 with an exact value is consistent")

    # Missing / malformed inputs are failures, never silent passes.
    ok &= check(within_sigma_failure(None, sigma, ref, 3.0, "e", "s") is not None,
                "missing value must fail")
    ok &= check(within_sigma_failure(value, None, ref, 3.0, "e", "s") is not None,
                "missing sigma must fail")
    ok &= check(within_sigma_failure(value, -1.0, ref, 3.0, "e", "s") is not None,
                "negative sigma must fail")
    nan = float("nan")
    ok &= check(within_sigma_failure(nan, sigma, ref, 3.0, "e", "s") is not None,
                "NaN value must fail")
    ok &= check(within_sigma_failure(value, nan, ref, 3.0, "e", "s") is not None,
                "NaN sigma must fail")
    ok &= check(within_sigma_failure(value, float("inf"), ref, 3.0, "e", "s") is not None,
                "infinite sigma must fail (it would pass everything)")
    return ok


def ar1_series(n, phi, seed, sigma_noise=1.0):
    """AR(1): x_t = phi*x_{t-1} + eps. Analytic tau_int = 0.5*(1+phi)/(1-phi).

    Burn in for 20 correlation times so the series starts stationary.
    """
    rng = random.Random(seed)
    x = 0.0
    burn = int(20.0 / max(1e-9, 1.0 - phi))
    for _ in range(burn):
        x = phi * x + rng.gauss(0.0, sigma_noise)
    out = []
    for _ in range(n):
        x = phi * x + rng.gauss(0.0, sigma_noise)
        out.append(x)
    return out


def test_g2_blocking():
    """G2: blocked sigma recovers the analytic value; the naive one visibly does not."""
    ok = True

    # --- correlated input: the naive error must be WRONG and blocking must fix it
    for phi, tol in ((0.8, 0.35), (0.9, 0.40)):
        tau_exact = 0.5 * (1.0 + phi) / (1.0 - phi)
        ratios = []
        for seed in range(8):                      # average over seeds: single-seed noise is large
            s = ar1_series(200_000, phi, seed)
            ratios.append(blocking.integrated_autocorrelation_time(s) / tau_exact)
        got = sum(ratios) / len(ratios)
        ok &= check(abs(got - 1.0) <= tol,
                    f"phi={phi}: blocked tau_int/tau_exact = {got:.3f}, want 1 +/- {tol}")

        # The contrast IS the test: naive must understate by ~sqrt(2*tau).
        s = ar1_series(200_000, phi, 0)
        naive = blocking.naive_standard_error(s)
        blocked = blocking.blocked_standard_error(s)
        expected_ratio = math.sqrt(2.0 * tau_exact)
        ok &= check(blocked > naive * 1.5,
                    f"phi={phi}: blocked ({blocked:.4f}) must exceed naive ({naive:.4f})")
        ok &= check(abs((blocked / naive) / expected_ratio - 1.0) <= 0.35,
                    f"phi={phi}: blocked/naive = {blocked/naive:.2f}, "
                    f"want ~{expected_ratio:.2f}")

    # --- independent input: blocking must NOT inflate the error bar
    ratios = []
    for seed in range(8):
        s = ar1_series(100_000, 0.0, 100 + seed)   # phi=0 is i.i.d.
        ratios.append(blocking.blocked_standard_error(s) / blocking.naive_standard_error(s))
    got = sum(ratios) / len(ratios)
    ok &= check(0.95 <= got <= 1.45,
                f"i.i.d.: blocked/naive = {got:.3f}, want ~1 (blocking must not inflate)")

    # --- the mean is preserved by blocking (a transformation error would move it)
    s = ar1_series(4096, 0.8, 7)
    curve = blocking.blocking_curve(s)
    ok &= check(len(curve) >= 8, f"blocking curve too short: {len(curve)}")
    ok &= check(all(nb >= 4 for _, _, nb in curve), "curve must stop above 4 blocks")

    # --- degenerate input is NaN, not a silently tiny sigma
    ok &= check(math.isnan(blocking.blocked_standard_error([1.0])),
                "a 1-sample series must give NaN, not 0")
    return ok


def test_g3_reproducibility():
    """G3: same seed -> bitwise-identical trajectory, and it must be able to FAIL."""
    ok = True
    pop = mc_estimator.make_population(2000)

    # Proven first on a DETERMINISTIC producer, before anything stochastic uses it.
    r, detail = reproducibility.check_reproducible(lambda: list(pop), n_runs=3)
    ok &= check(r, f"a constant producer must be reproducible: {detail}")

    # A seeded stochastic producer.
    r, detail = reproducibility.check_reproducible(
        lambda: mc_estimator.sample_trajectory(pop, 4000, seed=42), n_runs=3)
    ok &= check(r, f"fixed-seed trajectory must reproduce bitwise: {detail}")

    # NEGATIVE CONTROL 1: an unseeded producer must be caught.
    bad = random.Random()
    r, _ = reproducibility.check_reproducible(
        lambda: [bad.random() for _ in range(500)], n_runs=3)
    ok &= check(not r, "an UNSEEDED producer must FAIL the reproducibility gate")

    # NEGATIVE CONTROL 2: a producer that ignores its seed must be caught.
    r, _ = reproducibility.check_seed_sensitivity(
        lambda seed: mc_estimator.sample_trajectory(pop, 2000, seed=seed), 1, 2)
    ok &= check(r, "different seeds must give different trajectories")
    r, _ = reproducibility.check_seed_sensitivity(lambda seed: list(pop), 1, 2)
    ok &= check(not r, "a seed-ignoring producer must FAIL seed sensitivity")

    # The digest must be bit-sensitive, not print-precision-sensitive.
    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0 + 4.4e-16]           # ~1 ulp
    ok &= check(reproducibility.trajectory_digest(a) != reproducibility.trajectory_digest(b),
                "digest must distinguish a 1-ulp difference")
    return ok


def test_g4_end_to_end():
    """G4: a trivial stochastic estimator through the whole pipeline."""
    ok = True
    pop = mc_estimator.make_population(5000)
    exact = mc_estimator.exact_mean(pop)

    # --- the mean lands within 3 sigma of the exact answer, using the BLOCKED sigma
    outside = 0
    trials = 12
    for seed in range(trials):
        traj = mc_estimator.sample_trajectory(pop, 20_000, seed)
        mean = sum(traj) / len(traj)
        sigma = blocking.blocked_standard_error(traj)
        if within_sigma_failure(mean, sigma, exact, 3.0, "mc_mean", "mc_sigma") is not None:
            outside += 1
    ok &= check(outside <= 1, f"{outside}/{trials} trials outside 3 sigma (expect ~0)")

    # --- and the gate is NOT vacuous: the naive sigma understates badly enough
    #     that the same trials fail against it. If this passes, the gate would
    #     accept an under-reported error bar, which is the whole risk.
    naive_outside = 0
    for seed in range(trials):
        traj = mc_estimator.sample_trajectory(pop, 20_000, seed)
        mean = sum(traj) / len(traj)
        naive = blocking.naive_standard_error(traj)
        if within_sigma_failure(mean, naive, exact, 3.0, "mc_mean", "mc_sigma") is not None:
            naive_outside += 1
    ok &= check(naive_outside >= 3,
                f"only {naive_outside}/{trials} failed with the NAIVE sigma -- "
                "the fixture is not correlated enough to test anything")

    # --- sigma shrinks as 1/sqrt(N). The SLOPE is what catches a biased sampler.
    xs, ys = [], []
    for n in (4_000, 16_000, 64_000, 256_000):
        sigmas = [blocking.blocked_standard_error(mc_estimator.sample_trajectory(pop, n, s))
                  for s in range(6)]
        xs.append(math.log(n))
        ys.append(math.log(sum(sigmas) / len(sigmas)))
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    ok &= check(abs(slope + 0.5) <= 0.1, f"sigma ~ N^{slope:.3f}, want N^-0.5")

    # --- the error bar is calibrated, not merely large: |deviation| ~ sigma
    devs, sigs = [], []
    for seed in range(10):
        traj = mc_estimator.sample_trajectory(pop, 20_000, 500 + seed)
        devs.append(abs(sum(traj) / len(traj) - exact))
        sigs.append(blocking.blocked_standard_error(traj))
    rms = math.sqrt(sum(d * d for d in devs) / len(devs))
    ratio = rms / (sum(sigs) / len(sigs))
    ok &= check(0.4 <= ratio <= 2.0,
                f"rms(deviation)/mean(sigma) = {ratio:.2f}; sigma is not calibrated")
    return ok


if __name__ == "__main__":
    print("G1 -- metric_within_sigma")
    results = [test_g1_within_sigma()]
    print("G2 -- blocking analysis")
    results.append(test_g2_blocking())
    print("G3 -- fixed-seed reproducibility")
    results.append(test_g3_reproducibility())
    print("G4 -- stochastic estimator, end to end")
    results.append(test_g4_end_to_end())
    if all(results):
        print("\nAll statistical-gate checks passed.")
        sys.exit(0)
    print("\nFAILURES above.")
    sys.exit(1)
