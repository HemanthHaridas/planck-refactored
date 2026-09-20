#!/usr/bin/env python3
"""U10/U11 live molecular acceptance, never builds or enables production.

Default: all U0 open-shell fixtures, larger-basis water cation, closed-shell
reduction and RKS regression. Preserve every input, log, ordinary JSON and
state sidecar. No PySCF oracle, diagnostic environment flags or NumPy dependency.
Energy derivatives use a fourth-order centered stencil at each requested step;
raw second-order differences are retained separately as diagnostics.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from dh_channel_fd_audit import ANGSTROM_TO_BOHR, parse_geometry, render_gradient_input
from udh_u0_molecular_audit import FIXTURES


def transpose(a):
    return [list(x) for x in zip(*a)]


def matmul(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def maxdiff(a, b):
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        raise ValueError("Matrix shape mismatch")
    return max((abs(x-y) for r, s in zip(a, b) for x, y in zip(r, s)), default=0.)


def eigenvalues_symmetric(a):
    """Small independent Jacobi eigensolver for occupied-subspace singular values."""
    a = [r[:] for r in a]
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError("Non-square Gram matrix")
    for _ in range(max(1, 100*n*n)):
        if n < 2:
            break
        p, q = max(((i, j) for i in range(n) for j in range(i)), key=lambda ij: abs(a[ij[0]][ij[1]]))
        if abs(a[p][q]) < 1e-14:
            break
        angle = .5*math.atan2(2*a[p][q], a[q][q]-a[p][p])
        c, s = math.cos(angle), math.sin(angle)
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        for k in range(n):
            if k not in (p, q):
                akp, akq = a[k][p], a[k][q]
                a[k][p] = a[p][k] = c*akp-s*akq
                a[k][q] = a[q][k] = s*akp+c*akq
        a[p][p] = c*c*app+s*s*aqq-2*c*s*apq
        a[q][q] = s*s*app+c*c*aqq+2*c*s*apq
        a[p][q] = a[q][p] = 0.
    else:
        raise ValueError("Occupied-subspace eigensolve did not converge")
    return [a[i][i] for i in range(n)]


def continuity(center, endpoint, minimum=.99):
    if not endpoint.get("converged") or any(endpoint[k] != center[k] for k in ("nalpha", "nbeta")):
        raise ValueError("SCF convergence or spin population changed")
    if not math.isfinite(endpoint["s2"]) or abs(endpoint["s2"]-center["s2"]) > .01:
        raise ValueError("Spin contamination changed discontinuously")
    cross = endpoint["cross_overlap_reference_current"]
    result = {}
    for spin in ("alpha", "beta"):
        if center["n"+spin] == 0:
            result[spin] = 1.
            continue
        overlap = matmul(matmul(transpose(center["occupied_"+spin]), cross), endpoint["occupied_"+spin])
        sv = math.sqrt(max(0., min(eigenvalues_symmetric(matmul(overlap, transpose(overlap))))))
        if not math.isfinite(sv) or sv < minimum or sv > 1+1e-8:
            raise ValueError(f"{spin} occupied subspace discontinuity: minimum singular value {sv}")
        result[spin] = sv
    return result


def frame(x):
    def unit(v):
        norm = math.sqrt(sum(q*q for q in v))
        if norm < 1e-12:
            raise ValueError("Degenerate molecular frame")
        return [q/norm for q in v]
    e = unit([a-b for a, b in zip(x[1], x[0])])
    candidates = [[a-b for a, b in zip(row, x[0])] for row in x[2:]]
    candidates += [[float(i == j) for i in range(3)] for j in range(3)]
    for v in candidates:
        dot = sum(a*b for a, b in zip(v, e))
        perpendicular = [a-dot*b for a, b in zip(v, e)]
        if sum(a*a for a in perpendicular) > 1e-10:
            f = unit(perpendicular)
            break
    g = [e[1]*f[2]-e[2]*f[1], e[2]*f[0]-e[0]*f[2], e[0]*f[1]-e[1]*f[0]]
    return [e, f, g]


def requested_matrix(state, name):
    standard, requested = state["coordinates_standard"], state["coordinates_requested"]
    if maxdiff(standard, requested) < 1e-12:
        return state[name]
    rotation = matmul(transpose(frame(standard)), frame(requested))
    moved = matmul([[a-b for a, b in zip(row, standard[0])] for row in standard], rotation)
    want = [[a-b for a, b in zip(row, requested[0])] for row in requested]
    if maxdiff(moved, want) > 1e-8:
        raise ValueError("Prepared/requested geometry is not the declared rigid frame")
    return matmul(state[name], rotation)


def require_close(actual, expected, tol, label):
    if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual-expected) > tol:
        raise ValueError(f"{label}: {actual} versus {expected} (limit {tol})")


def bohr_separation(plus, minus, atom, axis):
    """Use the executable's requested-frame coordinates, not a second unit constant."""
    separation = plus["coordinates_requested"][atom][axis] - minus["coordinates_requested"][atom][axis]
    if not math.isfinite(separation) or separation <= 0:
        raise ValueError("Invalid physical Bohr displacement")
    return separation


def with_angular_points(template, points):
    """Change only the DFT angular control; retain radial, SCF and geometry inputs."""
    if points is None:
        return template
    if not isinstance(points, int) or points < 0 or points > 5810 or 0 < points < 6:
        raise ValueError("Invalid angular point count")
    match = re.search(r"%begin_dft\s*\n(.*?)%end_dft", template, re.S)
    if not match:
        raise ValueError("Missing DFT section")
    body = re.sub(r"^\s*grid_angular_points\s+[^\n]*\n", "", match[1], flags=re.M)
    return template[:match.start(1)]+body+f"    grid_angular_points {points}\n"+template[match.end(1):]


def centered_fourth_order(inner, outer, inner_separation, outer_separation):
    """Cancel the O(h^2) error using measured radii; return (fourth, raw second).

    inner/outer are (E_plus,E_minus), each centered at the same geometry.
    For exactly doubled radii this is (4*D(h)-D(2h))/3, the five-point
    first-derivative stencil. No analytic gradient enters the estimator.
    """
    if not all(math.isfinite(v) for v in (*inner, *outer, inner_separation, outer_separation)):
        raise ValueError("Nonfinite finite-difference input")
    if not 0 < inner_separation < outer_separation:
        raise ValueError("Finite-difference radii must be positive and increasing")
    second = (inner[0]-inner[1])/inner_separation
    wide = (outer[0]-outer[1])/outer_separation
    ratio2 = (outer_separation/inner_separation)**2
    return (ratio2*second-wide)/(ratio2-1), second


class Audit:
    def __init__(self, executable, production, root, steps):
        self.executable, self.production, self.root, self.steps = executable, production, root, steps
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("PLANCK_DFT_DH_")}
        self.env.setdefault("OMP_NUM_THREADS", "2")

    def run(self, tag, template, symbols, xyz, charge, mult, gradient=False, scale=None, reference=None):
        rendered = render_gradient_input(template, symbols, xyz, charge, mult)
        if not gradient:
            rendered = re.sub(r"(calculation\s+)gradient", r"\g<1>energy", rendered, count=1)
        inp, js = self.root/(tag+".hfinp"), self.root/(tag+".json")
        inp.write_text(rendered)
        command = [str(self.executable), str(inp), "--json", str(js)]
        if scale is not None:
            command += ["--scale", str(scale)]
        if reference is not None:
            command += ["--reference", str(reference)]
        with (self.root/(tag+".log")).open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=self.env)
        if process.returncode:
            raise ValueError(f"Calculation failed: {self.root/(tag+'.log')}")
        data, state = json.loads(js.read_text()), json.loads(Path(str(js)+".state.json").read_text())
        angular = re.search(r"^\s*grid_angular_points\s+(\d+)", rendered, re.M)
        if angular and state.get("grid_angular_points") != int(angular[1]):
            raise ValueError("Requested angular grid was not reported by the executable")
        if not state.get("converged") or not all(math.isfinite(data[k]) for k in ("total_energy", "scf_total_energy")):
            raise ValueError("Unconverged or nonfinite result")
        if gradient and (len(data.get("gradient", [])) != len(symbols) or any(
                len(row) != 3 or any(not math.isfinite(v) for v in row) for row in data["gradient"])):
            raise ValueError("Invalid gradient shape/value")
        return data, state, parse_geometry(rendered)[1], inp

    def molecule(self, name, template):
        symbols, xyz, charge, mult = parse_geometry(template)
        center, state, xyz, inp = self.run(name+"-center", template, symbols, xyz, charge, mult, True)
        if not state.get("assembled"):
            raise ValueError("UKS correction was not assembled")
        require_close(state["scaled_z_residual"], 0, 1e-10, "scaled Z residual")
        ks, pt2 = requested_matrix(state, "ks_standard"), requested_matrix(state, "correction_standard")
        require_close(maxdiff(center["gradient"], [[a+b for a, b in zip(r, s)] for r, s in zip(ks, pt2)]),
                      0, 1e-10, "KS+PT2 requested-frame assembly")
        scaling = {}
        for scale in (0., 2*state["scale"]):
            data, sweep, _, _ = self.run(name+f"-scale-{scale}", template, symbols, xyz, charge, mult, True, scale)
            wanted = [[a+scale/state["scale"]*b for a, b in zip(r, s)] for r, s in zip(ks, pt2)]
            error = maxdiff(data["gradient"], wanted)
            require_close(error, 0, 2e-9, "correction scaling")
            require_close(data["total_energy"], center["scf_total_energy"]+scale/state["scale"]*state["pt2_energy"],
                          1e-10, "energy scaling")
            scaling[str(scale)] = error
        if state["nbeta"] == 0:
            require_close(maxdiff(pt2, [[0., 0., 0.] for _ in symbols]), 0, 1e-13, "zero-beta PT2 gradient")
            require_close(state["pt2_energy"], 0, 1e-13, "zero-beta PT2 energy")
        rows = []
        endpoint_cache = {}
        for step in self.steps:
            for atom in range(len(symbols)):
                for axis in range(3):
                    pairs = []
                    for radius in (step, 2*step):
                        key = (atom, axis, radius)
                        if key not in endpoint_cache:
                            endpoints = []
                            for label, sign in (("plus", 1), ("minus", -1)):
                                moved = [row[:] for row in xyz]
                                moved[atom][axis] += sign*radius/ANGSTROM_TO_BOHR
                                data, spin_state, _, _ = self.run(
                                    f"{name}-a{atom}-q{axis}-h{radius}-{label}", template, symbols, moved, charge, mult,
                                    reference=inp)
                                subspaces = continuity(state, spin_state)
                                endpoints.append((data, spin_state, subspaces))
                            p, m = endpoints
                            # Rounded inputs must still describe a centered,
                            # one-coordinate displacement in the requested frame.
                            for a in range(len(symbols)):
                                for q in range(3):
                                    reference_q = state["coordinates_requested"][a][q]
                                    pq, mq = p[1]["coordinates_requested"][a][q], m[1]["coordinates_requested"][a][q]
                                    require_close(.5*(pq+mq), reference_q, 2e-10, "FD midpoint")
                                    if (a, q) != (atom, axis):
                                        require_close(pq, reference_q, 1e-12, "FD unchanged coordinate")
                                        require_close(mq, reference_q, 1e-12, "FD unchanged coordinate")
                            separation = bohr_separation(p[1], m[1], atom, axis)
                            endpoint_cache[key] = (p, m, separation)
                        pairs.append(endpoint_cache[key])
                    p, m, separation = pairs[0]
                    op, om, outer_separation = pairs[1]
                    total_fd, raw_total = centered_fourth_order(
                        (p[0]["total_energy"], m[0]["total_energy"]),
                        (op[0]["total_energy"], om[0]["total_energy"]), separation, outer_separation)
                    ks_fd, raw_ks = centered_fourth_order(
                        (p[0]["scf_total_energy"], m[0]["scf_total_energy"]),
                        (op[0]["scf_total_energy"], om[0]["scf_total_energy"]), separation, outer_separation)
                    row = dict(atom=atom, axis=axis, step=step, actual_half_step=separation/2,
                               actual_outer_half_step=outer_separation/2,
                               total_error=total_fd-center["gradient"][atom][axis],
                               ks_error=ks_fd-ks[atom][axis], pt2_error=total_fd-ks_fd-pt2[atom][axis],
                               raw_second_total_error=raw_total-center["gradient"][atom][axis],
                               raw_second_ks_error=raw_ks-ks[atom][axis],
                               raw_second_pt2_error=raw_total-raw_ks-pt2[atom][axis],
                               occupied_subspaces=[p[2], m[2]], outer_occupied_subspaces=[op[2], om[2]])
                    rows.append(row)
                    print(name, json.dumps(row), flush=True)
        translation = max(abs(sum(row[q] for row in center["gradient"])) for q in range(3))
        require_close(translation, 0, 5e-8, "gradient translation sum")
        # Actual translated/rotated inputs, preserving all SCF settings.
        shifted = [[v+t for v, t in zip(row, [.213, -.317, .149])] for row in xyz]
        translated, _, _, _ = self.run(name+"-translated", template, symbols, shifted, charge, mult, True)
        translation_error = maxdiff(translated["gradient"], center["gradient"])
        require_close(translation_error, 0, 5e-8, "translated gradient")
        require_close(translated["total_energy"], center["total_energy"], 1e-9, "translated energy")
        rotation = [[.8, -.6, 0.], [.48, .64, -.6], [.36, .48, .8]]
        transformed = matmul(xyz, rotation)
        transformed = [[v+t for v, t in zip(row, [.213, -.317, .149])] for row in transformed]
        moved, _, _, _ = self.run(name+"-rigid", template, symbols, transformed, charge, mult, True)
        covariance = maxdiff(moved["gradient"], matmul(center["gradient"], rotation))
        # Finite atom-centered angular quadrature is not perfectly rotation invariant.
        # Keep this separately labelled bound; do not relax the 5e-8 FD criterion.
        require_close(covariance, 0, 2e-6, "finite-grid rigid covariance")
        require_close(moved["total_energy"], center["total_energy"], 2e-6, "finite-grid rotated energy")
        errors = {key: max(abs(r[key]) for r in rows) for key in ("total_error", "ks_error", "pt2_error")}
        passed = all(v <= 5e-8 for v in errors.values())
        raw_errors = {key: max(abs(r[key]) for r in rows) for key in
                      ("raw_second_total_error", "raw_second_ks_error", "raw_second_pt2_error")}
        report = dict(name=name, passed=passed, errors=errors, rows=rows, scaling=scaling,
                      grid_angular_points=state.get("grid_angular_points", 0),
                      fd_stencil="fourth-order centered; actual Bohr radii", tolerance=5e-8,
                      raw_second_errors=raw_errors,
                      translation=translation, translation_error=translation_error, covariance=covariance,
                      rotation_energy_error=moved["total_energy"]-center["total_energy"], s2=state["s2"],
                      nalpha=state["nalpha"], nbeta=state["nbeta"], scaled_z_residual=state["scaled_z_residual"])
        (self.root/(name+"-results.json")).write_text(json.dumps(report, indent=2)+"\n")
        print("MOLECULE", json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
        return report

    def closed_shell(self, template):
        symbols, xyz, _, _ = parse_geometry(template)
        u, _, _, _ = self.run("closed-uks", template, symbols, xyz, 0, 1, True)
        rtext = re.sub(r"(scf_type\s+)uhf", r"\g<1>rhf", template)
        r, _, _, _ = self.run("closed-rks", rtext, symbols, xyz, 0, 1, True)
        require_close(u["total_energy"], r["total_energy"], 1e-9, "closed-shell UKS/RKS energy")
        require_close(maxdiff(u["gradient"], r["gradient"]), 0, 5e-8, "closed-shell UKS/RKS gradient")
        return dict(energy_error=u["total_energy"]-r["total_energy"], gradient_error=maxdiff(u["gradient"], r["gradient"]))

    def rejection(self, template):
        symbols, xyz, charge, mult = parse_geometry(template)
        gradient = render_gradient_input(template, symbols, xyz, charge, mult)
        cases = {"production": (gradient, self.production, "single-point energies"),
                 "spherical": (gradient.replace("basis_type  cartesian", "basis_type  spherical"), self.executable, "Spherical basis"),
                 "roks": (re.sub(r"(scf_type\s+)uhf", r"\g<1>rohf", gradient), self.executable, "ROKS/ROHF"),
                 "optimization": (gradient.replace("calculation gradient", "calculation geomopt"), self.executable, "single-point energies"),
                 "frequency": (gradient.replace("calculation gradient", "calculation frequency"), self.executable, "single-point energies")}
        for name, setting in (("frozen", "mp2_frozen 1"), ("ri", "mp2_use_ri .true."),
                              ("missing-t2", "mp2_with_t2 .false."), ("pt2-shift", "mp2_level_shift 0.3")):
            cases[name] = (gradient.replace("%end_scf", setting+"\n%end_scf"), self.executable,
                           "all-active conventional unshifted PT2")
        # The common PCM workflow guard runs before the UDH-specific guard.
        # Require its exact scope rejection, not a later unreachable message.
        cases["solvent"] = (gradient+"\n%begin_pcm\nmodel pcm\n%end_pcm\n", self.executable,
                            "PCM is currently implemented only for single-point energy calculations")
        results = []
        for name, (text, binary, message) in cases.items():
            inp = self.root/("reject-"+name+".hfinp"); inp.write_text(text)
            proc = subprocess.run([str(binary), str(inp), "--json", str(inp)+".json"], env=self.env, text=True, capture_output=True)
            output = proc.stdout+proc.stderr
            (self.root/("reject-"+name+".log")).write_text(output)
            if proc.returncode == 0 or message not in output:
                raise ValueError("Did not reject expected scope: "+name)
            results.append(name)
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build/planck-udh-validation"))
    parser.add_argument("--production", type=Path, default=Path("build/planck-dft"))
    parser.add_argument("--fixture", help="One U0 filename/stem (partial run, never full U11 acceptance)")
    parser.add_argument("--steps", type=float, nargs=2, default=[1e-4, 2e-4])
    parser.add_argument("--angular-points", type=int, help="Unpruned angular override; radial preset and SCF limits unchanged")
    args = parser.parse_args()
    if len(set(args.steps)) != 2 or any(not math.isfinite(x) or x <= 0 for x in args.steps):
        parser.error("Two distinct positive finite FD steps required")
    paths = [p for p in sorted(FIXTURES.glob("*.hfinp")) if not args.fixture or args.fixture in (p.name, p.stem)]
    if not paths:
        parser.error("No matching fixture")
    root = Path(tempfile.mkdtemp(prefix="udh-u11-", dir="/private/tmp"))
    print("Artifacts:", root, flush=True)
    audit = Audit(args.executable.resolve(), args.production.resolve(), root, args.steps)
    report = dict(scope="test-only molecular U11; no production promotion", complete=not bool(args.fixture), results=[], passed=False)
    try:
        for path in paths:
            report["results"].append(audit.molecule(path.stem, with_angular_points(path.read_text(), args.angular_points)))
        if not args.fixture:
            water = with_angular_points((FIXTURES/"water_cation_asymmetric_b2plyp_sto3g.hfinp").read_text(), args.angular_points)
            larger = re.sub(r"(basis\s+)sto-3g", r"\g<1>6-31g", water, count=1)
            report["results"].append(audit.molecule("water-cation-6-31g", larger))
            report["closed_shell"] = audit.closed_shell(water)
            report["rejections"] = audit.rejection(water)
            # Preserve current production RKS all-coordinate thresholds.
            fixture_dir = FIXTURES.parent
            for name in ("water_b2plyp_gradient_fd.hfinp", "h2o2_c1_b2plyp_gradient_fd.hfinp"):
                path = fixture_dir/name
                if not path.exists():
                    raise ValueError("Missing RKS regression fixture: "+str(path))
                import sys
                command = [sys.executable, str(Path(__file__).with_name("dh_cartesian_fd_audit.py")), str(path),
                           "--executable", str(args.production.resolve()), "--steps", *map(str, args.steps)]
                with (root/("rks-"+name+".log")).open("w") as log:
                    p = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=audit.env)
                if p.returncode:
                    raise ValueError("RKS full-coordinate regression failed: "+name)
        report["passed"] = all(r["passed"] for r in report["results"])
    except (ValueError, OSError, KeyError) as error:
        report["error"] = str(error)
    (root/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    print(("PASS" if report["passed"] else "FAIL") + (" full suite" if report["complete"] else " partial suite"), flush=True)
    if "error" in report:
        print(report["error"], flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
