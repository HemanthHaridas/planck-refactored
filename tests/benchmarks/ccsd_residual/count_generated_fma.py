"""Multiply-adds per CCSD iteration in the generated rank-2 RCC kernels
(docs/CCSD_T_SIZE_SCOPE.md, R0).

Usage: count_generated_fma.py <build>/generated/cc/ccsd_arbitrary_planck_generated.cpp <no> <nv> [nests.tsv]

For every accumulation statement, the product of the bounds of its enclosing
for-loops. Loops are `for (int x = 0; x < BOUND; ++x)`; a run of such lines
before a `{` is a nest's outer loops, for-lines inside it are inner (summed)
loops of the next statement.
"""
import re
import sys
from collections import defaultdict

path, no, nv = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
bound = {"no": no, "nv": nv}
# `j = i` is R1's half range: with the i loop, B(B+1)/2 pairs, so (B+1)/2 here.
FOR = re.compile(r"^\s*for \(int (\w+) = (\w+); \1 < (\w+); \+\+\1\)\s*$")
FUNC = re.compile(r"^\S.*\b(build_W_\w+|compute_ccsd_\w+_residual)\(")

per_func = defaultdict(int)
per_nest = []  # (func, nest index, fma, loop bounds)
func = None
nest = -1
outer, inner, in_body = [], [], False
for line in open(path):
    m = FUNC.match(line)
    if m:
        func, nest = m.group(1), -1
        outer, inner, in_body = [], [], False
        continue
    if func is None:
        continue
    if line.startswith("}"):
        func = None
        continue
    if "#pragma omp" in line:
        nest += 1
        outer, inner, in_body = [], [], False
        continue
    f = FOR.match(line)
    if f:
        b = f.group(3) if f.group(2) == "0" else f.group(3) + "/2"
        (inner if in_body else outer).append(b)
        continue
    s = line.strip()
    if s == "{":
        in_body = True
        continue
    if "+=" in s and outer:
        loops = outer + inner
        n = 1
        for b in loops:
            n *= (bound[b[:-2]] + 1) / 2 if b.endswith("/2") else bound[b]
        per_func[func] += n
        per_nest.append((func, nest, n, " ".join(loops)))
        inner = []
        if not in_body:  # single-statement nest without braces
            outer = []
    elif s == "}" and in_body:
        in_body, outer, inner = False, [], []

total = sum(per_func.values())
res = sum(v for k, v in per_func.items() if k.startswith("compute_"))
print(f"o={no} v={nv}: total {total:.3e} multiply-adds per iteration")
print(f"  residuals {res:.3e}, builders {total - res:.3e}")
for k, v in sorted(per_func.items(), key=lambda kv: -kv[1])[:6]:
    print(f"  {v:.3e}  {k}")
shapes = defaultdict(lambda: [0, 0])
for fn, nid, n, loops in per_nest:
    if fn == "compute_ccsd_doubles_residual":
        ls = [b.split("/")[0] for b in loops.split()]
        key = "o^%dv^%d" % (ls.count("no"), ls.count("nv"))
        shapes[key][0] += 1
        shapes[key][1] += n
print("  doubles residual by shape:")
for k, (c, n) in sorted(shapes.items(), key=lambda kv: -kv[1][1]):
    print(f"    {k}: {c} statements, {n:.3e}")
if len(sys.argv) > 4:
    with open(sys.argv[4], "w") as out:
        for fn, nid, n, loops in per_nest:
            out.write(f"{fn}\t{nid}\t{n}\t{loops}\n")
