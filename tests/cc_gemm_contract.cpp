// cc_gemm_contract (src/post_hf/cc/gemm_contract.h) against a label-driven
// reference loop. Covers the staircase (R3b, docs/CCSD_T_SIZE_SCOPE.md): i
// leading the rows with j leading the columns, and the reverse, on the
// contraction shapes the generated spatial doubles kernel uses. Under
// pair_half every element with j >= i must equal the reference and every
// element with j < i must be left as it was (the kernel's mirror owns it).
// Also covers the batched products (U2).
// Non-square o and v so a transposed group shows up.

#include "post_hf/cc/gemm_contract.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace HartreeFock::Correlation::CC;

namespace
{
    constexpr int no = 4, nv = 5;
    int dim(char c) { return std::string("ijkl").find(c) != std::string::npos ? no : nv; }

    TensorND random_tensor(const std::string &labels, std::mt19937 &rng)
    {
        std::vector<int> dims;
        for (char c : labels)
            dims.push_back(dim(c));
        TensorND t(dims);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for (double &x : t.data)
            x = u(rng);
        return t;
    }

    double at(const TensorND &t, const std::string &labels, const std::map<char, int> &v)
    {
        std::size_t off = 0;
        for (std::size_t p = 0; p < labels.size(); ++p)
            off = off * static_cast<std::size_t>(t.dims[p]) + static_cast<std::size_t>(v.at(labels[p]));
        return t.data[off];
    }

    // out(ijab) += coeff * sum over the labels of A and B not in out.
    double reference(const TensorND &a, const std::string &al, const TensorND &b, const std::string &bl,
                     std::map<char, int> v, const std::string &summed, std::size_t p = 0)
    {
        if (p == summed.size())
            return at(a, al, v) * at(b, bl, v);
        double s = 0.0;
        for (int x = 0; x < dim(summed[p]); ++x) {
            v[summed[p]] = x;
            s += reference(a, al, b, bl, v, summed, p + 1);
        }
        return s;
    }

    int check(const char *name, const std::string &al, const std::string &bl,
              const char *m, const char *k, const char *n, bool pair_half, std::mt19937 &rng,
              const char *batch = "")
    {
        const TensorND a = random_tensor(al, rng), b = random_tensor(bl, rng);
        const double coeff = -0.75, init = 3.25;
        TensorND out({no, no, nv, nv});
        for (double &x : out.data)
            x = init;
        cc_gemm_contract(out, "ijab", coeff, cc_operand(a), al, cc_operand(b), bl, m, k, n, pair_half,
                         batch);

        int failures = 0;
        double worst = 0.0;
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int x = 0; x < nv; ++x)
                    for (int y = 0; y < nv; ++y) {
                        const double got = out.data[((static_cast<std::size_t>(i) * no + j) * nv + x) * nv + y];
                        double want = init;
                        if (!pair_half || j >= i)
                            want += coeff * reference(a, al, b, bl, {{'i', i}, {'j', j}, {'a', x}, {'b', y}}, k);
                        worst = std::max(worst, std::abs(got - want));
                        failures += std::abs(got - want) > 1e-12;
                    }
        std::printf("%-34s max |diff| %.2e  %s\n", name, worst, failures ? "FAIL" : "ok");
        return failures;
    }
}

int main()
{
    std::mt19937 rng(20261125);
    int failures = 0;
    // o^3 v^3 ring: i leads the rows, j the columns (columns j >= i).
    failures += check("stair ia|kc|jb", "ikac", "jkbc", "ia", "kc", "jb", true, rng);
    // The reverse: j leads the rows (columns i <= j).
    failures += check("stair jb|kc|ia", "jkbc", "ikac", "jb", "kc", "ia", true, rng);
    // o^3 v^2 shapes: a single-index row group, and a single-index column group.
    failures += check("stair i|c|jba", "ic", "jcba", "i", "c", "jba", true, rng);
    failures += check("stair jab|k|i", "kjab", "ik", "jab", "k", "i", true, rng);
    // i and j in one group (the ladder's filter), and no pair_half at all.
    failures += check("same-group ab|cd|ij", "abcd", "ijcd", "ab", "cd", "ij", true, rng);
    failures += check("full ia|kc|jb", "ikac", "jkbc", "ia", "kc", "jb", false, rng);
    // Batched (U2): the batch labels lead one operand's storage. Slice used
    // in place, transposed, or copied; the batch on B and on A.
    failures += check("batch ia: j|c|b, slice in place", "jc", "iacb", "j", "c", "b", false, rng, "ia");
    failures += check("batch ia: j|c|b, slice transposed", "jc", "iabc", "j", "c", "b", false, rng, "ia");
    failures += check("batch i: j|c|ab, slice copied", "jc", "iacb", "j", "c", "ab", false, rng, "i");
    failures += check("batch ij on A: b|k|a", "ijkb", "ka", "b", "k", "a", false, rng, "ij");
    if (failures) {
        std::printf("FAIL: %d element(s) differ\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
