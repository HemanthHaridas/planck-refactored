#include "ks_matrix.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace DFT
{

    namespace
    {

        template <class AOGrid>
        std::expected<void, std::string> validate_grid_inputs(
            const MolecularGrid &molecular_grid,
            const AOGrid &ao_grid,
            const XCGridEvaluation &xc_grid)
        {
            if (molecular_grid.points.cols() != 4)
                return std::unexpected("XC matrix assembly requires quadrature weights in the fourth molecular-grid column");

            if (ao_grid.npoints() != molecular_grid.points.rows())
                return std::unexpected("AO grid and molecular grid point counts do not match");

            if (xc_grid.density.npoints() != molecular_grid.points.rows())
                return std::unexpected("XC grid density and molecular grid point counts do not match");

            if (xc_grid.vrho.rows() != molecular_grid.points.rows())
                return std::unexpected("XC vrho array and molecular grid point counts do not match");

            if (xc_grid.density.polarized)
            {
                if (xc_grid.vrho.cols() != 2)
                    return std::unexpected("Polarized XC matrix assembly requires vrho with two spin columns");
                if (!(xc_grid.vsigma.cols() == 0 || xc_grid.vsigma.cols() == 3))
                    return std::unexpected("Polarized XC matrix assembly requires vsigma with either 0 or 3 columns");
            }
            else
            {
                if (xc_grid.vrho.cols() != 1)
                    return std::unexpected("Unpolarized XC matrix assembly requires vrho with one column");
                if (!(xc_grid.vsigma.cols() == 0 || xc_grid.vsigma.cols() == 1))
                    return std::unexpected("Unpolarized XC matrix assembly requires vsigma with either 0 or 1 column");
            }

            return {};
        }

        constexpr Eigen::Index kXCMatrixMaxBins = 64;
        constexpr double kXCMatrixBinBudgetBytes = 256.0e6;

        // Bins of contiguous point blocks. The count depends on nb and the spin
        // count (a 256 MB cap on the nb x nb partials, at least 8 bins) and on
        // the block count, never on the thread count, so the bin-ordered sum is
        // identical at any number of threads.
        Eigen::Index xc_matrix_bins(Eigen::Index nbasis, Eigen::Index nblocks, int nspin)
        {
            const double per_bin = 8.0 * static_cast<double>(nbasis) * static_cast<double>(nbasis) * nspin;
            const auto cap = static_cast<Eigen::Index>(kXCMatrixBinBudgetBytes / std::max(per_bin, 1.0));
            return std::max<Eigen::Index>(1, std::min({kXCMatrixMaxBins, std::max<Eigen::Index>(cap, 8), nblocks}));
        }

        // One spin channel of rows [r0, r0 + n) of one AO block:
        // A = diag(w vrho / 2) Phi + diag(w) G, G = sum_k c_k dPhi/dk with
        // c = coeff (GGA only). Then
        // sum_p w vrho phi phi^T + w (phi g^T + g phi^T) = Phi^T A + A^T Phi,
        // so M += Phi^T A here and V = M + M^T after the reduction. A screened
        // block adds its k x k product at its kept AOs. Zero-weight points get
        // a zero row, so a non-finite XC value there cannot leak in.
        void accumulate_block(
            Eigen::MatrixXd &M,
            Eigen::MatrixXd &A,
            Eigen::MatrixXd &Mk,
            const AOBlockView &blk,
            Eigen::Index r0,
            Eigen::Index n,
            const Eigen::Ref<const Eigen::VectorXd> &w,
            const Eigen::Ref<const Eigen::VectorXd> &vrho,
            const Eigen::MatrixXd *coeff)
        {
            const auto phi = blk.phi.middleRows(r0, n);
            Eigen::VectorXd a0(n);
            for (Eigen::Index i = 0; i < n; ++i)
                a0(i) = (w(i) == 0.0) ? 0.0 : 0.5 * w(i) * vrho(i);
            A = a0.asDiagonal() * phi;
            if (coeff)
            {
                for (int k = 0; k < 3; ++k)
                {
                    Eigen::VectorXd ck(n);
                    for (Eigen::Index i = 0; i < n; ++i)
                        ck(i) = (w(i) == 0.0) ? 0.0 : w(i) * (*coeff)(i, k);
                    A.noalias() += ck.asDiagonal() * blk.grad[k].middleRows(r0, n);
                }
            }
            if (!blk.cols)
            {
                M.noalias() += phi.transpose() * A;
                return;
            }
            Mk.noalias() = phi.transpose() * A;
            M(*blk.cols, *blk.cols) += Mk;
        }

        template <class AOGrid>
        std::expected<XCMatrixContribution, std::string>
        assemble_xc_matrix_impl(
            const MolecularGrid &molecular_grid,
            const AOGrid &ao_grid,
            const XCGridEvaluation &xc_grid,
            Eigen::Index point_begin,
            Eigen::Index point_end)
        {
            if (auto valid = validate_grid_inputs(molecular_grid, ao_grid, xc_grid); !valid)
                return std::unexpected(valid.error());

            const Eigen::Index nbasis = ao_grid.nbasis();

            // MPI: each rank assembles only its contiguous point slice, and the
            // driver reduces the nb^2 partials. point_end < 0 means "all points"
            // (the serial default), so non-MPI and test call sites are unchanged.
            // The point loop runs [point_begin, point_end); the cross-rank sum of
            // disjoint slices is the same MPI_SUM pattern the Fock reduce uses.
            const Eigen::Index all_points = ao_grid.npoints();
            const Eigen::Index lo = (point_end < 0) ? 0 : point_begin;
            const Eigen::Index npoints = (point_end < 0) ? all_points : point_end;
            if (lo < 0 || npoints > all_points || lo > npoints)
                return std::unexpected("assemble_xc_matrix: point slice out of range");

            XCMatrixContribution contribution;
            contribution.polarized = xc_grid.density.polarized;

            const bool polarized = xc_grid.density.polarized;
            const bool gga = xc_grid.vsigma.cols() > 0;
            const int nspin = polarized ? 2 : 1;

            // Blocked GEMMs (see accumulate_block) over the AO blocks that
            // overlap the slice, in fixed bins of contiguous blocks summed in
            // bin order: bitwise identical at any thread count.
            const Eigen::Index b_lo = lo / kAOBlock;
            const Eigen::Index nblocks = ao_block_count(npoints) - b_lo;
            const Eigen::Index nbins = xc_matrix_bins(nbasis, nblocks, nspin);
            const Eigen::Index blocks_per_bin = (nblocks + nbins - 1) / nbins;
            std::vector<Eigen::MatrixXd> alpha_bins(nbins), beta_bins(polarized ? nbins : 0);

#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (Eigen::Index bin = 0; bin < nbins; ++bin)
            {
                Eigen::MatrixXd &Ma = alpha_bins[bin];
                Ma.setZero(nbasis, nbasis);
                if (polarized)
                    beta_bins[bin].setZero(nbasis, nbasis);
                Eigen::MatrixXd A, Mk, coeff_a, coeff_b;

                const Eigen::Index b_end = b_lo + std::min(nblocks, (bin + 1) * blocks_per_bin);
                for (Eigen::Index b = b_lo + bin * blocks_per_bin; b < b_end; ++b)
                {
                    const AOBlockView blk = ao_grid.block(b);
                    const Eigen::Index r0 = std::max(lo, blk.p0) - blk.p0;
                    const Eigen::Index n = std::min(npoints, blk.p0 + blk.n) - blk.p0 - r0;
                    if (n <= 0)
                        continue;
                    const Eigen::Index p0 = blk.p0 + r0;
                    const auto w = molecular_grid.points.col(3).segment(p0, n);
                    if ((w.array() == 0.0).all())
                        continue;

                    if (!polarized)
                    {
                        if (gga)
                        {
                            // coefficient = 2 vsigma grad(rho)
                            const auto &d = xc_grid.density.total;
                            coeff_a.resize(n, 3);
                            const auto vs = xc_grid.vsigma.col(0).segment(p0, n).array();
                            coeff_a.col(0) = 2.0 * vs * d.grad_x.segment(p0, n).array();
                            coeff_a.col(1) = 2.0 * vs * d.grad_y.segment(p0, n).array();
                            coeff_a.col(2) = 2.0 * vs * d.grad_z.segment(p0, n).array();
                        }
                        accumulate_block(Ma, A, Mk, blk, r0, n, w, xc_grid.vrho.col(0).segment(p0, n),
                                         gga ? &coeff_a : nullptr);
                        continue;
                    }

                    if (gga)
                    {
                        // alpha: 2 vs_aa grad_a + vs_ab grad_b; beta: vs_ab grad_a + 2 vs_bb grad_b
                        const auto &da = xc_grid.density.alpha;
                        const auto &db = xc_grid.density.beta;
                        const auto aa = xc_grid.vsigma.col(0).segment(p0, n).array();
                        const auto ab = xc_grid.vsigma.col(1).segment(p0, n).array();
                        const auto bb = xc_grid.vsigma.col(2).segment(p0, n).array();
                        const std::array<const Eigen::VectorXd *, 3> ga = {&da.grad_x, &da.grad_y, &da.grad_z};
                        const std::array<const Eigen::VectorXd *, 3> gb = {&db.grad_x, &db.grad_y, &db.grad_z};
                        coeff_a.resize(n, 3);
                        coeff_b.resize(n, 3);
                        for (int k = 0; k < 3; ++k)
                        {
                            const auto gak = ga[k]->segment(p0, n).array();
                            const auto gbk = gb[k]->segment(p0, n).array();
                            coeff_a.col(k) = 2.0 * aa * gak + ab * gbk;
                            coeff_b.col(k) = ab * gak + 2.0 * bb * gbk;
                        }
                    }
                    accumulate_block(Ma, A, Mk, blk, r0, n, w, xc_grid.vrho.col(0).segment(p0, n),
                                     gga ? &coeff_a : nullptr);
                    accumulate_block(beta_bins[bin], A, Mk, blk, r0, n, w, xc_grid.vrho.col(1).segment(p0, n),
                                     gga ? &coeff_b : nullptr);
                }
            }

            Eigen::MatrixXd Ma = Eigen::MatrixXd::Zero(nbasis, nbasis);
            Eigen::MatrixXd Mb = Eigen::MatrixXd::Zero(polarized ? nbasis : 0, polarized ? nbasis : 0);
            for (Eigen::Index bin = 0; bin < nbins; ++bin)
            {
                Ma += alpha_bins[bin];
                if (polarized)
                    Mb += beta_bins[bin];
            }
            contribution.alpha = Ma + Ma.transpose();
            contribution.beta = polarized ? Eigen::MatrixXd(Mb + Mb.transpose()) : contribution.alpha;

            return contribution;
        }

    } // namespace

    std::expected<XCMatrixContribution, std::string>
    assemble_xc_matrix(const MolecularGrid &molecular_grid, const AOGridEvaluation &ao_grid,
                       const XCGridEvaluation &xc_grid, Eigen::Index point_begin, Eigen::Index point_end)
    {
        return assemble_xc_matrix_impl(molecular_grid, ao_grid, xc_grid, point_begin, point_end);
    }

    std::expected<XCMatrixContribution, std::string>
    assemble_xc_matrix(const MolecularGrid &molecular_grid, const ScreenedAOGrid &ao_grid,
                       const XCGridEvaluation &xc_grid, Eigen::Index point_begin, Eigen::Index point_end)
    {
        return assemble_xc_matrix_impl(molecular_grid, ao_grid, xc_grid, point_begin, point_end);
    }

    KSPotentialMatrices combine_ks_potential(
        const Eigen::Ref<const Eigen::MatrixXd> &coulomb,
        const XCMatrixContribution &xc_matrix,
        double exact_exchange_coefficient,
        const Eigen::MatrixXd &exact_exchange_alpha,
        const Eigen::MatrixXd &exact_exchange_beta,
        double exact_exchange_energy)
    {
        KSPotentialMatrices potential;
        potential.polarized = xc_matrix.polarized;
        potential.coulomb = coulomb;
        potential.xc_alpha = xc_matrix.alpha;
        potential.xc_beta = xc_matrix.beta;
        potential.exact_exchange_alpha =
            exact_exchange_alpha.size() == 0
                ? Eigen::MatrixXd::Zero(coulomb.rows(), coulomb.cols())
                : exact_exchange_alpha;
        potential.exact_exchange_beta =
            exact_exchange_beta.size() == 0
                ? potential.exact_exchange_alpha
                : exact_exchange_beta;
        potential.exact_exchange_coefficient = exact_exchange_coefficient;
        potential.exact_exchange_energy = exact_exchange_energy;
        potential.alpha = coulomb + xc_matrix.alpha + potential.exact_exchange_alpha;
        if (xc_matrix.polarized)
            potential.beta = coulomb + xc_matrix.beta + potential.exact_exchange_beta;
        else
            potential.beta = potential.alpha;
        return potential;
    }

} // namespace DFT
