#include "post_hf/cc/common.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace
{
    // Every tensor constructor goes through the same checked product helper so a
    // bad dimension or integer overflow fails early with a readable message.
    std::size_t checked_product(
        std::initializer_list<int> dims,
        const char *label)
    {
        std::size_t total = 1;
        for (const int dim : dims)
        {
            if (dim < 0)
                throw std::invalid_argument(std::string(label) + ": tensor dimension cannot be negative");

            const std::size_t dim_size = static_cast<std::size_t>(dim);
            if (dim_size != 0 &&
                total > std::numeric_limits<std::size_t>::max() / dim_size)
                throw std::overflow_error(std::string(label) + ": tensor size overflow");

            total *= dim_size;
        }
        return total;
    }

    std::size_t checked_product(
        const std::vector<int> &dims,
        const char *label)
    {
        std::size_t total = 1;
        for (const int dim : dims)
        {
            if (dim < 0)
                throw std::invalid_argument(std::string(label) + ": tensor dimension cannot be negative");

            const std::size_t dim_size = static_cast<std::size_t>(dim);
            if (dim_size != 0 &&
                total > std::numeric_limits<std::size_t>::max() / dim_size)
                throw std::overflow_error(std::string(label) + ": tensor size overflow");

            total *= dim_size;
        }
        return total;
    }

    std::size_t flatten_index(
        const std::vector<int> &dims,
        const std::vector<int> &indices,
        const char *label)
    {
        if (dims.size() != indices.size())
            throw std::invalid_argument(std::string(label) + ": index count does not match tensor order");

        std::size_t offset = 0;
        for (std::size_t pos = 0; pos < dims.size(); ++pos)
        {
            const int dim = dims[pos];
            const int idx = indices[pos];
            if (idx < 0 || idx >= dim)
                throw std::out_of_range(std::string(label) + ": tensor index out of bounds");

            offset *= static_cast<std::size_t>(dim);
            offset += static_cast<std::size_t>(idx);
        }
        return offset;
    }

    std::vector<int> to_vector(std::initializer_list<int> values)
    {
        return std::vector<int>(values.begin(), values.end());
    }
} // namespace

namespace HartreeFock::Correlation::CC
{
    namespace
    {
        [[nodiscard]] int count_electrons(const HartreeFock::Calculator &calculator)
        {
            int n_electrons = 0;
            for (const auto Z : calculator._molecule.atomic_numbers)
                n_electrons += static_cast<int>(Z);
            n_electrons -= calculator._molecule.charge;
            return n_electrons;
        }
    } // namespace

    Tensor2D::Tensor2D(int d1, int d2, double value)
        : dim1(d1), dim2(d2), data(checked_product({d1, d2}, "Tensor2D"), value)
    {
    }

    Tensor2D::Tensor2D(int d1, int d2, std::vector<double> values)
        : dim1(d1), dim2(d2), data(std::move(values))
    {
        if (data.size() != checked_product({d1, d2}, "Tensor2D"))
            throw std::invalid_argument("Tensor2D: flat data size does not match dimensions");
    }

    std::size_t Tensor2D::size() const noexcept
    {
        return data.size();
    }

    double &Tensor2D::operator()(int i, int j) noexcept
    {
        return data[(static_cast<std::size_t>(i) * static_cast<std::size_t>(dim2)) +
                    static_cast<std::size_t>(j)];
    }

    const double &Tensor2D::operator()(int i, int j) const noexcept
    {
        return data[(static_cast<std::size_t>(i) * static_cast<std::size_t>(dim2)) +
                    static_cast<std::size_t>(j)];
    }

    Tensor4D::Tensor4D(int d1, int d2, int d3, int d4, double value)
        : dim1(d1), dim2(d2), dim3(d3), dim4(d4),
          data(checked_product({d1, d2, d3, d4}, "Tensor4D"), value)
    {
    }

    Tensor4D::Tensor4D(int d1, int d2, int d3, int d4, std::vector<double> values)
        : dim1(d1), dim2(d2), dim3(d3), dim4(d4), data(std::move(values))
    {
        if (data.size() != checked_product({d1, d2, d3, d4}, "Tensor4D"))
            throw std::invalid_argument("Tensor4D: flat data size does not match dimensions");
    }

    std::size_t Tensor4D::size() const noexcept
    {
        return data.size();
    }

    double &Tensor4D::operator()(int i, int j, int k, int l) noexcept
    {
        const std::size_t d2 = static_cast<std::size_t>(dim2);
        const std::size_t d3 = static_cast<std::size_t>(dim3);
        const std::size_t d4 = static_cast<std::size_t>(dim4);
        return data[(((static_cast<std::size_t>(i) * d2) + static_cast<std::size_t>(j)) * d3 +
                     static_cast<std::size_t>(k)) *
                        d4 +
                    static_cast<std::size_t>(l)];
    }

    const double &Tensor4D::operator()(int i, int j, int k, int l) const noexcept
    {
        const std::size_t d2 = static_cast<std::size_t>(dim2);
        const std::size_t d3 = static_cast<std::size_t>(dim3);
        const std::size_t d4 = static_cast<std::size_t>(dim4);
        return data[(((static_cast<std::size_t>(i) * d2) + static_cast<std::size_t>(j)) * d3 +
                     static_cast<std::size_t>(k)) *
                        d4 +
                    static_cast<std::size_t>(l)];
    }

    Tensor6D::Tensor6D(int d1, int d2, int d3, int d4, int d5, int d6, double value)
        : dim1(d1), dim2(d2), dim3(d3), dim4(d4), dim5(d5), dim6(d6),
          data(checked_product({d1, d2, d3, d4, d5, d6}, "Tensor6D"), value)
    {
    }

    Tensor6D::Tensor6D(int d1, int d2, int d3, int d4, int d5, int d6, std::vector<double> values)
        : dim1(d1), dim2(d2), dim3(d3), dim4(d4), dim5(d5), dim6(d6), data(std::move(values))
    {
        if (data.size() != checked_product({d1, d2, d3, d4, d5, d6}, "Tensor6D"))
            throw std::invalid_argument("Tensor6D: flat data size does not match dimensions");
    }

    std::size_t Tensor6D::size() const noexcept
    {
        return data.size();
    }

    double &Tensor6D::operator()(int i, int j, int k, int l, int m, int n) noexcept
    {
        const std::size_t d2 = static_cast<std::size_t>(dim2);
        const std::size_t d3 = static_cast<std::size_t>(dim3);
        const std::size_t d4 = static_cast<std::size_t>(dim4);
        const std::size_t d5 = static_cast<std::size_t>(dim5);
        const std::size_t d6 = static_cast<std::size_t>(dim6);
        return data[(((((static_cast<std::size_t>(i) * d2) + static_cast<std::size_t>(j)) * d3 +
                       static_cast<std::size_t>(k)) *
                          d4 +
                      static_cast<std::size_t>(l)) *
                         d5 +
                     static_cast<std::size_t>(m)) *
                        d6 +
                    static_cast<std::size_t>(n)];
    }

    const double &Tensor6D::operator()(int i, int j, int k, int l, int m, int n) const noexcept
    {
        const std::size_t d2 = static_cast<std::size_t>(dim2);
        const std::size_t d3 = static_cast<std::size_t>(dim3);
        const std::size_t d4 = static_cast<std::size_t>(dim4);
        const std::size_t d5 = static_cast<std::size_t>(dim5);
        const std::size_t d6 = static_cast<std::size_t>(dim6);
        return data[(((((static_cast<std::size_t>(i) * d2) + static_cast<std::size_t>(j)) * d3 +
                       static_cast<std::size_t>(k)) *
                          d4 +
                      static_cast<std::size_t>(l)) *
                         d5 +
                     static_cast<std::size_t>(m)) *
                        d6 +
                    static_cast<std::size_t>(n)];
    }

    TensorND::TensorND(std::vector<int> dims_in, double value)
        : dims(std::move(dims_in)),
          data(checked_product(dims, "TensorND"), value)
    {
    }

    TensorND::TensorND(std::vector<int> dims_in, std::vector<double> values)
        : dims(std::move(dims_in)), data(std::move(values))
    {
        if (data.size() != checked_product(dims, "TensorND"))
            throw std::invalid_argument("TensorND: flat data size does not match dimensions");
    }

    std::size_t TensorND::size() const noexcept
    {
        return data.size();
    }

    int TensorND::order() const noexcept
    {
        return static_cast<int>(dims.size());
    }

    double &TensorND::operator()(std::initializer_list<int> indices)
    {
        return (*this)(to_vector(indices));
    }

    const double &TensorND::operator()(std::initializer_list<int> indices) const
    {
        return (*this)(to_vector(indices));
    }

    double &TensorND::operator()(const std::vector<int> &indices)
    {
        return data[flatten_index(dims, indices, "TensorND")];
    }

    const double &TensorND::operator()(const std::vector<int> &indices) const
    {
        return data[flatten_index(dims, indices, "TensorND")];
    }

    std::size_t DenseTensorView::size() const
    {
        return checked_product(dims, "DenseTensorView");
    }

    int DenseTensorView::order() const noexcept
    {
        return static_cast<int>(dims.size());
    }

    double &DenseTensorView::operator()(std::initializer_list<int> indices)
    {
        return (*this)(to_vector(indices));
    }

    const double &DenseTensorView::operator()(std::initializer_list<int> indices) const
    {
        return (*this)(to_vector(indices));
    }

    double &DenseTensorView::operator()(const std::vector<int> &indices)
    {
        return data[flatten_index(dims, indices, "DenseTensorView")];
    }

    const double &DenseTensorView::operator()(const std::vector<int> &indices) const
    {
        return data[flatten_index(dims, indices, "DenseTensorView")];
    }

    std::size_t ConstDenseTensorView::size() const
    {
        return checked_product(dims, "ConstDenseTensorView");
    }

    int ConstDenseTensorView::order() const noexcept
    {
        return static_cast<int>(dims.size());
    }

    const double &ConstDenseTensorView::operator()(std::initializer_list<int> indices) const
    {
        return (*this)(to_vector(indices));
    }

    const double &ConstDenseTensorView::operator()(const std::vector<int> &indices) const
    {
        return data[flatten_index(dims, indices, "ConstDenseTensorView")];
    }

    DenseTensorView make_tensor_view(Tensor2D &tensor)
    {
        return DenseTensorView{
            .dims = {tensor.dim1, tensor.dim2},
            .data = tensor.data.data(),
        };
    }

    DenseTensorView make_tensor_view(Tensor4D &tensor)
    {
        return DenseTensorView{
            .dims = {tensor.dim1, tensor.dim2, tensor.dim3, tensor.dim4},
            .data = tensor.data.data(),
        };
    }

    DenseTensorView make_tensor_view(Tensor6D &tensor)
    {
        return DenseTensorView{
            .dims = {tensor.dim1, tensor.dim2, tensor.dim3, tensor.dim4, tensor.dim5, tensor.dim6},
            .data = tensor.data.data(),
        };
    }

    DenseTensorView make_tensor_view(TensorND &tensor)
    {
        return DenseTensorView{
            .dims = tensor.dims,
            .data = tensor.data.data(),
        };
    }

    ConstDenseTensorView make_tensor_view(const Tensor2D &tensor)
    {
        return ConstDenseTensorView{
            .dims = {tensor.dim1, tensor.dim2},
            .data = tensor.data.data(),
        };
    }

    ConstDenseTensorView make_tensor_view(const Tensor4D &tensor)
    {
        return ConstDenseTensorView{
            .dims = {tensor.dim1, tensor.dim2, tensor.dim3, tensor.dim4},
            .data = tensor.data.data(),
        };
    }

    ConstDenseTensorView make_tensor_view(const Tensor6D &tensor)
    {
        return ConstDenseTensorView{
            .dims = {tensor.dim1, tensor.dim2, tensor.dim3, tensor.dim4, tensor.dim5, tensor.dim6},
            .data = tensor.data.data(),
        };
    }

    ConstDenseTensorView make_tensor_view(const TensorND &tensor)
    {
        return ConstDenseTensorView{
            .dims = tensor.dims,
            .data = tensor.data.data(),
        };
    }

    std::expected<RHFReference, std::string> build_rhf_reference(
        HartreeFock::Calculator &calculator)
    {
        // The coupled-cluster code assumes a canonical closed-shell reference.
        // Keeping the validation here avoids repeating the same checks in each
        // solver entry point.
        if (!calculator._info._is_converged)
            return std::unexpected("build_rhf_reference: SCF not converged.");
        if (calculator._scf._scf != HartreeFock::SCFType::RHF ||
            calculator._info._scf.is_uhf)
            return std::unexpected("build_rhf_reference: canonical RHF reference required.");

        const int n_ao = static_cast<int>(calculator._shells.nbasis());
        const Eigen::MatrixXd &C = calculator._info._scf.alpha.mo_coefficients;
        const Eigen::VectorXd &eps = calculator._info._scf.alpha.mo_energies;
        const int n_electrons = count_electrons(calculator);

        if (n_electrons % 2 != 0)
            return std::unexpected("build_rhf_reference: closed-shell RHF reference required.");

        RHFReference ref;
        ref.n_ao = n_ao;
        ref.n_mo = n_ao;
        ref.n_occ = n_electrons / 2;
        ref.n_virt = n_ao - ref.n_occ;

        if (ref.n_occ <= 0 || ref.n_virt <= 0)
            return std::unexpected("build_rhf_reference: no occupied or virtual orbitals.");
        if (C.rows() != n_ao || C.cols() != n_ao)
            return std::unexpected("build_rhf_reference: MO coefficient matrix has wrong dimensions.");
        if (eps.size() != n_ao)
            return std::unexpected("build_rhf_reference: MO energy vector has wrong length.");

        // The SCF driver already orders canonical occupied orbitals before the
        // virtual block, so the CC partition is just a matrix slice.
        ref.C_occ = C.leftCols(ref.n_occ);
        ref.C_virt = C.middleCols(ref.n_occ, ref.n_virt);
        ref.eps_occ = eps.head(ref.n_occ);
        ref.eps_virt = eps.tail(ref.n_virt);
        return ref;
    }

    std::expected<UHFReference, std::string> build_uhf_reference(
        HartreeFock::Calculator &calculator)
    {
        if (!calculator._info._is_converged)
            return std::unexpected("build_uhf_reference: SCF not converged.");
        if (calculator._scf._scf != HartreeFock::SCFType::UHF ||
            !calculator._info._scf.is_uhf)
            return std::unexpected("build_uhf_reference: canonical UHF reference required.");

        const int n_ao = static_cast<int>(calculator._shells.nbasis());
        const Eigen::MatrixXd &Ca = calculator._info._scf.alpha.mo_coefficients;
        const Eigen::VectorXd &epsa = calculator._info._scf.alpha.mo_energies;
        const Eigen::MatrixXd &Cb = calculator._info._scf.beta.mo_coefficients;
        const Eigen::VectorXd &epsb = calculator._info._scf.beta.mo_energies;

        const int n_electrons = count_electrons(calculator);
        const int n_unpaired = static_cast<int>(calculator._molecule.multiplicity) - 1;
        const int n_alpha = (n_electrons + n_unpaired) / 2;
        const int n_beta = (n_electrons - n_unpaired) / 2;

        if (n_alpha < 0 || n_beta < 0)
            return std::unexpected("build_uhf_reference: invalid alpha/beta occupation counts.");
        if (n_alpha > n_ao || n_beta > n_ao)
            return std::unexpected("build_uhf_reference: occupied orbitals exceed MO dimension.");
        if (n_alpha == 0 || n_beta == 0)
            return std::unexpected("build_uhf_reference: both spin channels need at least one occupied orbital.");
        if (n_alpha == n_ao || n_beta == n_ao)
            return std::unexpected("build_uhf_reference: both spin channels need at least one virtual orbital.");
        if (Ca.rows() != n_ao || Ca.cols() != n_ao || Cb.rows() != n_ao || Cb.cols() != n_ao)
            return std::unexpected("build_uhf_reference: MO coefficient matrices have wrong dimensions.");
        if (epsa.size() != n_ao || epsb.size() != n_ao)
            return std::unexpected("build_uhf_reference: MO energy vectors have wrong length.");

        UHFReference ref;
        ref.n_ao = n_ao;
        ref.n_mo = n_ao;
        ref.n_occ_alpha = n_alpha;
        ref.n_occ_beta = n_beta;
        ref.n_virt_alpha = n_ao - n_alpha;
        ref.n_virt_beta = n_ao - n_beta;
        ref.C_alpha = Ca;
        ref.C_beta = Cb;
        ref.eps_alpha = epsa;
        ref.eps_beta = epsb;
        return ref;
    }
} // namespace HartreeFock::Correlation::CC
