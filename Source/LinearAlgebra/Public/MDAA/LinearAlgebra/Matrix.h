#pragma once

#include "MDAA/Core/Assert.h"
#include "MDAA/Core/Types.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace MDAA {

template <i32 Size>
using ColumnVector = Eigen::Matrix<f64, Size, 1>;

template <i32 Size>
using SquareMatrix = Eigen::Matrix<f64, Size, Size>;

template <i32 Size>
[[nodiscard]] SquareMatrix<Size> DiagonalSquareRoot(const ColumnVector<Size> &values);

template <i32 Size>
[[nodiscard]] SquareMatrix<Size> SpdMatrixRoot(const SquareMatrix<Size> &a);

template <i32 Size>
SquareMatrix<Size> DiagonalSquareRoot(const ColumnVector<Size> &values) {
    MDAA_CHECKF(
        values.minCoeff() > 0.0,
        "the values of a diagonal matrix have to be positive for a real root, the smallest is {}",
        values.minCoeff());

    return SquareMatrix<Size>(ColumnVector<Size>(values.array().sqrt()).asDiagonal());
}

template <i32 Size>
SquareMatrix<Size> SpdMatrixRoot(const SquareMatrix<Size> &a) {
    MDAA_CHECKF(a.isApprox(a.transpose()), "M * transpose(M) + n * I has to stay symmetric");

    const Eigen::SelfAdjointEigenSolver<SquareMatrix<Size>> solver(a);
    MDAA_CHECKF(solver.info() == Eigen::Success, "the eigen decomposition of A failed");

    const ColumnVector<Size> &eigenvalues = solver.eigenvalues();
    MDAA_CHECKF(
        eigenvalues.minCoeff() > 0.0,
        "A has to be positive definite, the smallest eigenvalue is {}",
        eigenvalues.minCoeff());

    const SquareMatrix<Size> root = solver.eigenvectors() *
                                    eigenvalues.array().sqrt().matrix().asDiagonal() *
                                    solver.eigenvectors().transpose();
    MDAA_CHECKF(
        ((root.transpose() * root) - a).norm() < 1.0e-9,
        "A^1/2 squared has to be A, the gap is {}",
        ((root.transpose() * root) - a).norm());

    return root;
}

} // namespace MDAA
