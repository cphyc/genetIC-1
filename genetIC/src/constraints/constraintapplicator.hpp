#ifndef _CONSTRAINTAPPLICATOR_HPP
#define _CONSTRAINTAPPLICATOR_HPP

#include "src/multilevelcontext.hpp"
#include "src/field/multilevelfield.hpp"

namespace constraints {
  template<typename DataType, typename T=strip_complex<DataType>>
  class ConstraintApplicator {
    /* This class takes responsibility for collecting a list of linear constraints (on all grid levels),
     * orthonormalizing and then applying them to the multi-level field */

  public:
    std::vector<fields::ConstraintField<DataType>> alphas;
    std::vector<DataType> values;
    std::vector<DataType> existing_values;

    MultiLevelContextInformation<DataType> *underlying;
    fields::OutputField<DataType> *outputField;

    ConstraintApplicator(MultiLevelContextInformation<DataType> *underlying_,
                         fields::OutputField<DataType> *outputField_) : underlying(underlying_),
                                                                        outputField(outputField_) {

    }

    void add_constraint(fields::ConstraintField<DataType> &&alpha, DataType value, DataType existing) {

      alphas.push_back(std::move(alpha));
      values.push_back(value);
      existing_values.push_back(existing);
    }

    void print_covariance() {
      /* This routine is provided mainly for debugging purposes and calculates/displays the covariance matrix
       * alpha_i^dagger alpha_j  */

      progress::ProgressBar pb("calculating covariance");

      size_t n = alphas.size();
      size_t done = 0;
      std::vector<std::vector<std::complex<T>>> c_matrix(n, std::vector<std::complex<T>>(n, 0));

      for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
          pb.setProgress(((float) done * 2) / (n * (1 + n)));
          c_matrix[i][j] = alphas[i].innerProduct(alphas[j]);
          c_matrix[j][i] = c_matrix[i][j];
          done += 1;
        }
      }

      std::cout << std::endl << "cov_matr = [";
      for (size_t i = 0; i < n; i++) {
        std::cout << "[";
        for (size_t j = 0; j < n; j++) {
          std::cout << std::real(c_matrix[i][j]);
          if (j < n - 1) std::cout << ",";
        }
        std::cout << "]";
        if (i < n - 1) std::cout << "," << std::endl;
      }
      std::cout << "]" << std::endl;
    }

    void orthonormaliseConstraints() {
      /* Constraints need to be orthonormal before applying (or alternatively one would need an extra matrix
       * manipulation on them, as in the original HR91 paper, which boils down to the same thing). */

      using namespace numerics;

      size_t n = alphas.size();
      size_t done = 0;

      size_t nCells = underlying->getNumCells();

      // It can be helpful to see a summary of the constraints being applied, with the starting and target values
      std::cout << "v0=" << real(existing_values) << std::endl;
      std::cout << "v1=" << real(values) << std::endl;

      // Store transformation matrix, just for display purposes
      std::vector<std::vector<std::complex<T>>> t_matrix(n, std::vector<std::complex<T>>(n, 0));

      for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
          if (i == j) t_matrix[i][j] = 1.0; else t_matrix[i][j] = 0.0;
        }
      }

      // Gram-Schmidt orthogonalization in-place

      progress::ProgressBar pb("orthogonalizing constraints");

      for (size_t i = 0; i < n; i++) {
        auto &alpha_i = alphas[i];
        for (size_t j = 0; j < i; j++) {
          auto &alpha_j = alphas[j];
          pb.setProgress(((float) done * 2) / (n * (1 + n)));
          DataType result = alpha_i.innerProduct(alpha_j);

          alpha_i.addScaled(alpha_j, -result);


          // update display matrix accordingly such that at each step
          // our current values array is equal to t_matrix . input_values
          for (size_t k = 0; k <= j; k++)
            t_matrix[i][k] -= result * t_matrix[j][k];

          // update constraining value
          values[i] -= result * values[j];
          done += 1; // one op for each of the orthognalizing constraints
        }

        // normalize
        DataType norm = sqrt(alpha_i.innerProduct(alpha_i));

        alpha_i /= norm;
        values[i] /= norm;


        for (size_t j = 0; j < n; j++) {
          t_matrix[i][j] /= norm;
        }
        done += 1; // one op for the normalization

      }

      // The existing values are now invalid because the constraint vectors have been changed.
      // Re-calculate them.
      // TODO: rather than recalculate them, it would save CPU time to update them alongside the vectors/target values.
      // This is actually pretty trivial but needs to be carefully tested.
      existing_values.clear();
      for (size_t i = 0; i < n; i++) {
        auto &alpha_i = alphas[i];
        existing_values.push_back(alpha_i.innerProduct(*outputField));
      }


      // Finally display t_matrix^{dagger} t_matrix, which is the matrix allowing one
      // to form chi^2: Delta chi^2 = d_1^dagger t_matrix^dagger t_matrix d_1 -
      // d_0^dagger t_matrix^dagger t_matrix d_0.

      std::cout << std::endl << "chi2_matr = [";
      for (size_t i = 0; i < n; i++) {
        std::cout << "[";
        for (size_t j = 0; j < n; j++) {
          std::complex<T> r = 0;
          for (size_t k = 0; k < n; k++) {
            r += std::conj(t_matrix[k][i]) * t_matrix[k][j];
          }
          std::cout << std::real(r) / nCells;
          if (j < n - 1) std::cout << ",";
        }
        std::cout << "]";
        if (i < n - 1) std::cout << "," << std::endl;
      }
      std::cout << "]" << std::endl;
    }


    void applyConstraints() {

      orthonormaliseConstraints();
      outputField->toFourier();

      for (size_t i = 0; i < alphas.size(); i++) {
        fields::MultiLevelField<DataType> &alpha_i = alphas[i];
        auto dval_i = values[i] - existing_values[i];

        // Constraints are essentially covectors. Convert to a vector, with correct weighting/filtering.
        // See discussion of why it's convenient to regard constraints as covectors in multilevelfield.hpp
        alpha_i.convertToVector();

        alpha_i.toFourier(); // almost certainly already is in Fourier space, but just to be safe
        outputField->addScaled(alpha_i, dval_i);
      }
    }

  };
}


#endif