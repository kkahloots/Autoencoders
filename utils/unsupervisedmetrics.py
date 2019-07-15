import scipy;
import numpy as np;



def kl_gaussians_numerically_unstable(mean_0, cov_0, mean_1, cov_1, k):
  """Unstable version used for testing gaussian_total_correlation."""
  det_0 = np.linalg.det(cov_0)
  det_1 = np.linalg.det(cov_1)
  inv_1 = np.linalg.inv(cov_1)
  return 0.5 * (
      np.trace(np.matmul(inv_1, cov_0)) + np.dot(mean_1 - mean_0,
                                                 np.dot(inv_1, mean_1 - mean_0))
      - k + np.log(det_1 / det_0))



def gaussian_total_correlation(cov):
  """Computes the total correlation of a Gaussian with covariance matrix cov.
  We use that the total correlation is the KL divergence between the Gaussian
  and the product of its marginals. By design, the means of these two Gaussians
  are zero and the covariance matrix of the second Gaussian is equal to the
  covariance matrix of the first Gaussian with off-diagonal entries set to zero.
  Args:
    cov: Numpy array with covariance matrix.
  Returns:
    Scalar with total correlation.
  """
  return 0.5 * (np.sum(np.log(np.diag(cov))) - np.linalg.slogdet(cov)[1])


def gaussian_wasserstein_correlation(cov):
  """Wasserstein L2 distance between Gaussian and the product of its marginals.
  Args:
    cov: Numpy array with covariance matrix.
  Returns:
    Scalar with score.
  """
  sqrtm = scipy.linalg.sqrtm(cov * np.expand_dims(np.diag(cov), axis=1))
  return 2 * np.trace(cov) - 2 * np.trace(sqrtm)