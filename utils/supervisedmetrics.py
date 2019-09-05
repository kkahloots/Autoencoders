import numpy as np
import scipy
import sklearn
from sklearn.ensemble import GradientBoostingClassifier


def mig(mus,ys):
    score_dict = {}
    discretized_mus =_histogram_discretize(mus,20) ;
    m = discrete_mutual_info(discretized_mus, ys)
    assert m.shape[0] == mus.shape[0]
    assert m.shape[1] == ys.shape[0]
    # m is [num_latents, num_factors]
    entropy = discrete_entropy(ys)
    sorted_m = np.sort(m, axis=0)[::-1]
    mig = np.mean(
    np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
    #print("Entropy {} M value {}".format(entropy,m))
    return mig;

def discrete_mutual_info(mus, ys):
  """Compute discrete mutual information."""
  num_codes = mus.shape[0]
  num_factors = ys.shape[0]
  #print(mus.shape,ys.shape)
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
      Temp = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
      #print("Temp variable {}".format(Temp))
      m[i, j] = Temp
  return m

def discrete_entropy(ys):
  """Compute discrete mutual information."""
  num_factors = ys.shape[0]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
  return h


def compute_importance_gbt(x_train, y_train):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  train_loss = []
  for i in range(num_factors):
    model = GradientBoostingClassifier()
    model.fit(x_train.T, y_train[i, :])
    importance_matrix[:, i] = np.abs(model.feature_importances_)
    train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
  return importance_matrix, np.mean(train_loss)


def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

  return np.sum(per_code*code_importance)



def completeness_per_code(importance_matrix):
  """Compute completeness of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)


def _compute_dci(z, y):
  """Computes score based on both training and testing codes and factors."""
  scores = {}
  importance_matrix, train_err = compute_importance_gbt(
      z, y)
  assert importance_matrix.shape[0] == z.shape[0]
  assert importance_matrix.shape[1] == y.shape[0]
  informativeness_train = train_err
  disentanglement_ = disentanglement(importance_matrix)
  completeness_ = completeness(importance_matrix)
  return informativeness_train,disentanglement_,completeness_

def _histogram_discretize(target, num_bins):
  """Discretization based on histograms."""
  discretized = np.zeros_like(target)
  for i in range(target.shape[0]):
    discretized[i, :] = np.digitize(target[i, :], np.histogram(
        target[i, :], num_bins)[1][:-1])
  return discretized
