import tensorflow as tf
import utils.constants as const

## ------------------- LOSS: EXPECTED LOWER BOUND ----------------------
# reconstruction loss
def get_ell( x, x_recons):
    """
    Returns the expected log-likelihood of the lower bound.
    For this we use a bernouilli LL.
    """
    # p(x|w)
    return - tf.reduce_sum((x) * tf.log(x_recons + const.epsilon) +
                           (1 - x) * tf.log(1 - x_recons + const.epsilon), 1)

def get_kl(mu, log_var):
    """
    d_kl(q(z|x)||p(z)) returns the KL-divergence between the prior p and the variational posterior q.
    :return: KL divergence between q and p
    """
    # Formula: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return - 0.5 * tf.reduce_sum( 1.0 + 2.0 * log_var - tf.square(mu) - tf.exp(2.0 * log_var), 1)


def anneal(c_max, step, iteration_threshold):
  """Anneal function for anneal_vae (https://arxiv.org/abs/1804.03599).
  Args:
    c_max: Maximum capacity.
    step: Current step.
    iteration_threshold: How many iterations to reach c_max.
  Returns:
    Capacity annealed linearly until c_max.
  """
  return tf.math.minimum(c_max * 1.,
                         c_max * 1. * tf.to_float(step) / iteration_threshold)


def get_QP_kl(meanQ, log_varQ, meanP, log_varP):
    """
    KL[Q || P] returns the KL-divergence between the prior p and the variational posterior q.
    :param meanQ: vector of means for q
    :param log_varQ: vector of log-variances for q
    :param meanP: vector of means for p
    :param log_varP: vector of log-variances for p
    :return: KL divergence between q and p
    """
    #meanQ = posterior_mean
    #log_varQ = posterior_logvar
    #meanP = prior_mean
    #log_varP = prior_logvar

    return - 0.5 * tf.reduce_sum(
        log_varP - log_varQ + (tf.square(meanQ - meanP) / tf.exp(log_varP)) + tf.exp(log_varQ - log_varP) - 1)