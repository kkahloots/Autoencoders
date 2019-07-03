
from graphs.AE_graph import AEGraph
from graphs.VAE_graph import VAEGraph
from graphs.BetaVAE_graph import BVAEGraph
from graphs.BetaTCVAE_graph import BTCVAEGraph
from graphs.AnnealedVAE_graph import AnnealedVAEGraph
from graphs.DIPIIVAE_graph import DIPIIVAEGraph
from graphs.DIPIVAE_graph import DIPIVAEGraph
from graphs.BayesianAE_graph import BayesianAEGraph


from utils.constants import Models

def Factory(configuration):
    print('building {} graph ... '.format(configuration['model_type']))

    if configuration['model_type'] == Models.AE:
        return AEGraph(configuration)
    elif configuration['model_type'] == Models.VAE:
        return VAEGraph(configuration)
    elif configuration['model_type'] == Models.BTCVAE:
        return BTCVAEGraph(configuration)
    elif configuration['model_type'] == Models.BVAE:
        return BVAEGraph(configuration)
    elif configuration['model_type'] == Models.AnnealedVAE:
        return AnnealedVAEGraph(configuration)
    elif configuration['model_type'] == Models.DIPIVAE:
        return DIPIVAEGraph(configuration)
    elif configuration['model_type'] == Models.DIPIIVAE:
        return DIPIIVAEGraph(configuration)
    elif configuration['model_type'] == Models.BayVAE:
        return BayesianAEGraph(configuration)