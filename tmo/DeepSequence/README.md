# DeepSequence
DeepSequence is a generative, unsupervised latent variable model for biological sequences. Given a multiple sequence alignment as an input, it can be used to predict accessible mutations, extract quantitative features for supervised learning, and generate libraries of new sequences satisfying apparent constraints. It models higher-order dependencies in sequences as a nonlinear combination of constraints between subsets of residues. For more information, check out the paper on [biorxiv](https://www.biorxiv.org/content/early/2017/12/18/235655.1) and the examples below.

For ease of analysis, we advise that alignments be generated with the [EVcouplings package](https://github.com/debbiemarkslab/EVcouplings), though any sequence alignment can be used.

Codebase is compatible with Python 2.7 and Theano 1.0.1. For GPU-enabled computation, CUDA will have to be installed separately. See INSTALL for more details.

## Examples
For reasonable training time, we advise training DeepSequence on a GPU:

    THEANO_FLAGS='floatX=float32,device=cuda' python run_svi.py

However, it can be run on the CPU with:

    python run_svi.py

Other usage examples and features of the analysis are available in iPython notebooks in the examples subfolder.
