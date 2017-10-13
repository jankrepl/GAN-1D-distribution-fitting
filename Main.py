# Imports
from utils import *

# PARAMETERS

# True distribution
param_td = {'parametric_family': 'norm',
            'true_parameters': {'loc': 0, 'scale': 1},
            }

# Noise distribution
param_nd = {'parametric_family': 'uniform',
            'true_parameters': {'loc': 0, 'scale': 1}
            }

# Computational Graph
param_cg = {'arch_D': [120, 80, 60],
            'arch_G': [80, 30],
            'OS_label_smoothing': [False, 0.99],
            'optimizer_parameter_D': 0.001,
            'optimizer_parameter_G': 0.001,
            }

# Pre-training
param_p = {'param_nd': param_nd,
           'param_td': param_td,
           'iter': 8000,
           'batch_size': 32,
           'summary_frequency': 10,
}

# Training
param_t = {'param_nd': param_nd,
           'param_td': param_td,
           'sort_samples': True,
           'iter': 100000,
           'batch_size': 20,  # make higher number should be able to explain tails
           'steps_D': 1,
           'steps_G': 1,
           'summary_frequency': 10,
           'visualizer_frequency': 1000,
           'visualizer_samples': 1000,
           }


# ALGORITHM
my_cg = Computational_graph(**param_cg)
my_pretraining = PreTrainer(my_cg, **param_p)
my_training = Trainer(my_cg, **param_t)

