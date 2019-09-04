import numpy as np

density_mean = 1.225
radius_mean = 0.23
drag_mean = 0.1
mass_mean = 0.145
h_mean = 1
alpha_mean = np.pi/6
v0_mean = 25
epsilon = 0.1
samples_vec_ML = [168, 192, 216, 240, 288, 336, 408]
samples_vec_SL = [72, 96, 144, 168, 192, 216, 288]
sample_vec_MLMC = [10, 20, 40, 80, 160, 336, 672, 1360, 2720]
sample_vec_MC = [10, 20, 40, 80, 160, 336, 672, 1360]
parameters_name = ["density", "radius", "drag_coefficient", "mass", "h", "alpha", "v0"]

convergence_rate = {
        "x_max": [0.81],
        "strength": [1.2],
        "Lift": [0.82],
        "Drag": [0.81],
        "y": [1.246]
}

parameter_grid_parab = {
        "regularization_parameter": [0.000001],
        "kernel_regularizer": ["L2"],
        "learning_rate": [0.01],
        "hidden_layers": [5],
        "neurons": [10],
        "dropout_value": [0]
    }
parameter_grid_parab_diff = {
        "regularization_parameter": [0.000001],
        "kernel_regularizer": ["L2"],
        "learning_rate": [0.01],
        "hidden_layers": [5],
        "neurons": [10],
        "dropout_value": [0]
    }
'''
parameter_grid_shock = {
        "regularization_parameter": [7.8e-6],
        "kernel_regularizer": ["L2"],
        "learning_rate": [0.01],
        "hidden_layers": [4],
        "neurons": [10],
        "dropout_value": [0]
    }
'''

'''
# Variable EK3 
parameter_grid_shock = {
        "regularization_parameter": [1e-5],
        "kernel_regularizer": ["L2"],
        "learning_rate": [0.01],
        "hidden_layers": [4],
        "neurons": [10],
        "dropout_value": [0]
    }
parameter_grid_shock_diff = {
        "regularization_parameter": [1e-5],
        "kernel_regularizer": ["L2"],
        "learning_rate": [0.01],
        "hidden_layers": [7],
        "neurons": [10],
        "dropout_value": [0]
    }
'''
# Variable strength
# for sod shock
'''
parameter_grid_shock = {
        "regularization_parameter": [0.000001],
        "kernel_regularizer": ["L2"],
        "learning_rate": [0.001],
        "hidden_layers": [4],
        "neurons": [10],
        "dropout_value": [0]
    }
parameter_grid_shock_diff = {
        "regularization_parameter": [5.000000e-07],
        "kernel_regularizer": ["L2"],
        "learning_rate": [0.01],
        "hidden_layers": [7],
        "neurons": [10],
        "dropout_value": [0]
    }
'''
# variable pressure at -3.5 for lax-sod
parameter_grid_shock = {
        "regularization_parameter": [0.000001],
        "kernel_regularizer": ["L1"],
        "learning_rate": [0.01],
        "hidden_layers": [4],
        "neurons": [10],
        "dropout_value": [0]
    }
parameter_grid_shock_diff = {
        "regularization_parameter": [0.000005],
        "kernel_regularizer": ["L2"],
        "learning_rate": [0.01],
        "hidden_layers": [9],
        "neurons": [10],
        "dropout_value": [0]
    }


# Variable: Lift
parameter_grid_airf = {
        "regularization_parameter": [5e-07],
        "kernel_regularizer": ["L2"],
        "learning_rate": [0.01],
        "hidden_layers": [8],
        "neurons": [12],
        "dropout_value": [0]
    }
parameter_grid_airf_diff = {
        "regularization_parameter": [5e-7],
        "kernel_regularizer": ["L2"],
        "learning_rate": [0.01],
        "hidden_layers": [8],
        "neurons": [12],
        "dropout_value": [0]
    }


# Variable: Drag
parameter_grid_airf_diff_drag_01 = {
        "regularization_parameter": [5e-7],
        "kernel_regularizer": ["L2"],
        "learning_rate": [0.001],
        "hidden_layers": [12],
        "neurons": [12],
        "dropout_value": [0]
    }
parameter_grid_airf_diff_drag_12 = {
        "regularization_parameter": [5e-7],
        "kernel_regularizer": ["L1"],
        "learning_rate": [0.001],
        "hidden_layers": [8],
        "neurons": [12],
        "dropout_value": [0]
    }
parameter_grid_airf_diff_drag_23 = {
        "regularization_parameter": [5e-6],
        "kernel_regularizer": ["L1"],
        "learning_rate": [0.01],
        "hidden_layers": [8],
        "neurons": [8],
        "dropout_value": [0]
    }
parameter_grid_airf_diff_drag_34 = {
        "regularization_parameter": [1e-6],
        "kernel_regularizer": ["L1"],
        "learning_rate": [0.001],
        "hidden_layers": [12],
        "neurons": [8],
        "dropout_value": [0]
    }
parameter_grid_airf_diff_drag_02 = {
        "regularization_parameter": [1e-6],
        "kernel_regularizer": ["L2"],
        "learning_rate": [0.01],
        "hidden_layers": [8],
        "neurons": [8],
        "dropout_value": [0]
    }
parameter_grid_airf_diff_drag_24 = {
        "regularization_parameter": [5e-7],
        "kernel_regularizer": ["L1"],
        "learning_rate": [0.001],
        "hidden_layers": [12],
        "neurons": [20],
        "dropout_value": [0]
    }
parameter_grid_airf_diff_drag_04 = {
        "regularization_parameter": [1e-5],
        "kernel_regularizer": ["L1"],
        "learning_rate": [0.01],
        "hidden_layers": [12],
        "neurons": [20],
        "dropout_value": [0]
    }
parameter_grid_airf_diff_drag_03 = {
        "regularization_parameter": [1e-6],
        "kernel_regularizer": ["L1"],
        "learning_rate": [0.01],
        "hidden_layers": [8],
        "neurons": [8],
        "dropout_value": [0]
    }
parameter_grid_airf_drag = {
        "regularization_parameter": [5e-7],
        "kernel_regularizer": ["L2"],
        "learning_rate": [0.005],
        "hidden_layers": [8],
        "neurons": [8],
        "dropout_value": [0]
    }
