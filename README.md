# A Multi-level procedure for enhancing accuracy of machine learning algorithms
Repository to reproduce the experiments in the paper "A Multi-level procedure for enhancing accuracy of machine learning algorithms" 

## Reproduce the plots for the convergence study of generalization error
In the project foler, run:

    python CollectDataBound.py

## Reproduce the plots of the sensitivity study
In the project foler, run:

    python SensitivityStudy.py

	 
## Reproduce the plots of UQ
In the project foler, run:

     python FinalComparison.py

## Train your own model
### Choice of the neural network hyperparameters
Perform the ensemble training model of https://arxiv.org/abs/1903.03040

In the project foler, run:

     python search_network_cluster.py "keyword" "variable_name" "samples" "loss" "level_single" "level_c" "level_f" "selection_method" "validation_size" "string_norm" "scaler" "search_folder" "point"
     
**keyword**
Choose the problem of interest (Airfoil problem or Projectile Motion) and if the target of the ensemble training is the variable **map** or the **detail**:
- airf
- airf_diff
- parab
- prab_diff
**variable_name**
**samples**
**loss**
**level_single**
**level_c**
**level_f**
**selection_method**
**validation_size**
**string_norm**
**scaler**
### Python Dependencies for plotting
- matplotlib   2.2.3
- numpy        1.15.4
- pandas       0.23.4
- scipy        1.1.0
- seaborn      0.9.0


### Python Dependencies for the model assembling
- matplotlib   2.2.3
- numpy        1.15.4
- pandas       0.23.4
- scipy        1.1.0
- seaborn      0.9.0
