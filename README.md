# A Multi-level procedure for enhancing accuracy of machine learning algorithms
Repository to reproduce the experiments in the paper "A Multi-level procedure for enhancing accuracy of machine learning algorithms" 

If you are on Windows OS, make sure to save your python executable path in an **evironnment variable** as **PYTHON36**. Then, the **python** variable below should be replaced by **%PYTHON36%**.
If you are on Linux OS, make sure to import your python exectuable path in your .bashrc as **python3**. Then, the **python** variable below should be replaced by **python3**.
This is important for the section **Ensemble training for the selection of the models hyperparameters**
## Reproduce the plots for the convergence study of generalization error
In the project folder, run:

    python CollectDataBound.py

The resulting plots will be stored in the folder **Images**

## Reproduce the plots of the sensitivity study
In the project folder, run:

    python SensitivityStudy.py

	 
The resulting plots will be stored in the folder **Images**

## Reproduce the plots of UQ
In the project folder, run:

     python FinalComparison.py

The resulting plots will be stored in the folder **Images**

## Ensemble training for the selection of the models hyperparameters (Neural Network and Gaussian Process Regressors)
### Choice of the neural network hyperparameters
Perform the ensemble training model of https://arxiv.org/abs/1903.03040 (https://github.com/kjetil-lye/learning_airfoils
)


In the project folder, run:

     python search_network_cluster.py "keyword" "variable_name" "samples" "loss" "level_single" "level_c" "level_f" "selection_method" "validation_size" "string_norm" "scaler" "search_folder" "point" "cluster"
     
Once the training is complete, run:

     python GetBestPerformingConf.py "keyword" "search_folder"
     
     
**keyword**: choose the problem of interest (Airfoil problem or Projectile Motion) and if the target of the ensemble training is the variable **map** or the **detail**:
- *airf*, for the selected **observable** at given grid resolution of the Airfoil Problem
- *airf_diff*,  for the **selected observable detail** between two grid resolutions of the Airfoil Problem
- *parab, for* the selected **observable** at given grid resolution of the Projectile Motion Problem
- *prab_diff*, for the **selected observable detail** between two grid resolutions of the Airfoil Problem
```diff
! Note: to run  GetBestPerformingConf.py airf and airf_diff or parab and parab_diff can be chosen indifferently for the airfoil and projectile motion problem.
```

**variable_name**: ame of the observables:
- *x_max*, for the Projectile Motion example
- *Lift*, for the Airfoil example
- *Drag*, for the Airfoil example

**samples**: number of training samples for the choice of the model hyperparameters

**loss**: loss function:
- *mae*, Mean Absolute Error
- *mse*, Mean Squared Error

**level_single**: resolution mesh to approximate the **observable** (the input will be ingored if **keyword** contains 'diff' ):
- One value between *0 and 4* for the Airfoil problem, 4 for the finest resolution, 0 for the coarsest
- One value between *0 and 6* for the Projectile Motion problem, 6 for the finest resolution, 0 for the coarsest

**level_c**: coarser mesh reoslution to approximate the **detail** (the input will be ingored if **keyword** does not contain "diff" )
- One value between *0 and 4* for the Airfoil problem, 4 for the finest resolution, 0 for the coarsest
- One value between *0 and 6* for the Projectile Motion problem, 6 for the finest resolution, 0 for the coarsest

**level_f**: finer mesh reoslution to approximate the **detail** (the input will be ingored if **keyword** does not contain "diff" )
- One value between *0 and 4* for the Airfoil problem, 4 for the finest resolution, 0 for the coarsest
- One value between *0 and 6* for the Projectile Motion problem, 6 for the finest resolution, 0 for the coarsest

```diff
! Note 1: **level_f** > **level_c**
```


```diff
! Note 2: if you perform the ensembel training for the selection of the hyperpaarmeter of the detail, assign in any case a value to level_single and if you perform the ensembel training for the selection of the hyperpaarmeter of the observable , assign in any case a value to level_c and level_f
```

**selection_method**: selection criterion for the best performin configuration:
- *validation_loss*: value of the loss function on the validation set
- *train_loss*: value of the loss function on the training set
- *wasserstein_train*: value of the Waserstain distance on the training set

**validation_size**: size of the validation set (value between 0 and 1).
- Give value 1 if **selection_method**=*train_loss* or **selection_method**=*wasserstein_train*

**string_norm**: choose if normalize the data or not
- *true*: normalize the data
- *false*: do not normalize the data

**scaler**: type of scaler for data normalization (if **string_norm**=*false*, this input will be ingored in the code)
- *s*: standard scaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- *m*: minmax scaler (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

**search_folder**: name of the folder that will contain the folder corresponding to different confiugrations of the design parameters

**point**: class of points used to generate the data (For the airfoil problem only Sobol samples are available. Therefore, this input will be ignored if the airfoil problem is chosen.)
- *sobol*: low discrepansy sobol points
- *random*: unniformely distributed random points

**cluster**: option to run the code on LSF HPC (Euler ETH)
- *true*
- *false*
Set **cluster==false** if you do not have access to a LSF cluster


### Choice of the Gaussian Process covariance function

In the project folder, run:

     python GP_model_selection.py "keyword" "variable_name" "samples" "level_single" "level_c" "level_f" "validation_size" "string_norm" "scaler" "point"

## Training of the multi-level model
**In preparation..**

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
