:: search_network keyword variable_name samples loss level_single level_c level_f
:: keyword = "parab" , "parab_diff", "shock" , "shock_diff", "airfoil" , "airf_diff"
:: variable_name = "Lift", "Drag"
:: variable_name = "x_max"
:: loss = "mse", "mea"
SET KEYWORD="parab"
SET VARIABLE_NAME="x_max"
SET SAMPLES="256"
SET LEVEL_SINGLE="0"
SET LEVEL_C="0"
SET LEVEL_F="2"
SET NORMALIZE="true"
SET SCALER="m"
SET POINT="sobol"


%python36% GaussianProcess.py %KEYWORD% %VARIABLE_NAME% %SAMPLES% %LEVEL_SINGLE% %LEVEL_C% %LEVEL_F% %NORMALIZE% %SCALER% %POINT%
