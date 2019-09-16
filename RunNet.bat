:: search_network keyword variable_name samples loss level_single level_c level_f
:: keyword = "parab" , "parab_diff", "shock" , "shock_diff", "airfoil" , "airf_diff"
:: variable_name = "Lift", "Drag"
:: variable_name = "x_max"
:: loss = "mse", "mea"
SET KEYWORD="shock_diff"
SET VARIABLE_NAME="pressure"
SET SAMPLES="128"
SET LOSS="mse"
SET LEVEL_SINGLE="0"
SET LEVEL_C="0"
SET LEVEL_F="1"
SET SELECTION="validation_loss"
SET VALIDATION="0.15"
SET NORMALIZE="true"
SET SCALER="m"
ECHO OFF

%python36% search_network.py %KEYWORD% %VARIABLE_NAME% %SAMPLES% %LOSS% %LEVEL_SINGLE% %LEVEL_C% %LEVEL_F% %SELECTION% %VALIDATION% %NORMALIZE% %SCALER%