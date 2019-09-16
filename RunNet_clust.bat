:: search_network keyword variable_name samples loss level_single level_c level_f
:: keyword = "parab" , "parab_diff", "shock" , "shock_diff", "airfoil" , "airf_diff"
:: variable_name = "Lift", "Drag"
:: variable_name = "x_max"
:: loss = "mse", "mea"
SET KEYWORD="parab"
SET VARIABLE_NAME="x_max"
SET SAMPLES="200"
SET LOSS="mse"
SET LEVEL_SINGLE="0"
SET LEVEL_C="0"
SET LEVEL_F="2"
SET SELECTION="validation_loss"
SET VALIDATION="0.50"
SET NORMALIZE="true"
SET SCALER="m"
SET FOLDER="search_net_test"
SET POINT="random"
ECHO OFF

%python36% search_network_cluster.py %KEYWORD% %VARIABLE_NAME% %SAMPLES% %LOSS% %LEVEL_SINGLE% %LEVEL_C% %LEVEL_F% %SELECTION% %VALIDATION% %NORMALIZE% %SCALER% %FOLDER% %POINT%