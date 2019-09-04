import os
import subprocess

python = os.environ['PYTHON36']
keyword = "parab"
variable = "x_max"
point = "random"
model = "NET"

if keyword == "parab":
    case_study = "Parabolic"
elif keyword == "airf":
    case_study = "Airfoil"
else:
    raise ValueError("Chose one between parab and airf")

base_path = "CaseStudies/"+case_study+"/Models/"
directories_model = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

for directory in directories_model:
    if "Depth_" in directory and variable in directory:
        print("\n")
        print(directory)
        arguments = list()
        arguments.append(str(directory))
        arguments.append(str(case_study))
        arguments.append(str(point))
        arguments.append(str(model))
        p = subprocess.Popen([python, "MultiLevModel.py"] + arguments)
        p.wait()

