import numpy as np
import warnings
import math
warnings.filterwarnings('ignore')
# The script is meant to solve the object ODE for different conditions
# and report the object output x_max as function of the input (v0, alpha, h, cd, r, m, tho) in a file
# This data will be used to perform uncertainty quantification


def sol_ex(t, param_initial_conditions):
    g = 9.81
    v0 = param_initial_conditions["v0"]
    alpha = param_initial_conditions["alpha"]
    h = param_initial_conditions["h"]
    return v0*np.cos(alpha)*t, -0.5*g*t**2 + v0*np.sin(alpha)*t + h


def rhs(variables, t, param):
    g = param["gravity"]
    rho = param["density"]
    r = param["radius"]
    m = param["mass"]
    cd = param["drag_coefficient"]
    # Definition of RHS function: y'=f(y,t)
    v_square = variables[2]**2 + variables[3]**2
    f1 = variables[2]
    f2 = variables[3]
    f3 = -0.5*rho*v_square*cd*np.pi*r**2/m
    f4 = -g

    return np.array([f1, f2, f3, f4])


def solve_object_ODE(delta_t, param_initial_conditions, param_equation):
    # Define Initial Conditions
    x0 = 0
    y0 = param_initial_conditions["h"]
    vx0 = param_initial_conditions["v0"] * np.cos(param_initial_conditions["alpha"])
    vy0 = param_initial_conditions["v0"] * np.sin(param_initial_conditions["alpha"])

    # Assemble IC into an array
    vec_0 = np.array([x0, y0, vx0, vy0])

    # Sequence of x, y and time values of numerical and analytical solution of ODE
    x = []
    y = []
    time = []

    x_ex = []
    y_ex = []

    # First element of the time sequence
    t_old = 0
    vec_old = vec_0

    x.append(vec_old[0])
    y.append(vec_old[1])
    time.append(t_old)

    sol = sol_ex(t_old, param_initial_conditions)
    x_ex.append(sol[0])
    y_ex.append(sol[1])

    # Solve the ODE until y>0 (the object reaches the ground) for each step
    while vec_old[1] > 0:
        t_new = t_old + delta_t
        vec_new = step_forward(rhs, vec_old, t_old, delta_t, param_equation)
        x.append(vec_new[0])
        y.append(vec_new[1])
        time.append(t_new)
        vec_old = vec_new
        t_old = t_new
        sol = sol_ex(t_new, param_initial_conditions)
        x_ex.append(sol[0])
        y_ex.append(sol[1])

    return np.array(time), np.array(x), np.array(y)


def solve_for_input_data(collocation_points, index, delta_t, multilevel=False):
    params_eq = {
        "gravity": 9.81,
        "density": collocation_points["density"][index],
        "radius": collocation_points["radius"][index],
        "mass": collocation_points["mass"][index],
        "drag_coefficient": collocation_points["drag_coefficient"][index]
    }
    param_initial_conditions = {
        "h": collocation_points["h"][index],
        "alpha": collocation_points["alpha"][index],
        "v0": collocation_points["v0"][index],
    }

    time_, x, y = solve_object_ODE(delta_t, param_initial_conditions, params_eq)
    max_x = (x[-1] - x[-2]) / (y[-1] - y[-2]) * (-y[-2]) + x[-2]
    max_x_ml = 0
    if multilevel:
        time_ml, x_ml, y_ml = solve_object_ODE(2*delta_t, param_initial_conditions, params_eq)
        max_x_ml = (x_ml[-1] - x_ml[-2]) / (y_ml[-1] - y_ml[-2]) * (-y_ml[-2]) + x_ml[-2]
    if max_x_ml < 0 or max_x_ml > 50 or math.isnan(max_x_ml) or max_x <0 or max_x > 50 or math.isnan(max_x):
        max_x_ml = 0
        max_x = 0

    if multilevel:
        if max_x != 0 and max_x_ml != 0:
            return [max_x, max_x_ml]
        else:
            print("Multilevel: bad Values")
    else:
        if max_x != 0:
            return max_x
        else:
            print("Singlelevel: bad Values")


def step_forward(rhs_func, y_old, t_old, delta_t, parameter_equation):
    k1 = delta_t * rhs_func(y_old, t_old, parameter_equation)
    k2 = delta_t * rhs_func(y_old + k1/2, t_old + delta_t/2, parameter_equation)
    k3 = delta_t * rhs_func(y_old + k2/2, t_old + delta_t/2, parameter_equation)
    k4 = delta_t * rhs_func(y_old + k3  , t_old + delta_t, parameter_equation)
    #return y_old + (k1 + 2*k2 + 2*k3 + k4)/6
    return y_old + k1
    #return y_old + k2