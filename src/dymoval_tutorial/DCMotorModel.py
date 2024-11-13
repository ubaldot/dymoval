import control as ct
import numpy as np

# Discretization
Ts = 0.01  # s


def get_ss_matrices(
    R: float, L: float, K: float, J: float, b: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A = np.array([[-R / L, 0, -K / L], [0, 0, 1], [K / J, 0, -b / J]])
    B = np.array([[1.0 / L], [0], [0]])
    C = np.array([[1.0, 0, 0], [0, 0, 9.5493]])
    D = np.array([[0], [0]])

    return A, B, C, D


# El motor SS representation
# define motor parameters
# Nominal values
L = 1e-3
R = 1.0
J = 5e-5
b = 1e-4
K = 0.1


# States:
# x1: current
# x2: position
# x3: speed

A, B, C, D = get_ss_matrices(R=R, L=L, K=K, J=J, b=b)
DCMotor_nominal_ct = ct.ss(A, B, C, D)
DCMotor_nominal_dt = ct.sample_system(DCMotor_nominal_ct, Ts, method="zoh")

# ======= Model values =================
# L_model = 1e-3
# R_model = 1.08
# J_model = 5.5e-5
# b_model = 1.5e-4
# K_model = 0.103


L_model = 2e-3
R_model = 1.28
J_model = 6.5e-5
b_model = 1.8e-4
K_model = 0.11

A_model, B_model, C_model, D_model = get_ss_matrices(
    R=R_model, L=L_model, K=K_model, J=J_model, b=b_model
)
DCMotor_model_ct = ct.ss(A_model, B_model, C_model, D_model)
DCMotor_model_dt = ct.sample_system(DCMotor_model_ct, Ts, method="zoh")
