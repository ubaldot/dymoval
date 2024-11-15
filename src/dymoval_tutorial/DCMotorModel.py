"""
This is the model that we want to validate.
"""

import control as ct
from dymoval_tutorial.virtual_lab import get_ss_matrices, Ts


# ====== Model to be validated =====================
# This is the model to be validated. The parameters are perturbed.
# We simulate this model in the tutorial


L_model = 1.2e-3
R_model = 1.4
J_model = 6.5e-5
b_model = 1.08e-4
K_model = 0.102

# Nominal values
# L_model = 1e-3
# R_model = 1.0
# J_model = 5e-5
# b_model = 1e-4
# K_model = 0.1

A_model, B_model, C_model, D_model = get_ss_matrices(
    R=R_model, L=L_model, K=K_model, J=J_model, b=b_model
)

DCMotor_model_ct = ct.ss(A_model, B_model, C_model, D_model)
DCMotor_model_dt = ct.sample_system(DCMotor_model_ct, Ts, method="zoh")
