import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

def is_symmetric(x):
    return (x.transpose() == x).all()

def is_pos_definite(x):
    return np.all(np.linalg.eigvals(x) > 0)

def conditioned_mu(mu, Sigmas, x_h):
    erg={new_list: [] for new_list in range(len(Sigmas))}
    for wlt in erg:
        dif_vec=x_h-mu[wlt]["mu_h"]
        added_mu_A=np.matmul(Sigmas[wlt]["Sigma_fh"],np.linalg.inv(Sigmas[wlt]["Sigma_hh"]))
        added_mu=np.matmul(added_mu_A, dif_vec)
        erg[wlt]=mu[wlt]["mu_f"]+added_mu
    return erg

def conditioned_Sigma(Sigmas):
    erg = {new_list: [] for new_list in range(len(Sigmas))}
    for wlt in erg:
        a=np.matmul(Sigmas[wlt]["Sigma_fh"], np.linalg.inv(Sigmas[wlt]["Sigma_hh"]))
        second_mat=np.matmul(a, Sigmas[wlt]["Sigma_hf"])
        erg[wlt]=Sigmas[wlt]["Sigma_ff"]-second_mat
    return erg

def conditioned_pi(mu, Sigmas, x_h, pi):
    erg={new_list: [] for new_list in range(len(Sigmas))}
    vart = [multivariate_normal(mean=mu[qrt]["mu_h"], cov=Sigmas[qrt]["Sigma_hh"]).pdf(x_h) for qrt in range(0, len(erg))]
    for wlt in erg:        # test=var.pdf([0, 0])
        erg[wlt]=vart[wlt]/np.sum(vart)
    return erg


pi={0: .8, 1: .2}
mu={0:
        {"mu_h": np.array([1, 2]),
        "mu_f": np.array([2, 3])},
    1:
        {"mu_h": np.array([3, 4]),
        "mu_f": np.array([2, 3])}}

Sigmas={0:
            {"Sigma_hh": np.array([[1, 0], [0, 1]]),
            "Sigma_hf": np.array([[1, 0], [0, 1]]),
            "Sigma_fh": np.array([[1, 0], [0, 1]]),
            "Sigma_ff": np.array([[1, 0], [0, 1]])},
        1:
           {"Sigma_hh": np.array([[1, 0], [0, 1]]),
            "Sigma_hf": np.array([[1, 0], [0, 1]]),
            "Sigma_fh": np.array([[1, 0], [0, 1]]),
            "Sigma_ff": np.array([[1, 0], [0, 1]])}}
# bool_A=is_pos_definite(Sigmas["Sigma_hh"])
# print(bool_A)
# bool_B=is_symmetric(Sigmas["Sigma_hh"])
# print(bool_B)


new_mu=conditioned_mu(mu, Sigmas, np.array([8, 8]))
print(new_mu)
new_Sigma=conditioned_Sigma(Sigmas)
print(new_Sigma)
new_pi=conditioned_pi(mu, Sigmas, np.array([8, 8]), pi)
print(new_pi)


