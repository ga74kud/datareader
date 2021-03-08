import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mlt
from matplotlib import cm
class causal_prob(object):
    def __init__(self, **kwargs):
        self.ax=None
        self.prob={"mu": None, "Sigma": None}

    '''
        set mean vector of Gaussian distribution
    '''
    def set_mu(self, mu):
        self.prob["mu"]=mu

    '''
        set Covariance matrix
    '''
    def set_Sigma(self, Sigma):
        self.prob["Sigma"]=Sigma
    '''
        get the probabilities for the coordinates
    '''
    def get_probabilities_position(self, coordinates):
        erg=[]
        for wlt in coordinates:
            x = np.matrix([wlt[0]], wlt[1])
            erg.append(np.float(self.multivariate_gaussian_distribution(x, self.prob["mu"], self.prob["Sigma"])))
        return erg
    def mahalabonis_dist(self, x, mu, Sigma):
        return -0.5*np.transpose(x-mu)*np.linalg.inv(Sigma)*(x-mu)
    def multivariate_gaussian_distribution(self, x, mu, Sigma):
        factor_A=1/np.sqrt((2*np.pi)**2*np.linalg.det(Sigma))
        factor_B=np.exp(self.mahalabonis_dist(x, mu, Sigma))
        erg=factor_A*factor_B
        return erg[0]
    def visualize_multivariate_gaussian(self):
        fig = plt.figure()
        self.ax = fig.add_subplot()
        Z=np.zeros((np.size(self.X, 0), np.size(self.X, 1)))
        for idx_A in range(0, np.size(self.X, 0)):
            for idx_B in range(0, np.size(self.X, 1)):
                x = np.matrix([[self.X[idx_A, idx_B]], [self.Y[idx_A, idx_B]]])
                Z[idx_A, idx_B]=self.multivariate_gaussian_distribution(x, self.prob["mu"], self.prob["Sigma"])
        #self.ax.plot_surface(self.X, self.Y, Z,  cmap='viridis',
        #               linewidth=0, antialiased=False, alpha=.3)

        self.ax.contour(self.X, self.Y, Z, 10, lw=3, cmap="autumn_r", linestyles="solid", offset=0)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
    def show(self):
        plt.grid()
        plt.show()
    def set_fixed_domain(self):
        N = 16
        X = np.linspace(-3, 3, N)
        Y = np.linspace(-3, 4, N)
        self.X, self.Y = np.meshgrid(X, Y)
        self.x_rav = np.ravel(X)
        self.y_rav = np.ravel(Y)
    def plot_arrow(self, mu, w, v):
        Q=self.ax.quiver(mu[0], mu[1],  w*v[0,0], w*v[0, 1],  color="red", linewidth=2,
                         alpha=.5)
    def kullback_leibler(self, mu_B, Sigma_B):
        k=len(self.prob["mu"])
        sum_A=np.trace(np.linalg.inv(Sigma_B)*self.prob["Sigma"])
        dif_mu=mu_B-self.prob["mu"]
        sum_B=np.transpose(dif_mu)*np.linalg.inv(Sigma_B)*dif_mu-2
        sum_C=np.log(np.linalg.det(Sigma_B)/np.linalg.det(self.prob["Sigma"]))
        return 0.5*(sum_A+sum_B+sum_C)

    def plot_eigen_vectors_Sigma(self):
        w, v = np.linalg.eigh(self.prob["Sigma"])
        for idx, wlt in enumerate(v):
            self.plot_arrow(self.prob["mu"], w[idx], wlt)

def start_visualize():
    fig=plt.figure()
    ax=fig.add_subplot()


def get_eigen_Sigma(single_sigma):
    w,v=np.linalg.eigh(single_sigma)
    return w,v
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



new_mu=conditioned_mu(mu, Sigmas, np.array([8, 8]))
print(new_mu)
new_Sigma=conditioned_Sigma(Sigmas)
print(new_Sigma)
new_pi=conditioned_pi(mu, Sigmas, np.array([8, 8]), pi)
print(new_pi)

mu = np.matrix([[0.], [0.]])
Sigma = np.matrix([[1., .3], [0.3, 1.]])

obj_causal = causal_prob()
obj_causal.set_fixed_domain()
obj_causal.set_mu(mu)
obj_causal.set_Sigma(Sigma)

obj_causal.visualize_multivariate_gaussian()
obj_causal.plot_eigen_vectors_Sigma()
obj_causal.show()

