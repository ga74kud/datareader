import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mlt
from matplotlib import cm
from sklearn.mixture import GaussianMixture
class causal_prob(object):
    def __init__(self, **kwargs):
        self.ax=None
        self.prob={"mu": None, "Sigma": None}
        self.set_fixed_domain()
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
                for qrt in range(0, len(self.prob["mu"])):
                    Z[idx_A, idx_B]+=self.multivariate_gaussian_distribution(x, self.prob["mu"][qrt], self.prob["Sigma"][qrt])
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
        X = np.linspace(-4, 4, N)
        Y = np.linspace(-4, 4, N)
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
    def get_eigen_Sigma(self, single_sigma):
        w,v=np.linalg.eigh(single_sigma)
        return w,v
    def is_symmetric(self, x):
        return (x.transpose() == x).all()

    def is_pos_definite(self,x):
        eig_vals=np.linalg.eigvals(x)
        return np.all(eig_vals > 0)

    def conditioned_mu(self, mu, Sigmas, x_h):
        erg={new_list: [] for new_list in range(len(Sigmas))}
        for wlt in erg:
            dif_vec=x_h-mu[wlt]["h"]
            added_mu_A=np.matmul(Sigmas[wlt]["fh"],np.linalg.inv(Sigmas[wlt]["hh"]))
            added_mu=np.matmul(added_mu_A, dif_vec)
            erg[wlt]=mu[wlt]["f"]+added_mu
        return erg

    def conditioned_Sigma(self, Sigmas):
        erg = {new_list: [] for new_list in range(len(Sigmas))}
        for wlt in erg:
            a=np.matmul(Sigmas[wlt]["fh"], np.linalg.inv(Sigmas[wlt]["hh"]))
            second_mat=np.matmul(a, Sigmas[wlt]["hf"])
            erg[wlt]=Sigmas[wlt]["ff"]-second_mat
        return erg

    def conditioned_pi(self, mu, Sigmas, x_h, pi):
        erg={new_list: [] for new_list in range(len(Sigmas))}
        vart = [multivariate_normal(mean=mu[qrt]["h"], cov=Sigmas[qrt]["hh"]).pdf(x_h) for qrt in range(0, len(erg))]
        for wlt in erg:        # test=var.pdf([0, 0])
            erg[wlt]=vart[wlt]/np.sum(vart)
        return erg

    def set_mu_Sigmas(self, pi, mu, Sigmas):
        obj_causal.set_mu(mu)
        obj_causal.set_Sigma(Sigmas)

    def check_matrices(self, Sigmas):
        for qrt in ["ff", "hf", "fh", "hh"]:
            is_sym_vec=np.all([self.is_symmetric(Sigmas[wlt][qrt]) for wlt in range(0, len(Sigmas))])
            is_posdef_vec = np.all([self.is_pos_definite(Sigmas[wlt][qrt]) for wlt in range(0, len(Sigmas))])
            bool_val=np.all([is_sym_vec, is_posdef_vec])
            if(bool_val==False):
                return False
        return True
    def get_global_matrices(self, Sigmas):
        erg = {new_list: [] for new_list in range(len(Sigmas))}
        for wlt in erg:
            a, b, c, d=Sigmas[wlt]["hh"], Sigmas[wlt]["hf"], Sigmas[wlt]["fh"], Sigmas[wlt]["ff"]
            rb=np.concatenate((a, b), axis=1)
            qr = np.concatenate((c, d), axis=1)
            erg[wlt]=np.concatenate((rb, qr), axis=0)
        return erg
    def get_all_global_matrices(self, Sigmas):
        glob_matrices=self.get_global_matrices(Sigmas)
        is_sym_vec = np.all([self.is_symmetric(glob_matrices[wlt]) for wlt in range(0, len(glob_matrices))])
        is_posdef_vec = np.all([self.is_pos_definite(glob_matrices[wlt]) for wlt in range(0, len(glob_matrices))])
        bool_val = np.all([is_sym_vec, is_posdef_vec])
        None


A=np.array([[1, 8, 3, 7], [5, 2, 6, 8], [3, 4, 2, 8], [1, 7, 3, 9]])
cov=np.transpose(A)*A
pi={0: .8, 1: .2}
mu={0:
        {"h": np.array([1, 2]),
        "f": np.array([2, 3])},
    1:
        {"h": np.array([3, 4]),
        "f": np.array([2, 3])}}

Sigmas={0:
            {"hh": np.array([[1, 0.3], [0.3, 1]]),
            "hf": np.array([[1, 0.8], [0.8, 1]]),
            "fh": np.array([[1, 0.8], [0.8, 1]]),
            "ff": np.array([[1, 0.3], [0.3, 1]])},
        1:
           {"hh": np.array([[1, 0.6], [0.6, 1]]),
            "hf": np.array([[1, 0.4], [0.4, 1]]),
            "fh": np.array([[1, 0.4], [0.4, 1]]),
            "ff": np.array([[1, 0.6], [0.6, 1]])}}


obj_causal = causal_prob()

rt=obj_causal.is_pos_definite(cov)
obj_causal.check_matrices(Sigmas)
obj_causal.get_all_global_matrices(Sigmas)

new_mu=obj_causal.conditioned_mu(mu, Sigmas, np.array([8, 8]))
print(new_mu)
new_Sigma=obj_causal.conditioned_Sigma(Sigmas)
print(new_Sigma)
new_pi=obj_causal.conditioned_pi(mu, Sigmas, np.array([8, 8]), pi)
print(new_pi)

#obj_causal.set_mu_Sigmas(new_pi, new_mu, new_Sigma)
#obj_causal.visualize_multivariate_gaussian()
#obj_causal.plot_eigen_vectors_Sigma()
#obj_causal.show()