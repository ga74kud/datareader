import warnings
import pandas as pd
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
    def get_small_pieces(self):
        mu = self.mean
        mu_h = mu[0:2]
        mu_f = mu[2:]
        cov = self.cov
        cov_hh = cov[0:2, 0:2]
        cov_fh = cov[2:, 0:2]
        cov_hf = cov[0:2, 2:]
        cov_ff = cov[2:, 2:]
        return mu_h, mu_f, cov_hh, cov_fh, cov_hf, cov_ff
    def conditioned_mu(self, x_h):
        mu_h, mu_f, cov_hh, cov_fh, cov_hf, cov_ff=self.get_small_pieces()
        dif_vec=x_h-mu_h
        added_mu_A=np.matmul(cov_fh,np.linalg.inv(cov_hh))
        added_mu=np.matmul(added_mu_A, dif_vec)
        erg=mu_f+added_mu
        return erg

    def conditioned_Sigma(self):
        mu_h, mu_f, cov_hh, cov_fh, cov_hf, cov_ff=self.get_small_pieces()
        a=np.matmul(cov_fh, np.linalg.inv(cov_hh))
        second_mat=np.matmul(a, cov_hf)
        erg=cov_ff-second_mat
        return erg

    def set_mu_Sigmas(self, pi, mu, Sigmas):
        self.set_mu(mu)
        self.set_Sigma(Sigmas)

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
    def get_dataset(self, start_pos,end_pos):

        xs, ys = np.random.multivariate_normal(start_pos['mu'], start_pos['Sigma'], 5000).T
        xe, ye = np.random.multivariate_normal(end_pos['mu'], end_pos['Sigma'], 5000).T
        d = {'xs': xs, 'ys': ys, 'xe': xe, 'ye': ye}
        self.df = pd.DataFrame(data=d)
        self.mean=np.array(self.df.mean())
        self.cov=np.array(self.df.cov())
        rt=self.is_pos_definite(self.cov)
        if(rt==False):
            warnings.warn("not positive definite")


start_pos={'mu': np.array([1, 2]), 'Sigma': np.array([[1, 0.3], [0.3, 1]])}
end_pos={'mu': np.array([4, 4]), 'Sigma': np.array([[1, 0.5], [0.5, 1]])}

obj_causal = causal_prob()
obj_causal.get_dataset(start_pos,end_pos)
new_mu=obj_causal.conditioned_mu(np.array([5, 8]))
print(new_mu)
new_Sigma=obj_causal.conditioned_Sigma()
print(new_Sigma)