import warnings
import pandas as pd
import numpy as np

class causal_prob(object):
    def __init__(self, **kwargs):
        None

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
    def get_dataset(self, start_pos,vel):

        xs, ys = np.random.multivariate_normal(start_pos['mu'], start_pos['Sigma'], 5000).T
        vx, vy = np.random.multivariate_normal(vel['mu'], vel['Sigma'], 5000).T
        d = {'xs': xs, 'ys': ys, 'xe': xs+vx, 'ye': ys+vy}
        self.df = pd.DataFrame(data=d)
        self.mean=np.array(self.df.mean())
        self.cov=np.array(self.df.cov())
        rt=self.is_pos_definite(self.cov)
        if(rt==False):
            warnings.warn("not positive definite")


start_pos={'mu': np.array([1, 2]), 'Sigma': np.array([[1, 0.3], [0.3, 1]])}
vel={'mu': np.array([4, 4]), 'Sigma': np.array([[1, 0.5], [0.5, 1]])}

obj_causal = causal_prob()
obj_causal.get_dataset(start_pos,vel)
new_mu=obj_causal.conditioned_mu(np.array([5, 8]))
print(new_mu)
new_Sigma=obj_causal.conditioned_Sigma()
print(new_Sigma)