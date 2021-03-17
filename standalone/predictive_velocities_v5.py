import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

class causal_prob(object):
    def __init__(self, **kwargs):
        None

    def is_pos_definite(self,x):
        eig_vals=np.linalg.eigvals(x)
        return np.all(eig_vals > 0)
    def get_small_pieces(self, idx):
        mu = self.mean[idx]
        mu_h = mu[0:2]
        mu_f = mu[2:]
        cov = self.cov[idx]
        cov_hh = cov[0:2, 0:2]
        cov_fh = cov[2:, 0:2]
        cov_hf = cov[0:2, 2:]
        cov_ff = cov[2:, 2:]
        return mu_h, mu_f, cov_hh, cov_fh, cov_hf, cov_ff
    def conditioned_mu(self, x_h):
        erg = {new_list: [] for new_list in range(0, len(self.mean))}
        for wlt in erg:
            mu_h, mu_f, cov_hh, cov_fh, cov_hf, cov_ff=self.get_small_pieces(wlt)
            dif_vec=x_h-mu_h
            added_mu_A=np.matmul(cov_fh,np.linalg.inv(cov_hh))
            added_mu=np.matmul(added_mu_A, dif_vec)
            erg[wlt]=mu_f+added_mu
        return erg

    def conditioned_Sigma(self):
        erg = {new_list: [] for new_list in range(0, len(self.mean))}
        for wlt in erg:
            mu_h, mu_f, cov_hh, cov_fh, cov_hf, cov_ff=self.get_small_pieces(wlt)
            a=np.matmul(cov_fh, np.linalg.inv(cov_hh))
            second_mat=np.matmul(a, cov_hf)
            erg[wlt]=cov_ff-second_mat
        return erg

    def predict(self, xh, vel):
        new_mu = obj_causal.conditioned_mu(xh)
        print(new_mu)
        new_sigma = obj_causal.conditioned_Sigma()
        print(new_sigma)
        new_pi = obj_causal.conditioned_pi(vel, xh)
        print(new_pi)
        return new_mu, new_sigma, new_pi
    def conditioned_pi(self, vel, x_h):
        all_val=[]
        for wlt in vel:
            mu_h, mu_f, cov_hh, cov_fh, cov_hf, cov_ff = self.get_small_pieces(wlt)
            all_val.append(vel[wlt]['pi']*multivariate_normal(mean=mu_h, cov=cov_hh).pdf(x_h))
        erg = {nl: all_val[nl]/np.sum(all_val) for nl in range(0, len(self.mean))}
        return erg

    def get_dataset(self, params, start_pos, vel):

        xs, ys = np.random.multivariate_normal(start_pos['mu'], start_pos['Sigma'], params['N']).T
        X = np.vstack((xs, ys))
        kmeans = KMeans(n_clusters=len(vel), random_state=0).fit(X.T)
        idx = kmeans.labels_
        # idx=np.random.choice(len(vel), params['N'], p=[vel[rqt]['pi'] for rqt in vel])
        xe, ye = np.zeros((len(xs),)), np.zeros((len(ys),))
        for ix, qrt in enumerate(idx):
            vx, vy = np.random.multivariate_normal(vel[qrt]['mu'], vel[qrt]['Sigma'])
            xe[ix], ye[ix] = xs[ix] + vx, ys[ix] + vy
        d = {'idx': idx, 'xs': xs, 'ys': ys, 'xe': xe, 'ye': ye}
        self.df = pd.DataFrame(data=d)
        self.sub_df = [self.df.loc[self.df['idx'] == qrt].drop('idx', axis=1) for qrt in range(0, len(vel))]
        self.mean = [np.array(self.sub_df[qrt].mean()) for qrt in range(0, len(self.sub_df))]
        self.cov = [np.array(self.sub_df[qrt].cov()) for qrt in range(0, len(self.sub_df))]
        self.pi = [vel[qrt]['pi'] for qrt in vel]

    def plot_the_scene(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        three_col = ["blue", "green", "red"]
        for count, act_idx in enumerate(np.unique(self.df["idx"])):
            new_dat = self.df.loc[self.df["idx"] == act_idx]
            ax.scatter(new_dat['xs'], new_dat['ys'], label='initial position ' + str(count), color=three_col[count],
                       alpha=.3)
        sel_col = ['blue', 'green', 'red', 'cyan', 'orange', 'yellow']
        [ax.scatter(qrt['xe'], qrt['ye'], label=str(idx), color=sel_col[idx]) for idx, qrt in enumerate(self.sub_df)]
        ax.legend()
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 16}
        ax.set_xlabel('x [m]', **font)
        ax.set_ylabel('y [m]', **font)
        plt.axis([-12, 12, -12, 12])
        plt.grid()
        plt.show()
    def do_xe_ye(self, do_prob):
        for idx, qrt in enumerate(do_prob):
            new_df=self.df.loc[self.df["idx"]==idx]
            new_xe, new_ye = np.random.multivariate_normal(do_prob[qrt]['mu'], do_prob[qrt]['Sigma'], np.size(new_df,0)).T
            self.df.loc[self.df["idx"]==idx, "xe"] = new_xe
            self.df.loc[self.df["idx"]==idx, "ye"] = new_ye
        #self.df["idx"] = 0
        self.sub_df = [self.df.loc[self.df['idx'] == qrt].drop('idx', axis=1) for qrt in range(0, len(vel))]
        self.mean = [np.array(self.sub_df[qrt].mean()) for qrt in range(0, len(self.sub_df))]
        self.cov = [np.array(self.sub_df[qrt].cov()) for qrt in range(0, len(self.sub_df))]
        self.pi = [vel[qrt]['pi'] for qrt in vel]

    def mahalabonis_dist(self, x, mu, Sigma):
        return -0.5*np.transpose(x-mu)*np.linalg.inv(Sigma)*(x-mu)
    def multivariate_gaussian_distribution(self, x, mu, Sigma):
        factor_A=1/np.sqrt((2*np.pi)**2*np.linalg.det(Sigma))
        factor_B=np.exp(self.mahalabonis_dist(x, mu, Sigma))
        erg=factor_A*factor_B
        return erg[0]

    def contour_plot(self, xh, new_mu, new_pi):
        NGRID = 40
        X = np.linspace(-11, 11, NGRID)
        Y = np.linspace(-11, 11, NGRID)
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure()
        ax = fig.add_subplot()
        Z = np.zeros((np.size(X, 0), np.size(X, 1)))
        for idx_A in range(0, np.size(X, 0)):
            for idx_B in range(0, np.size(X, 1)):
                x = np.array([X[idx_A, idx_B], Y[idx_A, idx_B]])
                for qrt in range(0, len(self.mean)):
                    mu_h, mu_f, cov_hh, cov_fh, cov_hf, cov_ff = self.get_small_pieces(qrt)
                    new_val = self.pi[qrt] * multivariate_normal(mean=mu_f, cov=cov_ff).pdf(x)
                    Z[idx_A, idx_B] += new_val
        # self.ax.plot_surface(self.X, self.Y, Z,  cmap='viridis',
        #               linewidth=0, antialiased=False, alpha=.3)
        plt.arrow(xh[0], xh[1], new_mu[0][0] - xh[0], new_mu[0][1] - xh[1],
                  fc="blue", ec="black", alpha=.5, width=new_pi[0] + .5,
                  head_width=1.4, head_length=.63)
        plt.arrow(xh[0], xh[1], new_mu[1][0] - xh[0], new_mu[1][1] - xh[1],
                  fc="green", ec="black", alpha=.5, width=new_pi[1] + .5,
                  head_width=1.4, head_length=.63)
        plt.arrow(xh[0], xh[1], new_mu[2][0] - xh[0], new_mu[2][1] - xh[1],
                  fc="red", ec="black", alpha=.5, width=new_pi[2] + .5,
                  head_width=1.4, head_length=.63)
        ax.contour(X, Y, Z, 10, lw=3, cmap="autumn_r", linestyles="solid", offset=0)
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 16}
        ax.set_xlabel('x [m]', **font)
        ax.set_ylabel('y [m]', **font)
        plt.axis([-12, 12, -12, 12])
        plt.grid()
        plt.show()

    def resulting_contour_plot(self, new_mu, new_sigma, new_pi):
        NGRID = 40
        X = np.linspace(-11, 11, NGRID)
        Y = np.linspace(-11, 11, NGRID)
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure()
        ax = fig.add_subplot()
        Z = np.zeros((np.size(X, 0), np.size(X, 1)))
        for idx_A in range(0, np.size(X, 0)):
            for idx_B in range(0, np.size(X, 1)):
                x = np.array([X[idx_A, idx_B], Y[idx_A, idx_B]])
                for qrt in range(0, len(self.mean)):
                    new_val=new_pi[qrt]*multivariate_normal(mean=new_mu[qrt], cov=new_sigma[qrt]).pdf(x)
                    Z[idx_A, idx_B] += new_val
        # self.ax.plot_surface(self.X, self.Y, Z,  cmap='viridis',
        #               linewidth=0, antialiased=False, alpha=.3)
        plt.arrow(xh[0], xh[1], new_mu[0][0]-xh[0], new_mu[0][1]-xh[1],
                  fc="blue", ec="black", alpha=.5, width=new_pi[0]+.5,
                  head_width=1.4, head_length=.63)
        plt.arrow(xh[0], xh[1], new_mu[1][0] - xh[0], new_mu[1][1] - xh[1],
                  fc="green", ec="black", alpha=.5, width=new_pi[1]+.5,
                  head_width=1.4, head_length=.63)
        plt.arrow(xh[0], xh[1], new_mu[2][0] - xh[0], new_mu[2][1] - xh[1],
                  fc="red", ec="black", alpha=.5, width=new_pi[2]+.5,
                  head_width=1.4, head_length=.63)
        ax.contour(X, Y, Z, 10, lw=3, cmap="autumn_r", linestyles="solid", offset=0)
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 16}
        ax.set_xlabel('x [m]', **font)
        ax.set_ylabel('y [m]', **font)
        plt.axis([-12, 12, -12, 12])
        plt.grid()
        plt.show()



start_pos={'mu': np.array([0, 0]), 'Sigma': np.array([[1, 0], [0, 1]])}
vel={0: {'pi':.6, 'mu': np.array([8, 8]), 'Sigma': np.array([[1, 0.5], [0.5, 1]])},
     1:{'pi':.3, 'mu': np.array([-8, 2]), 'Sigma': np.array([[.4, -0.6], [-0.6, .4]])},
     2: {'pi': .1, 'mu': np.array([1, -7]), 'Sigma': np.array([[.4, -0.6], [-0.6, .4]])}
     }

params={'N': 600}
xh=np.array([0, 2])
obj_causal = causal_prob()
obj_causal.get_dataset(params, start_pos,vel)
obj_causal.do_xe_ye({0: {'mu': np.array([8, 2]), 'Sigma': np.array([[1.4, 0], [0, 1.4]])},
                         1: {'mu': np.array([8, 0]), 'Sigma': np.array([[1.4, 0], [0, 1.4]])},
                         2: {'mu': np.array([8, -2]), 'Sigma': np.array([[1.4, 0], [0, 1.4]])}})
new_mu, new_sigma, new_pi=obj_causal.predict(xh, vel)
obj_causal.plot_the_scene()
obj_causal.contour_plot(xh, new_mu, new_pi)
obj_causal.resulting_contour_plot(new_mu, new_sigma, new_pi)