import os
import datareader
from datareader.__init__ import *
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
########################
### user preferences ###
########################
id=0
selection_A="opt_1"

options={
        'opt_1':
            {"window": 71, "poly_order": 2},
        }

val_A_window=options[selection_A]["window"]
val_A_polyorder=options[selection_A]["poly_order"]

#############################
###     initial steps     ###
#############################
params=get_params()
logging.basicConfig(level=logging.INFO)
path = datareader.__path__[0]
location="bookstore"
video="video0"
annotation_path = os.path.join(path, "data/input/stanford/", location, video, "annotations.txt")
image_path = os.path.join(path, "data/input/stanford/", location, video, "reference.jpg")

################################
###     read the dataset     ###
################################
X = read_dataset(params, annotation_path)
extrema_X=get_extrema(X)

###########################################
###  get the values of a certain id     ###
###########################################
r, Y=get_dataset_by_column_value(X, "id", id)
###########################################
### get arrays for velocity computation ###
###########################################
rt,rx,ry=get_center_value(Y)
#################################
### computation of velocities ###
#################################
ax=(rx[:-1]-rx[1:])
ay=(ry[:-1]-ry[1:])
b=(rt[:-1]-rt[1:])
cx=ax/b
cy=ay/b
f, (ax1) = plt.subplots(1, 1, sharey=True)
vx_a = savgol_filter(rx, val_A_window, val_A_polyorder, 1)
vy_a = savgol_filter(ry, val_A_window, val_A_polyorder, 1)
naive_vel=np.sqrt(cx**2+cy**2)
center_vel=np.sqrt(vx_a**2+vy_a**2)
ax1.plot(rt[:-1], naive_vel, label='naive_v')
ax1.plot(rt, center_vel, label='v_'+str(val_A_window)+'_'+str(val_A_polyorder))
ax1.legend()
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
X=np.transpose(np.vstack((rx, ry)))
y=center_vel


plt.show()


################################
### compare position methods ###
################################
plt.plot(rx, ry, label='ground')


# ##############
# px = savgol_filter(rx, 71, 2, 0)
# py = savgol_filter(ry, 71, 2, 0)
# plt.plot(px, py, label='0-derivative')
##############
abx=np.cumsum(vx_a)
aby=np.cumsum(vy_a)
px_a=rx[0]+abx
py_a=ry[0]+aby
plt.plot(px_a, py_a, label='filter_'+str(val_A_window)+'_'+str(val_A_polyorder))
##############
px_b=rx[0]+abx
py_b=ry[0]+aby
##############
plt.axis(extrema_X)
plt.legend()
##############
# Arrows
for wlt in range(0, len(rx)):
    plt.arrow(rx[wlt],ry[wlt],vx_a[wlt],vy_a[wlt],
                    fc="black", ec='black', alpha=.7, width=.1,
                    head_width=1.0, head_length=1)
plt.show()
