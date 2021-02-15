import os
import datareader
from datareader.__init__ import *
import matplotlib.pyplot as plt

########################
### user preferences ###
########################
id=0
selection_A="opt_1"
selection_B="opt_2"

options={
        'opt_1':
            {"window": 71, "poly_order": 2},
        'opt_2':
            {"window": 21, "poly_order": 2},
        'opt_3':
            {"window": 71, "poly_order": 5},
        'opt_4':
            {"window": 21, "poly_order": 5},
        }

val_A_window=options[selection_A]["window"]
val_A_polyorder=options[selection_A]["poly_order"]
val_B_window=options[selection_B]["window"]
val_B_polyorder=options[selection_B]["poly_order"]
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
logging.info(X)

###########################################
###  get the values of a certain id     ###
###########################################
r, Y=get_dataset_by_column_value(X, "id", id)
logging.info(Y)
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
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
vx_a = savgol_filter(rx, val_A_window, val_A_polyorder, 1)
vy_a = savgol_filter(ry, val_A_window, val_A_polyorder, 1)
vx_b = savgol_filter(rx, val_B_window, val_B_polyorder, 1)
vy_b = savgol_filter(ry, val_B_window, val_B_polyorder, 1)
ax1.plot(rt[:-1], cx, label='naive_vx')
ax1.plot(rt, vx_a, label='vx_'+str(val_A_window)+'_'+str(val_A_polyorder))
ax1.plot(rt, vx_b, label='vx_'+str(val_B_window)+'_'+str(val_B_polyorder))
ax1.legend()
ax2.plot(rt[:-1], cy, label='naive_vy')
ax2.plot(rt, vy_a, label='vy_'+str(val_A_window)+'_'+str(val_A_polyorder))
ax2.plot(rt, vy_b, label='vy_'+str(val_B_window)+'_'+str(val_B_polyorder))
ax2.legend()
plt.show()


################################
### compare position methods ###
################################
logging.info(rt)
logging.info(rx)
logging.info(ry)
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
abx=np.cumsum(vx_b)
aby=np.cumsum(vy_b)
px_b=rx[0]+abx
py_b=ry[0]+aby
plt.plot(px_b, py_b, label='filter_'+str(val_B_window)+'_'+str(val_B_polyorder))
##############
plt.axis(extrema_X)
plt.legend()
plt.show()
