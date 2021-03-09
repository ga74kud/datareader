from math import atan2, degrees
from datareader.__init__ import *
import matplotlib.pyplot as plt

########################
### user preferences ###
########################
big_N=100
Ts=0.02
id=0
moving_N=100
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
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
vx_a = savgol_filter(rx, val_A_window, val_A_polyorder, 1)
vy_a = savgol_filter(ry, val_A_window, val_A_polyorder, 1)
naive_vel=np.sqrt(cx**2+cy**2)
center_vel=np.sqrt(vx_a**2+vy_a**2)
diff_vel=(naive_vel-center_vel[1:])**2
pos_err, neg_err=0*np.abs(diff_vel), 0*np.abs(diff_vel)
for wlt in range(0, len(diff_vel)):
    if(wlt>moving_N):
        sel_window=diff_vel[wlt-moving_N:wlt]
        diff_vel[wlt]=np.mean(sel_window)
ax1 = plt.subplot(211)
ax1.plot(rt[:-1], naive_vel, label='naive_v')
ax1.plot(rt, center_vel, label='v_'+str(val_A_window)+'_'+str(val_A_polyorder))
ax1.fill_between(rt[1:], center_vel[1:] - diff_vel, center_vel[1:] + diff_vel, alpha=0.2, label="confidence")
ax1.legend()
ax1.set_ylabel('Velocity')
ax1.set_xlabel('Time')
ax1.grid(True)
ax1.set_ylim([0,10])
#ax2=ax1.twinx()
ax2 = plt.subplot(212)
curvature=[round(degrees(atan2(vy_a[i], vx_a[i]))) for i in range(0, len(vx_a))]
ax2.plot(rt, curvature, label="Curvature")
ax2.set_ylim([-180,180])
ax2.grid(True)
ax2.set_ylabel('Angle in Degress Celsius (°C)')
ax2.set_xlabel('Time')
ax2.legend()
plt.show()

