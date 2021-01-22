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
rt,rxmin, rxmax,rymin, rymax=get_values_rectangle_for_time_range(Y)
#################################
### computation of velocities ###
#################################
plt.plot(rxmin, rymin)
plt.plot(rxmax, rymax)
[plt.plot([rxmin[i], rxmax[i]], [rymin[i], rymax[i]], 'green') for i in range(0, len(rxmin), 15)]
plt.axis(extrema_X)
plt.legend()
plt.show()
