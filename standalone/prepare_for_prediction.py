import os
import datareader
from datareader.__init__ import *
import matplotlib.pyplot as plt

########################
### user preferences ###
########################
select_time=8000
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
###  get the values of a certain timestep     ###
###########################################
r, Y=get_dataset_by_column_value(X, "t", select_time)
logging.info(Y)
########################################
### get ids for a certain time step  ###
########################################
id_array=get_all_ids(Y)
#########################################
### get past measurements for each id ###
#########################################
past_dict={}
for act_id in id_array:
    past_dict[str(act_id)]=get_past_measurements_indices_for_id(X, act_id, select_time)
###########################################
### get future measurements for each id ###
###########################################
future_dict={}
for act_id in id_array:
    future_dict[str(act_id)]=get_future_measurements_for_id(X, act_id, select_time)
######################################################################
### get initial and past states for an id and a specific timestamp ###
######################################################################
initial_state_dict={}
past_states_dict={}
for act_id in id_array:
    if(len(past_dict[str(act_id)])>params["window_size"]+2):
        initial_state, past_states=get_past_states(X, past_dict[str(act_id)], params)
        initial_state_dict[str(act_id)] = initial_state
        past_states_dict[str(act_id)] = past_states
########################################################################
### get initial and future states for an id and a specific timestamp ###
########################################################################
initial_state_dict={}
future_states_dict={}
for act_id in id_array:
    if (len(future_dict[str(act_id)]) > params["window_size"] + 2):
        future_states=get_future_states(X, future_dict[str(act_id)], params)
        future_states_dict[str(act_id)] = future_states