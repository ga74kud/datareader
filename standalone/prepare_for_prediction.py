import os
import datareader
from datareader.__init__ import *
import matplotlib.pyplot as plt
import reachab
import time
params=get_params()

########################
### user preferences ###
########################
Ts=0.02
select_time=7740
t_min=select_time-params["window_size"] - 80
t_max=select_time+params["window_size"] + 20
init_time = time.time()

#############################
###     initial steps     ###
#############################
logging.basicConfig(level=logging.INFO)
path = datareader.__path__[0]
location="bookstore"
video="video0"
annotation_path = os.path.join(path, "data/input/stanford/", location, video, "annotations.txt")
image_path = os.path.join(path, "data/input/stanford/", location, video, "reference.jpg")

################################
###     read the dataset     ###
################################
X_all = read_dataset(params, annotation_path)
extrema_X=get_extrema(X_all)
logging.info("read dataset: Done after " + str(get_elapsed_time(init_time)))

######################################
### read the dataset by time-range ###
######################################
X=get_dataset_by_range(X_all, 't', t_min, t_max)
logging.info("selection of the dataset: Done after " + str(get_elapsed_time(init_time)))

###########################################
###  get the values of a certain timestep     ###
###########################################
r, Y=get_dataset_by_column_value(X, "t", select_time)
logging.info("select dataset for specific time: Done after " + str(get_elapsed_time(init_time)))

########################################
### get ids for a certain time step  ###
########################################
id_array=get_all_ids(Y)
logging.info("get the IDs for certain timestep: Done after " + str(get_elapsed_time(init_time)))

##########################################
### get initial state for certain row  ###
##########################################
sel_line=32
initial_state, initial_state_set, past_states=get_initial_state_for_line(X_all, sel_line, params)
future_states=get_future_states_for_line(X_all, sel_line, params)
Omega_0, U=get_past_states_and_predict_initial_zonotypes(initial_state, initial_state_set, past_states)
evaluate_prediction_with_zonotypes(Omega_0, U, sel_line, future_states, params)
##################################################
### get past measurements indices for each IDs ###
##################################################
past_dict={}
for act_id in id_array:
    past_dict[str(act_id)]=get_past_measurements_indices_for_id(X, act_id, select_time)
logging.info("get past measurement indices for each IDs: Done after " + str(get_elapsed_time(init_time)))

####################################################
### get future measurements indices for each IDs ###
####################################################
future_dict={}
for act_id in id_array:
    future_dict[str(act_id)]=get_future_measurements_for_id(X, act_id, select_time)
logging.info("get future measurement indices for each IDs: Done after " + str(get_elapsed_time(init_time)))

######################################################################
### get initial and past states for an ID and a specific timestamp ###
######################################################################
initial_state_dict={}
past_states_dict={}
initial_state_set_dict={}
for act_id in id_array:
    if(len(past_dict[str(act_id)])>params["window_size"]):
        initial_state, past_states, initial_state_set=get_past_states(X, past_dict[str(act_id)], params)
        initial_state_dict[str(act_id)] = initial_state
        initial_state_set_dict[str(act_id)] = initial_state_set
        past_states_dict[str(act_id)] = past_states
logging.info("get initial and past states for an ID and a specific timestamp: Done after " + str(get_elapsed_time(init_time)))

############################################################
### get future states for an ID and a specific timestamp ###
############################################################
future_states_dict={}
for act_id in id_array:
    future_states=get_future_states(X, future_dict[str(act_id)], params)
    future_states_dict[str(act_id)] = future_states
logging.info("get future states for an ID and a specific timestamp: Done after " + str(get_elapsed_time(init_time)))

############################################################
### use of reachability analysis for movement prediction ###
############################################################
for act_id in id_array:
    act_val=str(act_id)
    if(act_val in initial_state_dict):
        x_initial=initial_state_dict[str(act_id)]
        x_initial_set=initial_state_set_dict[str(act_id)]
        Omega_0 = {'c': x_initial, 'g': x_initial_set}
        U = {'c': np.matrix([[0],
                             [0],
                             [0],
                             [0],
                             ]),
             'g': np.matrix([[10, 0],
                             [0, 10],
                             [0, 0],
                             [0, 0]
                             ])
             }
        params_reach= {}
        params_reach['box_function']='with_box'
        params_reach['steps']=2
        params_reach['time_horizon'] = 14.1
        params_reach['visualization']='y'
        zonoset = reachab.reach(Omega_0, U, params_reach)
reachab.show_all()
logging.info("use of reachability analysis for movement prediction: Done after " + str(get_elapsed_time(init_time)))



################################################################
### get the dataset of a certain id and perform a prediction ###
################################################################
id=2
r, Z=get_dataset_by_column_value(X, "id", id)
b_dataset=Z.loc[Z['t']>select_time]
total_time=(np.max(b_dataset['t'])-np.min(b_dataset['t']))*Ts
x_initial=initial_state_dict[str(id)]
x_initial_set=initial_state_set_dict[str(act_id)]
Omega_0 = {'c': x_initial, 'g': x_initial_set}
U = {'c': np.matrix([[0], [0], [0], [0], ]),
     'g': np.matrix([[10, 0], [0, 10], [0, 0], [0, 0] ]) }
params_reach['box_function']='with_box'
params_reach['steps']=10
params_reach['time_horizon'] = total_time
params_reach['visualization']='y'
zonoset = reachab.reach(Omega_0, U, params_reach)
evaluate_zonoset_selected_dataset(b_dataset, zonoset, extrema_X)
q=1