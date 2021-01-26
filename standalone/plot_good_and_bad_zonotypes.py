from datareader.__init__ import *
import reachab
import time
params=get_params()

########################
### user preferences ###
########################
select_time=7740
t_min=select_time-params["window_size"] - 80
t_max=select_time+params["window_size"] + 20
init_time = time.time()

#############################
###     initial steps     ###
#############################
logging.basicConfig(level=logging.WARNING)
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
logging.warn("read dataset: Done after " + str(get_elapsed_time(init_time)))

######################################
### read the dataset by time-range ###
######################################
X=get_dataset_by_range(X_all, 't', t_min, t_max)
logging.warn("selection of the dataset: Done after " + str(get_elapsed_time(init_time)))

###########################################
###  get the values of a certain timestep     ###
###########################################
r, Y=get_dataset_by_column_value(X, "t", select_time)
logging.warn("select dataset for specific time: Done after " + str(get_elapsed_time(init_time)))

########################################
### get ids for a certain time step  ###
########################################
id_array=get_all_ids(Y)
logging.warn("get the IDs for certain timestep: Done after " + str(get_elapsed_time(init_time)))

#################################################################################################
### get initial state and future states for certain row. Perform Reachability Analysis for    ###
###                                      movement prediction                                  ###
#################################################################################################
sel_line=32
does_not_capture, U=check_if_line_is_captured_by_zonotypes(X_all, sel_line, params)
logging.warn("get initial state and future states for certain row. Perform Reachability Analysis for "
             "movement prediction  : Done after " + str(get_elapsed_time(init_time)))
##################################################
### get past measurements indices for each IDs ###
##################################################
past_dict={}
for act_id in id_array:
    past_dict[str(act_id)]=get_past_measurements_indices_for_id(X, act_id, select_time)
logging.warn("get past measurement indices for each IDs: Done after " + str(get_elapsed_time(init_time)))

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
logging.warn("get initial and past states for an ID and a specific timestamp: Done after " + str(get_elapsed_time(init_time)))


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
        params_reach= params['params_reach']
        zonoset = reachab.reach(Omega_0, U, params_reach)
reachab.show_all()
logging.warn("use of reachability analysis for movement prediction: Done after " + str(get_elapsed_time(init_time)))



