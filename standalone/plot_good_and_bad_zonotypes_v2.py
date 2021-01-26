from datareader.__init__ import *
import time

########################
### start ###
########################
init_time = time.time()
logging.warn("start: Done after " + str(get_elapsed_time(init_time)))

########################
### user preferences ###
########################
select_time=7740
params=get_params()
t_min=select_time-params["window_size"] - 80
t_max=select_time+params["window_size"] + 20

#############################
###     initial steps     ###
#############################
logging.basicConfig(level=logging.WARNING)
path = datareader.__path__[0]
location="bookstore"
video="video0"
annotation_path = os.path.join(path, "data/input/stanford/", location, video, "annotations.txt")
image_path = os.path.join(path, "data/input/stanford/", location, video, "reference.jpg")
logging.warn("initial steps: Done after " + str(get_elapsed_time(init_time)))
################################
###     read the dataset     ###
################################
X_all = read_dataset(params, annotation_path)
extrema_X=get_extrema(X_all)
logging.warn("read dataset: Done after " + str(get_elapsed_time(init_time)))

###########################################
###  get the values of a certain timestep     ###
###########################################
r, Y=get_dataset_by_column_value(X_all, "t", select_time)
logging.warn("select dataset for specific time: Done after " + str(get_elapsed_time(init_time)))

#################################################################################################
### get initial state and future states for certain row. Perform Reachability Analysis for    ###
###                                      movement prediction                                  ###
#################################################################################################

for sel_line in Y.index:
    does_not_capture, Omega_0, U=check_if_line_is_captured_by_zonotypes(X_all, sel_line, params)
    params_reach = params['params_reach']
    if(does_not_capture==True):
        params_reach['face_color'] = "red"
    else:
        params_reach['face_color'] = "green"
    zonoset = reachab.reach(Omega_0, U, params_reach)
reachab.show_all()
logging.warn("get initial state and future states for certain row. Perform Reachability Analysis for "
             "movement prediction  : Done after " + str(get_elapsed_time(init_time)))


