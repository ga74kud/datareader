from datareader.__init__ import *
import pandas as pd
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
font = {'family': 'normal',
                'weight': 'bold',
                'size': 16}
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
#############################################
### find similiar initial starting points ###
#############################################
columns=["idx", "spx", "spy", "epx", "epy", "st", "et", "label"] #idx, initial_start_x, initial_start_y, end_x, end_y, start_time, end_time
df = pd.DataFrame([], columns=columns)
all_ids=np.unique(X["id"])
for idx, wlt in enumerate(all_ids):
    r, Y = get_dataset_by_column_value(X, "id", wlt)
    lab=Y["label"].iloc[0]
    rt, rx, ry = get_center_value(Y)
    df.loc[idx]=[np.int(wlt), np.int(rx[0]), np.int(ry[0]), np.int(rx[-1]), np.int(ry[-1]), np.int(rt[0]), np.int(rt[-1]), lab]
df.to_pickle("./data.pkl")