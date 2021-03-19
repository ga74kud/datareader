import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import reachab as rb
from datareader.__init__ import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def read_whole_dataset():
    params=get_params()
    logging.basicConfig(level=logging.INFO)
    path = datareader.__path__[0]
    location="bookstore"
    video="video0"
    annotation_path = os.path.join(path, "data/input/stanford/", location, video, "annotations.txt")
    image_path = os.path.join(path, "data/input/stanford/", location, video, "reference.jpg")
    X = read_dataset(params, annotation_path)
    extrema_X=get_extrema(X)
    return X, extrema_X

def plot_id_movement(id, X, col="blue"):
    r, Y=get_dataset_by_column_value(X, "id", id)
    rt,rx,ry=get_center_value(Y)
    plt.plot(rx, ry, color=col, alpha=.6)

def show_plot(extrema_X):
    plt.axis(extrema_X)
    plt.legend()
    plt.show()

fig, ax = plt.subplots(1, 1, figsize=(9, 9))

X_All, extrema_X_All=read_whole_dataset()
df = pd.read_pickle("./data.pkl")
ref=df.iloc[0]


############################
### select single person ###
############################


df=df.loc[df["et"]<ref["st"]]
bool_vec=((df["spx"]-ref["spx"])**2+(df["spy"]-ref["spy"])**2)**.5<100
df=df.loc[bool_vec]
df=df.loc[df["label"]==ref["label"]]

plt.scatter(df["spx"], df["spy"], label="initial positions ("+ref["label"]+")")
plt.scatter(df["epx"], df["epy"], label="end positions ("+ref["label"]+")")

X=df[["epx", "epy"]].astype(float)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
            color="k", zorder=10, label="kmeans-center")
col=["red", "blue", "green", "cyan"]
for idx,wlt in enumerate(kmeans.labels_):
    plot_id_movement(df["idx"].iloc[idx], X_All, col[wlt])

plt.legend()
font = {'family': 'normal',
                'weight': 'bold',
                'size': 16}
plt.xlabel('x [-]', **font)
plt.ylabel('y [-]', **font)
plt.grid()
#plt.show()
show_plot(extrema_X_All)
