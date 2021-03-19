import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

"""
Function confindence_ellipse from https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
19.03.21
"""
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

df = pd.read_pickle("./data.pkl")
ref=df.iloc[0]


df_bik=df.loc[df["label"]=="Biker"]
df_ped=df.loc[df["label"]=="Pedestrian"]
df_ska=df.loc[df["label"]=="Skater"]

plt.scatter(df_bik["spx"], df_bik["spy"], label="initial positions (biker)")
plt.scatter(df_bik["epx"], df_bik["epy"], label="end positions (biker)")

plt.scatter(df_ped["spx"], df_ped["spy"], label="initial positions (pedestrian)")
plt.scatter(df_ped["epx"], df_ped["epy"], label="end positions (pedestrian)")

plt.scatter(df_ska["spx"], df_ska["spy"], label="initial positions (skater)")
plt.scatter(df_ska["epx"], df_ska["epy"], label="end positions (skater)")

plt.legend()
font = {'family': 'normal',
                'weight': 'bold',
                'size': 16}
plt.xlabel('x [-]', **font)
plt.ylabel('y [-]', **font)
plt.grid()
plt.show()

############################
### select single person ###
############################

fig, ax = plt.subplots(1, 1, figsize=(9, 9))
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
for i in np.unique(kmeans.labels_):
    rbt=(kmeans.labels_==i)
    sel_idx=[idx for idx, iqt in enumerate(rbt) if iqt]
    sel_X=X.loc[rbt]
    act_mean=sel_X.mean()
    rct=sel_X.cov()
    x=sel_X["epx"].to_numpy()
    y=sel_X["epy"].to_numpy()
    if(len(x)>4):
        confidence_ellipse(x, y, ax, edgecolor='red')

plt.legend()
font = {'family': 'normal',
                'weight': 'bold',
                'size': 16}
plt.xlabel('x [-]', **font)
plt.ylabel('y [-]', **font)
plt.grid()
plt.show()