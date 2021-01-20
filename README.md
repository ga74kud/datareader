# datareader

```
params = datareader.get_params()
path = datareader.__path__[0]
newpath = os.path.join(path, "data/input/stanford/bookstore/video0/annotations.txt")
X = datareader.read_dataset(params, newpath)
print(X)
```


# Third party

Stanford Drone Dataset annotations from https://cvgl.stanford.edu/projects/uav_data/ are used: 

Please cite also the Stanford Drone Dataset:
A. Robicquet, A. Sadeghian, A. Alahi, S. Savarese, Learning Social Etiquette: Human Trajectory Prediction In Crowded Scenes in European Conference on Computer Vision (ECCV), 2016. 