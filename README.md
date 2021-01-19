# datareader

```python
params = datareader.get_params()
path = datareader.__path__[0]
newpath = os.path.join(path, "data/input/stanford/bookstore/video0/annotations.txt")
X = datareader.read_dataset(params, newpath)
print(X)