import glob
my_data_files = []
directories = glob.glob('./datareader/data')
for directory in directories:
    files = glob.glob(directory+'*')
    my_data_files.append((directory, files))

None