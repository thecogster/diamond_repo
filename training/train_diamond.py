import pickle
import os
import numpy as np
import pandas as pd
import json
import subprocess
import sklearn

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import svm, preprocessing
from joblib import dump
from typing import Tuple, List

import matplotlib.pyplot as plt
import seaborn as sns

from azureml.core import Workspace, Datastore, Dataset
from azureml.core.run import Run
from azureml.core.model import Model

run = Run.get_context()
exp = run.experiment
ws = run.experiment.workspace

print("Loading training data...")
datastore = ws.get_default_datastore()
datastore_paths = [(datastore, 'diamonds/diamonds.csv')]
traindata = Dataset.Tabular.from_delimited_files(path=datastore_paths)
diamonds = traindata.to_pandas_dataframe()
print("Columns:", diamonds.columns) 
print("Diamonds data set dimensions : {}".format(diamonds.shape))


newdf = sklearn.utils.shuffle(diamonds)
X= newdf.drop('price', axis =1).values
y = newdf['price'].values
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.004)
clf = svm.SVR(kernel='linear')
clf.fit(X_train,y_train)
score = print(clf.score(X_test, y_test))
run.log("Score", score)


# Save model as part of the run history
print("Exporting the model as pickle file...")
outputs_folder = './model'
os.makedirs(outputs_folder, exist_ok=True)

model_filename = "sklearn_diamonds_model.pkl"
model_path = os.path.join(outputs_folder, model_filename)
dump(reg, model_path)

# upload the model file explicitly into artifacts
print("Uploading the model into run artifacts...")
run.upload_file(name="./outputs/models/" + model_filename, path_or_stream=model_path)
print("Uploaded the model {} to experiment {}".format(model_filename, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)
print("Following files are uploaded ")
print(run.get_file_names())

run.complete()
