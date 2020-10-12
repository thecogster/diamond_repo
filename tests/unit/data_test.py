# test integrity of the input data
import os
import numpy as np
import pandas as pd

# get absolute path of csv files from data folder
def get_absPath(filename):
    """Returns the path of the notebooks folder"""
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", filename
        )
    )
    return path

expected_columns = 10


# distribution of features in the training set
historical_mean = np.array(
    [
0.797940,
3.904097,
4.405803,
6.051020,
61.749405,
57.457184,
3932.799722,
5.731157,
5.734526,
3.538734,
]    
)

historical_std = np.array(
[
0.474011,
1.116600,
1.701105,
1.647136,
1.432621,
2.234491,
3989.439738,
1.121761,
1.142135,
0.705699,
]
)

# maximal relative change in feature mean or standrd deviation
# that we can tolerate
shift_tolerance = 3

def test_check_schema():
    datafile = get_absPath("diamonds.csv")
    # check that file exists
    assert os.path.exists(datafile)
    dataset = pd.read_csv(datafile)
    header = dataset[dataset.columns[:-1]]
    actual_columns = header.shape[1]
    # check header has expected number of columns
    assert actual_columns == expected_columns


def test_check_missing_values():
    datafile = get_absPath("diamonds_missing_values.csv")
    # check that file exists
    assert os.path.exists(datafile)
    dataset = pd.read_csv(datafile)
    n_nan = np.sum(pd.isnull(dataset.values))
    assert n_nan > 0