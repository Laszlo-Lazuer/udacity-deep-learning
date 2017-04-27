# file: scalar_+_vector.py
# Author: A. Laszlo Lazuer
# date: 04/26/17
# Python Version: 3.x
# Description: Following tutorial from Deep Learning nano Degree
# Github for example: https://github.com/llSourcell/linear_regression_liveimport numpy as np

##Add a scalar to a vector using numPY

values = [1,2,3,4,5]
original_values = values
values = np.array(values) + 5
print("Adding 5 to values:\nOriginal Vector: {0}\nResulting vector: {1}".format(original_values, values))
