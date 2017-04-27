# file: scalar_+_vector.py
# Author: A. Laszlo Lazuer
# date: 04/27/17
# Python Version: 3.x
# Description: Following tutorial from Deep Learning nano Degree

##Task: Add a scalar to a vector using NumPy
values = [1,2,3,4,5]
original_values = values
values = np.array(values) + 5
print("Adding 5 to values:\nOriginal Vector: {0}\nResulting vector: {1}".format(original_values, values))
