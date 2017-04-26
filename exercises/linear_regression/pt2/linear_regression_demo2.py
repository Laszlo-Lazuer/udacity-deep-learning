# file: linear_regression_demo2.py
# Author: A. Laszlo Lazuer
# date: 04/25/17
# Python Version: 3.x
# Description: Following tutorial from Deep Learning nano Degree
# Github for example: https://github.com/llSourcell/linear_regression_live

from numpy import *

def compute_error_for_line_given_points(b, m, points):
    #initialize at 0
    totalError = 0
    #for every point
    #Eq for sum of squred distances
    for i in range(0, len(points)):
        #get the x value
        x = points[i,0]
        #get the y value
        y = points[i, 1]
        #get the difference, square it, add it to the total
        totalError += (y - (m * x +b)) **2

    #get the average
    return totalError / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    #graident descent inproves the line

    #starting b and m
    b = starting_b
    m = starting_m

    #gradient descent
    for i in range(num_iterations):
        #update b and m with the new more accurate b and m by performing
        #this gradient step
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

#magic, the greatest the greatest

def step_gradient(b_current, m_current, points, learningRate):
    #Colorful parabolic graph
    #Local minima point at the very bottom: where we are trying to get to.
    ##Point where the error is the smalles, found with gradient descent.
    ##Gradient = slope, used everywhere in ML.

    #starting points for our gradients
    b_gradient = 0
    m_gradient = 0

    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i, 1]
        #direction with respect to b and m
        #computing partial derivatives of our error function.
        ##Calculate the partial derivative with respect to b & m
        ##Gradient descent is a search policy.
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x *  (y - ((m_current * x) + b_current))

    #update our b an dm values using our partial derivatives
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]



def run():
    #Step 1 - collect our data
    points = genfromtxt('data.csv', delimiter=',')

    #Step 2 - define our hyperparameters
    #how fast should our model converge? *When you get the optimal model
    learning_rate = 0.0001
    #y = mx + b (slope formula)
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    #Step 3 - train our model
    print 'Starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print 'Ending point at b = {1}, m = {2}, error = {3}'.format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))

if __name__ == '__main__':
    run()
