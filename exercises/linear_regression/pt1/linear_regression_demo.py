## Github for example: https://github.com/llSourcell/linear_regression_live
##

from numpy import *

def comput_error_for_given_points(b, m, points):
    totalError = 0
    for i in range(0,len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m*x+b)) ** 2 #Sum of squared errors
    return totalError / float(len(points))



def step_gradient(b_current, m_current, points, learningRate):
    #gradient descent
    ##gradient => Direction
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i  in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
        return [b,m]

def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001 #Hyper parameter tuning knobs for the model, guess & check

#y = mx + b (slope formula)
    initial_b = 0 #y intercept
    initial_m = 0 #m slope
    num_iterations = 1000
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, comput_error_for_given_points)
    print "Running..."
    [b,m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, comput_error_for_given_points)
    print "b = {0}".format(b)
    print "m = {0}".format(m)
    print "after"

if __name__ == '__main__':
    run()
