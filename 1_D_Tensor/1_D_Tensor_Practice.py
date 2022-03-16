# These are the libraries will be used for this lab.

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


#Construct a tensor with 25 steps in the range 0 to π/2. Print out the Maximum and Minimum number. Also, plot a graph showing the diagram that shows the result.
# Practice: Create your tensor, print max and min number, plot the sin result diagram
# Type your code here

x_axis_tensor = torch.linspace(0, np.pi/2, 25)
print(x_axis_tensor)
print("Max Number: ", x_axis_tensor.max())
print("Min Number", x_axis_tensor.min())
y_axis_tensor = torch.sin(x_axis_tensor)
plt.plot(y_axis_tensor.numpy(), y_axis_tensor.numpy())
#plt.show()

#Convert the list [-1, 1] and [1, 1] to tensors u and v. Then, plot the tensor u and v as a vector by using the function plotVec and find the dot product:
# Practice: calculate the dot product of u and v, and plot out two vectors
# Type your code here
# Plot vecotrs, please keep the parameters in the same length
# @param: Vectors = [{"vector": vector variable, "name": name of vector, "color": color of the vector on diagram}]

def plotVec(vectors):
    ax = plt.axes()

    # For loop to draw the vectors
    for vec in vectors:
        ax.arrow(0, 0, *vec["vector"], head_width=0.05, color=vec["color"], head_length=0.1)
        plt.text(*(vec["vector"] + 0.1), vec["name"])

    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.show()

u_list=[-1,1]
v_list=[1,1]
u=torch.tensor(u_list)
v=torch.tensor(v_list)
plotVec([
    {"vector": u.numpy(), "name": 'u', "color": 'r'},
    {"vector": v.numpy(), "name": 'v', "color": 'b'}
])
print(f'Dot Product: {np.dot(u,v)}')



