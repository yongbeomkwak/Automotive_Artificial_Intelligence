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

#2_D Tensor

# Practice: Calculate the product of two tensors (X and Y) with different sizes
# Type your code here
X = torch.tensor([[0, 1], [1, 2]])  # 2X2
Y = torch.tensor([[-1, -2, 0], [2, 1, 2]]) #2x3
X_times_Y = torch.mm(X, Y)
print("The result of X * Y: ", X_times_Y)


#Derivatives
# Practice: Calculate the derivative of y = 2x^3 + x at x = 1
# Type your code here

x=torch.tensor(1.0,requires_grad=True)
y=2*x**3+x
y.backward()
print("x.grad",x.grad) # 6x**2 +1

# Practice: Calculate the derivative of f = u * v + (u * v) ** 2 at u = 2, v = 1
# Type the code here
u=torch.tensor(2.0,requires_grad=True)
v=torch.tensor(1.0,requires_grad=True)
f=u*v+(u*v)**2
f.backward()
print("U.grad",u.grad) #

#Build Data Set
from torch.utils.data import Dataset
# Practice: Create a new object with length 50, and print the length of object out.
# Type your code here

class toy_set(Dataset): # 반드시 Dataset을 반드시 상속 받아야 함

    # Constructor with defult values
    def __init__(self, length=100, transform=None):
        self.len = length
        self.x = 2 * torch.ones(length, 2) #2*tesor([1,1])가 100개=[[2,2]......]
        self.y = torch.ones(length, 1)  #tensor([1]) [[1,1]......]
        self.transform = transform

    # Getter
    def __getitem__(self, index): #메소드 재정의 #해당 [index] 요소를  tuple로 넣어 리턴
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    # Get Length
    def __len__(self): #메소드 재정의
        return self.len

myDataSet=toy_set(length=50)
print(f'Len: {len(myDataSet)}')

# Practice: Construct your own my_add_mult transform. Apply my_add_mult on a new toy_set object. Print out the first three elements from the transformed dataset.
# Type your code here.
class My_add_mult(object):

    def __init__(self,add=2,mul=3):
        self.add=add
        self.mul=mul

    def __call__(self,sample):
        x=sample[0]
        y=sample[1]
        x=x+self.add
        x=x*self.mul
        y=y+self.add
        y=y*self.mul
        sample=x,y
        return sample
my_add_mult=My_add_mult()
my_data_set=toy_set(transform=my_add_mult)
for i in range(3):
    x_,y_=my_data_set[i]
    print(f'Index:{i} Transformed x_:{x_}, Transformed y_:{y_}')

from torchvision import transforms


# Create tranform class mult

class mult(object):

    # Constructor
    def __init__(self, mult=100):
        self.mult = mult

    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mult
        y = y * self.mult
        sample = x, y
        return sample
# Practice: Make a compose as mult() execute first and then add_mult(). Apply the compose on toy_set dataset. Print out the first 3 elements in the transformed dataset.

# Type your code here.
my_compose=transforms.Compose([mult(),My_add_mult()]) #mult 먼저
my_transformed_data_set=toy_set(transform=my_compose)
for i in range(3):
    x_, y_ = my_transformed_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)


