import torch
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt


# Plot vecotrs, please keep the parameters in the same length
# @param: Vectors = [{"vector": vector variable, "name": name of vector, "color": color of the vector on diagram}]

def plotVec(vectors):
    ax = plt.axes()
    print
    # For loop to draw the vectors
    for vec in vectors:
        ax.arrow(0, 0, *vec["vector"], head_width=0.05, color=vec["color"], head_length=0.1)
        plt.text(*(vec["vector"] + 0.1), vec["name"])

    plt.ylim(-2, 2)
    plt.xlim(-2, 2)

ints_to_tensor=torch.tensor([0,1,2,3,4])
print(f'The dtype of tensor object, {ints_to_tensor.dtype} ') # torch.int64
print(f"The type of tensor object, {ints_to_tensor.type()}") # torch.LongTensor
print(f'{type(ints_to_tensor)}') # torch.Tensor

floats_to_tensor=torch.tensor([0.0,1.0,2.0,3.0,4.0])
print(f'The dtype of tensor object, {floats_to_tensor.dtype} ') # torch.float32
print(f"The type of tensor object, {floats_to_tensor.type()}") # torch.FloatTensor


old_int_tensor=torch.tensor([0,1,2,3,4])
new_float_tensor=old_int_tensor.type(torch.FloatTensor)
print(f"The type of the new_float_tensor:, {new_float_tensor.type()}") #torch.FloatTensor

print(f"size of new_float_tensor:, {new_float_tensor.size()}") # torch.Size([5])
print(f"The dimension of the new_float_tensor:, {new_float_tensor.ndimension()}") #1

twoD_float_tensor=new_float_tensor.view(5,1)
# a.view(row,column):2차원 텐서로 변환
print(f"Original Size, {new_float_tensor}") # tensor([0,1,2,3,4])
print(f"Size after view method, {twoD_float_tensor}")   # tensor([[0],[1],[2],[3],[4]])
twoD_float_tensor2=new_float_tensor.view(-1,1) # 동적으로 설정하고 싶을 때 -1(any Size)
print(f"{twoD_float_tensor2}")  # tensor([[0],[1],[2],[3],[4]]) 결과는 같음

# Convert a numpy array to a tensor

numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
new_tensor = torch.from_numpy(numpy_array) #torch.from_numpy(numpy_array): 넘파일 배열을 텐서로 변환
# 사실은  가르키는 것 numpy_array 요소 값이 바뀌면 변경됨

print("The dtype of new tensor: ", new_tensor.dtype) #torch.float64
print("The type of new tensor: ", new_tensor.type()) #torch.DoubleTensor

# Convert a tensor to a numpy array

back_to_numpy = new_tensor.numpy() #다시 텐서를 넘파일 배열로 ,이 역시 new_tensor를 가르킴 ,new_tensor 변경 시 바뀜
print("The numpy array from tensor: ", back_to_numpy)
print("The dtype of numpy array: ", back_to_numpy.dtype) #float64

# Set all elements in numpy array to zero
numpy_array[:] = 0 #

# numpy_array <- new_tensor <-back_to_numpy

print("The new tensor points to numpy_array : ", new_tensor)
print("and back to numpy array points to the tensor: ", back_to_numpy)

# Convert a panda series to a tensor

pandas_series=pd.Series([0.1, 2, 0.3, 10.1])  #array와 비슷
new_tensor=torch.from_numpy(pandas_series.values)
print("The new tensor from numpy array: ", new_tensor)
print("The dtype of new tensor: ", new_tensor.dtype)
print("The type of new tensor: ", new_tensor.type())


this_tensor=torch.tensor([0,1, 2,3])
# item(): 텐서의 값을 파이썬 기본 타입으로 변경해서 리턴
print("the first item is given by",this_tensor[0].item(),"the first tensor value is given by ",this_tensor[0])
# this_tensor[0].item(): 0   , this_tensor[0] : tensor(0)
print("the second item is given by",this_tensor[1].item(),"the second tensor value is given by ",this_tensor[1])

torch_to_list=this_tensor.tolist()  #tolist():역시 마찬가지로 파이썬 리스트로 리턴 해줌
print('tensor:', this_tensor,"\nlist:",torch_to_list)  # tensor([0, 1, 2, 3]) ,list: [0, 1, 2, 3]




# A tensor for showing how to change value according to the index
tensor_sample = torch.tensor([20, 1, 2, 3, 4])
# Change the value on the index 0 to 100
print("Inital value on index 0:", tensor_sample[0]) #tensor(20)
tensor_sample[0] = 100
print("Modified tensor:", tensor_sample)



# Slice tensor_sample
subset_tensor_sample = tensor_sample[1:4] #tensor([1,2,3]), 깊은 복사
print("Original tensor sample: ", tensor_sample)
print("The subset of tensor sample:", subset_tensor_sample)

print("Inital value on index 3 and index 4:", tensor_sample[3:5]) # tensor([3,4])
tensor_sample[3:5] = torch.tensor([300.0, 400.0]) #해당 범위 한번에 변경하기 tensor([100,1,2,300,400])
print("Modified tensor:", tensor_sample)


# Function

#Calculate the mean for math_tensor
# Sample tensor for mathmatic calculation methods on tensor

math_tensor = torch.tensor([1.0, -1.0, 1, -1])
print("Tensor example: ", math_tensor)
mean = math_tensor.mean() #평균 tensor(0.)
print("The mean of math_tensor: ", mean)
#Calculate the standard deviation for math_tensor
standard_deviation = math_tensor.std() # 편차
print("The standard deviation of math_tensor: ", standard_deviation)
# Sample for introducing max and min methods
max_min_tensor = torch.tensor([1, 1, 3, 5, 5])
# Method for finding the maximum value in the tensor
max_val = max_min_tensor.max() #tensor(5)
print("Maximum number in the tensor: ", max_val)
# Method for finding the minimum value in the tensor
min_val = max_min_tensor.min() #tensor(1)
print("Minimum number in the tensor: ", min_val)
# Method for calculating the sin result of each element in the tensor
pi_tensor = torch.tensor([0, np.pi/2, np.pi])
sin = torch.sin(pi_tensor) #싸인 연산
print("The sin result of pi_tensor: ", sin)

# First try on using linspace to create tensor
len_5_tensor = torch.linspace(-2, 2, steps = 5) #start:-2 이상 end=2미만 을 steps=5등분
print ("First Try on linspace", len_5_tensor) #tensor([-2,-1,-0,1,2])
# Second try on using linspace to create tensor
len_9_tensor = torch.linspace(-2, 2, steps = 9) #tensor([-2.0000, -1.5000, -1.0000, -0.5000,  0.0000,  0.5000,  1.0000,  1.5000,2.0000])
print ("Second Try on linspace", len_9_tensor)
# Plot sin_result
pi_tensor = torch.linspace(0, 2*np.pi, 100)
sin_result = torch.sin(pi_tensor)
plt.plot(pi_tensor.numpy(), sin_result.numpy())






# Create two sample tensors

u = torch.tensor([1, 0])
v = torch.tensor([0, 1])

# Add u and v
w = u + v
print("The result tensor: ", w) # 1,1

# tensor + scalar

u = torch.tensor([1, 2, 3, -1])
v = u + 1
print ("Addition Result: ", v) # 2,3,4,0

# tensor * tensor

u = torch.tensor([1, 2])
v = torch.tensor([3, 2])
w = u * v
print ("The result of u * v", w) # tensor([3,4])

# Calculate dot product of u, v

u = torch.tensor([1, 2])
v = torch.tensor([3, 2])

print("Dot Product of u, v:", torch.dot(u,v)) #내적 tensor(1*3+2*2)=tensor(7)


print("########################################################################")



