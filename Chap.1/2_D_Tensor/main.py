# These are the libraries will be used for this lab.

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

# Convert 2D List to 2D Tensor

twoD_list = [[11, 12, 13], [21, 22, 23], [31, 32, 33]] # 파이썬 2차원 리스트
twoD_tensor = torch.tensor(twoD_list)  # 2차원 텐서
print("The New 2D Tensor: ", twoD_tensor)

# Try tensor_obj.ndimension(), tensor_obj.shape, tensor_obj.size()

print("The dimension of twoD_tensor: ", twoD_tensor.ndimension()) # 차원 :2
print("The shape of twoD_tensor: ", twoD_tensor.shape) # torch.Size(row,col)
print("The shape of twoD_tensor: ", twoD_tensor.size()) # torch.Size(row,col)
print("The number of elements in twoD_tensor: ", twoD_tensor.numel()) # 총 데이터 개수 : 3*3=9


# Convert tensor to numpy array; Convert numpy array to tensor

twoD_numpy = twoD_tensor.numpy() #넘파이 2차원 배열로
print("Tensor -> Numpy Array:")
print("The numpy array after converting: ", twoD_numpy)
print("Type after converting: ", twoD_numpy.dtype) #int64

print("================================================")

new_twoD_tensor = torch.from_numpy(twoD_numpy) #2차원 tensor
print("Numpy Array -> Tensor:")
print("The tensor after converting:", new_twoD_tensor)
print("Type after converting: ", new_twoD_tensor.dtype) #torch.int64


# Try to convert the Panda Dataframe to tensor

df = pd.DataFrame({'a':[11,21,31],'b':[12,22,312]})

print("Pandas Dataframe to numpy: ", df.values) #numpy
print("Type BEFORE converting: ", df.values.dtype) # int64

print("================================================")

new_tensor = torch.from_numpy(df.values) #tensor
print("Tensor AFTER converting: ", new_tensor) # tensor([....])
print("Type AFTER converting: ", new_tensor.dtype) #torch.int64


# Use tensor_obj[row, column] and tensor_obj[row][column] to access certain position

tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
print("What is the value on 2nd-row 3rd-column? ", tensor_example[1, 2])  #tensor(23)
print("What is the value on 2nd-row 3rd-column? ", tensor_example[1][2])  # 같음

# Use tensor_obj[begin_row_number: end_row_number, begin_column_number: end_column number]
# and tensor_obj[row][begin_column_number: end_column number] to do the slicing

tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
print("What is the value on 1st-row first two columns? ", tensor_example[0, 0:2]) #1행 (0~1열까지) tensor([11,12])
print("What is the value on 1st-row first two columns? ", tensor_example[0][0:2]) # 같음


# Give an idea on tensor_obj[number: number][number]
sliced_tensor_example = tensor_example[1:3] # tensor([[21,22,23],[31,32,33]]) 2행~3행
print("1. Slicing step on tensor_example: ")
print("Result after tensor_example[1:3]: ", sliced_tensor_example)
print("Dimension after tensor_example[1:3]: ", sliced_tensor_example.ndimension()) #2차원
print("================================================")
print("2. Pick an index on sliced_tensor_example: ")
print("Result after sliced_tensor_example[1]: ", sliced_tensor_example[1]) #[31,32,33]
print("Dimension after sliced_tensor_example[1]: ", sliced_tensor_example[1].ndimension()) #1
print("================================================")
print("3. Combine these step together:")
print("Result: ", tensor_example[1:3][1])
print("Dimension: ", tensor_example[1:3][1].ndimension())


#부분 구역 접근
#tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
# Use tensor_obj[begin_row_number: end_row_number, begin_column_number: end_column number]
tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
print("What is the value on 3rd-column last two rows? ", tensor_example[1:3, 2]) #1~2행 2열 [23,33]


# Calculate [[1, 0], [0, 1]] + [[2, 1], [1, 2]]
X = torch.tensor([[1, 0],[0, 1]])
Y = torch.tensor([[2, 1],[1, 2]])
X_plus_Y = X + Y
print("The result of X + Y: ", X_plus_Y)

# Calculate 2 * [[2, 1], [1, 2]]

Y = torch.tensor([[2, 1], [1, 2]])
two_Y = 2 * Y
print("The result of 2Y: ", two_Y)

# Calculate [[1, 0], [0, 1]] * [[2, 1], [1, 2]]

X = torch.tensor([[1, 0], [0, 1]])
Y = torch.tensor([[2, 1], [1, 2]])
X_times_Y = X * Y
print("The result of X * Y: ", X_times_Y)


# Calculate [[0, 1, 1], [1, 0, 1]] * [[1, 1], [1, 1], [-1, 1]]
A = torch.tensor([[0, 1, 1], [1, 0, 1]])
B = torch.tensor([[1, 1], [1, 1], [-1, 1]])
A_times_B = torch.mm(A,B) # 행렬의 곱
print("The result of A * B: ", A_times_B)

# Practice: Calculate the product of two tensors (X and Y) with different sizes
# Type your code here
X = torch.tensor([[0, 1], [1, 2]])  # 2X2
Y = torch.tensor([[-1, -2, 0], [2, 1, 2]]) #2x3
X_times_Y = torch.mm(X, Y)
print("The result of X * Y: ", X_times_Y)
