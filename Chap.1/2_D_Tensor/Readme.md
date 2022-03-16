<center>
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork/Template/module%201/images/IDSNlogo.png" width="300" alt="cognitiveclass.ai logo"  />
</center>


<h1>Two-Dimensional Tensors</h1>


<h2>Objective</h2><ul><li> How to perform tensor operations on 2D tensors.</li></ul> 


<h2>Table of Contents</h2>


<p>In this lab, you will learn the basics of tensor operations on 2D tensors.</p>
<ul>
    <li><a href="https://#Types_Shape">Types and Shape </a></li>
    <li><a href="https://#Index_Slice">Indexing and Slicing</a></li>
    <li><a href="https://#Tensor_Op">Tensor Operations</a></li>
</ul>

<p>Estimated Time Needed: <b>10 min</b></p>
<hr>


<h2>Preparation</h2>


The following are the libraries we are going to use for this lab.



```python
# These are the libraries will be used for this lab.

import numpy as np 
import matplotlib.pyplot as plt
import torch
import pandas as pd
```

<!--Empty Space for separating topics-->


<h2 id="Types_Shape">Types and Shape</h2>


The methods and types for 2D tensors is similar to the methods and types for 1D tensors which has been introduced in <i>Previous Lab</i>.


Let us see how to convert a 2D list to a 2D tensor. First, let us create a 3X3 2D tensor. Then let us try to use <code>torch.tensor()</code> which we used for converting a 1D list to 1D tensor. Is it going to work?



```python
# Convert 2D List to 2D Tensor

twoD_list = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
twoD_tensor = torch.tensor(twoD_list)
print("The New 2D Tensor: ", twoD_tensor)
```

Bravo! The method <code>torch.tensor()</code> works perfectly.Now, let us try other functions we studied in the <i>Previous Lab</i>.


<!--Empty Space for separating topics-->


Let us try <code><i>tensor_obj</i>.ndimension()</code> (<code>tensor_obj</code>: This can be any tensor object), <code><i>tensor_obj</i>.shape</code>, and <code><i>tensor_obj</i>.size()</code>



```python
# Try tensor_obj.ndimension(), tensor_obj.shape, tensor_obj.size()

print("The dimension of twoD_tensor: ", twoD_tensor.ndimension())
print("The shape of twoD_tensor: ", twoD_tensor.shape)
print("The shape of twoD_tensor: ", twoD_tensor.size())
print("The number of elements in twoD_tensor: ", twoD_tensor.numel())
```

Because it is a 2D 3X3 tensor,  the outputs are correct.


<!--Empty Space for separating topics-->


Now, let us try converting the tensor to a numpy array and convert the numpy array back to a tensor.



```python
# Convert tensor to numpy array; Convert numpy array to tensor

twoD_numpy = twoD_tensor.numpy()
print("Tensor -> Numpy Array:")
print("The numpy array after converting: ", twoD_numpy)
print("Type after converting: ", twoD_numpy.dtype)

print("================================================")

new_twoD_tensor = torch.from_numpy(twoD_numpy)
print("Numpy Array -> Tensor:")
print("The tensor after converting:", new_twoD_tensor)
print("Type after converting: ", new_twoD_tensor.dtype)
```

The result shows the tensor has successfully been converted to a numpy array and then converted back to a tensor.


<!--Empty Space for separating topics-->


Now let us try to convert a Pandas Dataframe to a tensor. The process is the  Same as the 1D conversion, we can obtain the numpy array via the attribute <code>values</code>. Then, we can use <code>torch.from_numpy()</code> to convert the value of the Pandas Series to a tensor.



```python
# Try to convert the Panda Dataframe to tensor

df = pd.DataFrame({'a':[11,21,31],'b':[12,22,312]})

print("Pandas Dataframe to numpy: ", df.values)
print("Type BEFORE converting: ", df.values.dtype)

print("================================================")

new_tensor = torch.from_numpy(df.values)
print("Tensor AFTER converting: ", new_tensor)
print("Type AFTER converting: ", new_tensor.dtype)
```

<!--Empty Space for separating topics-->


<!--Empty Space for separating topics-->


<h3>Practice</h3>


Try to convert the following Pandas Dataframe  to a tensor



```python
# Practice: try to convert Pandas Series to tensor

df = pd.DataFrame({'A':[11, 33, 22],'B':[3, 3, 2]})
```

Double-click <b>here</b> for the solution.

<!--
converted_tensor = torch.tensor(df.values)
print ("Tensor: ", converted_tensor)
-->


<h2 id="Index_Slice">Indexing and Slicing</h2>


You can use rectangular brackets to access the different elements of the tensor. The correspondence between the rectangular brackets and the list and the rectangular representation is shown in the following figure for a 3X3 tensor:


<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/chapter%201/1.2index1.png" width=500 alt="Matrix Structure Introduce">


You can access the 2nd-row 3rd-column as shown in the following figure:


<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/chapter%201/1.2index.png" width="500" alt="Example of Matrix Index">


You simply use the square brackets and the indices corresponding to the element that you want.


Now, let us try to access the value on position 2nd-row 3rd-column. Remember that the index is always 1 less than how we count rows and columns. There are two ways to access the certain value of a tensor. The example in code will be the same as the example picture above.



```python
# Use tensor_obj[row, column] and tensor_obj[row][column] to access certain position

tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
print("What is the value on 2nd-row 3rd-column? ", tensor_example[1, 2])
print("What is the value on 2nd-row 3rd-column? ", tensor_example[1][2])
```

As we can see, both methods return the true value (the same value as the picture above). Therefore, both of the methods work.


<!--Empty Space for separating topics-->


Consider the elements shown in the following figure:


<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/chapter%201/1.2_index2.png" width="500" alt="Example of Matrix Index" />


Use the method above, we can access the 1st-row 1st-column by <code>tensor_example\[0]\[0]</code>



```python
tensor_example[0][0]
```

But what if we want to get the value on both 1st-row 1st-column and 1st-row 2nd-column?


You can also use slicing in a tensor. Consider the following figure. You want to obtain the 1st two columns in the 1st row:


<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/chapter%201/1.2sliceing.png" width="500" alt="Example of Matrix Index and Slicing" />


## Let us see how  we use slicing with 2D tensors to get the values in the above picture.



```python
# Use tensor_obj[begin_row_number: end_row_number, begin_column_number: end_column number] 
# and tensor_obj[row][begin_column_number: end_column number] to do the slicing

tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
print("What is the value on 1st-row first two columns? ", tensor_example[0, 0:2])
print("What is the value on 1st-row first two columns? ", tensor_example[0][0:2])
```

We get the result as <code>tensor(\[11, 12])</code> successfully.


<!--Empty Space for separating topics-->


But we <b>can't</b> combine using slicing on row and pick one column by using the code <code>tensor_obj\[begin_row_number: end_row_number]\[begin_column_number: end_column number]</code>. The reason is that the slicing will be applied on the tensor first. The result type will be a two dimension again. The second bracket will no longer represent the index of the column it will be the index of the row at that time. Let us see an example.



```python
# Give an idea on tensor_obj[number: number][number]

tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
sliced_tensor_example = tensor_example[1:3]
print("1. Slicing step on tensor_example: ")
print("Result after tensor_example[1:3]: ", sliced_tensor_example)
print("Dimension after tensor_example[1:3]: ", sliced_tensor_example.ndimension())
print("================================================")
print("2. Pick an index on sliced_tensor_example: ")
print("Result after sliced_tensor_example[1]: ", sliced_tensor_example[1])
print("Dimension after sliced_tensor_example[1]: ", sliced_tensor_example[1].ndimension())
print("================================================")
print("3. Combine these step together:")
print("Result: ", tensor_example[1:3][1])
print("Dimension: ", tensor_example[1:3][1].ndimension())
```

See the results and dimensions in 2 and 3 are the same. Both of them contains the 3rd row in the <code>tensor_example</code>, but not the last two values in the 3rd column.


<!--Empty Space for separating topics-->


So how can we get the elements in the 3rd column with the last two rows? As the below picture.


<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/chapter%201/1.2slicing2.png" width="500" alt="Example of Matrix Index and Slicing" />


Let's see the code below.



```python
# Use tensor_obj[begin_row_number: end_row_number, begin_column_number: end_column number] 

tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
print("What is the value on 3rd-column last two rows? ", tensor_example[1:3, 2])
```

Fortunately, the code <code>tensor_obj\[begin_row_number: end_row_number, begin_column_number: end_column number]</code> is still works.


<!--Empty Space for separating topics-->


<h3>Practice</h3>


Try to change the values on the second column and the last two rows to 0. Basically, change the values on <code>tensor_ques\[1]\[1]</code> and <code>tensor_ques\[2]\[1]</code> to 0.



```python
# Practice: Use slice and index to change the values on the matrix tensor_ques.

tensor_ques = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])

```

Double-click <b>here</b> for the solution.

<!--
tensor_ques[1:3, 1] = 0
print("The result: ", tensor_ques)
-->


<!--Empty Space for separating topics-->


<h2 id="Tensor_Op">Tensor Operations</h2> 


We can also do some calculations on 2D tensors.


<!--Empty Space for separating topics-->


<h3>Tensor Addition</h3>


You can also add tensors; the process is identical to matrix addition. Matrix addition of <b>X</b> and <b>Y</b> is shown in the following figure:


<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/chapter%201/1.2add.png" width="500" alt="Tensor Addition in 2D">


Let us see how tensor addition works with <code>X</code> and <code>Y</code>.



```python
# Calculate [[1, 0], [0, 1]] + [[2, 1], [1, 2]]

X = torch.tensor([[1, 0],[0, 1]]) 
Y = torch.tensor([[2, 1],[1, 2]])
X_plus_Y = X + Y
print("The result of X + Y: ", X_plus_Y)
```

Like the result shown in the picture above. The result is <code>\[\[3, 1], \[1, 3]]</code>


<!--Empty Space for separating topics-->


<h3> Scalar Multiplication </h3>


Multiplying a tensor by a scalar is identical to multiplying a matrix by a scaler. If you multiply the matrix <b>Y</b> by the scalar 2, you simply multiply every element in the matrix by 2 as shown in the figure:


<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/chapter%201/1.2scaller_mult.png" width="500" alt="The product of tensor and scalar">


Let us try to calculate the product of <b>2Y</b>.



```python
# Calculate 2 * [[2, 1], [1, 2]]

Y = torch.tensor([[2, 1], [1, 2]]) 
two_Y = 2 * Y
print("The result of 2Y: ", two_Y)
```

<!--Empty Space for separating topics-->


<h3>Element-wise Product/Hadamard Product</h3>


Multiplication of two tensors corresponds to an element-wise product or Hadamard product.  Consider matrix the <b>X</b> and <b>Y</b> with the same size. The Hadamard product corresponds to multiplying each of the elements at the same position, that is, multiplying elements with the same color together. The result is a new matrix that is the same size as matrix <b>X</b> and <b>Y</b> as shown in the following figure:


<a><img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/chapter%201/1.2tensor_pruduct.png" width=500 align="center"> </a>


The code below calculates the element-wise product of the tensor <strong>X</strong> and <strong>Y</strong>:



```python
# Calculate [[1, 0], [0, 1]] * [[2, 1], [1, 2]]

X = torch.tensor([[1, 0], [0, 1]])
Y = torch.tensor([[2, 1], [1, 2]]) 
X_times_Y = X * Y
print("The result of X * Y: ", X_times_Y)
```

This is a simple calculation. The result from the code matches the result shown in the picture.


<!--Empty Space for separating topics-->


<h3>Matrix Multiplication </h3>


We can also apply matrix multiplication to two tensors, if you have learned linear algebra, you should know that in the multiplication of two matrices order matters. This means if <i>X \* Y</i> is valid, it does not mean <i>Y \* X</i> is valid. The number of columns of the matrix on the left side of the multiplication sign must equal to the number of rows of the matrix on the right side.


First, let us create a tensor <code>X</code> with size 2X3. Then, let us create another tensor <code>Y</code> with size 3X2. Since the number of columns of <code>X</code> is equal to the number of rows of <code>Y</code>. We are able to perform the multiplication.


We use <code>torch.mm()</code> for calculating the multiplication between tensors with different sizes.



```python
# Calculate [[0, 1, 1], [1, 0, 1]] * [[1, 1], [1, 1], [-1, 1]]

A = torch.tensor([[0, 1, 1], [1, 0, 1]])
B = torch.tensor([[1, 1], [1, 1], [-1, 1]])
A_times_B = torch.mm(A,B)
print("The result of A * B: ", A_times_B)
```

<!--Empty Space for separating topics-->


<h3>Practice</h3>


Try to create your own two tensors (<code>X</code> and <code>Y</code>) with different sizes, and multiply them.



```python
# Practice: Calculate the product of two tensors (X and Y) with different sizes 

# Type your code here
```

Double-click <b>here</b> for the solution.

<!--
X = torch.tensor([[0, 1], [1, 2]])
Y = torch.tensor([[-1, -2, 0], [2, 1, 2]])
X_times_Y = torch.mm(X, Y)
print("The result of X * Y: ", X_times_Y)
-->


<a href="https://dataplatform.cloud.ibm.com/registration/stepone?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0110ENSkillsNetwork20647811-2021-01-01&context=cpdaas&apps=data_science_experience%2Cwatson_machine_learning"><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork/Template/module%201/images/Watson_Studio.png"/></a>


<!--Empty Space for separating topics-->


<h2>About the Authors:</h2> 

<a href="https://www.linkedin.com/in/joseph-s-50398b136/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0110ENSkillsNetwork20647811-2021-01-01">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.


Other contributors: <a href="https://www.linkedin.com/in/michelleccarey/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0110ENSkillsNetwork20647811-2021-01-01">Michelle Carey</a>, <a href="https://www.linkedin.com/in/jiahui-mavis-zhou-a4537814a?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0110ENSkillsNetwork20647811-2021-01-01">Mavis Zhou</a>


## Change Log

| Date (YYYY-MM-DD) | Version | Changed By | Change Description                                          |
| ----------------- | ------- | ---------- | ----------------------------------------------------------- |
| 2020-09-21        | 2.0     | Shubham    | Migrated Lab to Markdown and added to course repo in GitLab |


<hr>


## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>

