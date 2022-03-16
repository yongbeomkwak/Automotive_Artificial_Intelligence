<center>
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork/Template/module%201/images/IDSNlogo.png" width="300" alt="cognitiveclass.ai logo"  />
</center>


<h1>Simple Dataset</h1> 


<h2>Objective</h2><ul><li> How to create a dataset in pytorch.</li><li> How to perform transformations on the dataset.</li></ul> 


<h2>Table of Contents</h2>


<p>In this lab, you will construct a basic dataset by using PyTorch and learn how to apply basic transformations to it.</p> 
<ul>
    <li><a href="https://#Simple_Dataset">Simple dataset</a></li>
    <li><a href="https://#Transforms">Transforms</a></li>
    <li><a href="https://#Compose">Compose</a></li>
</ul>
<p>Estimated Time Needed: <strong>30 min</strong></p>
<hr>


<h2>Preparation</h2>


The following are the libraries we are going to use for this lab. The <code>torch.manual_seed()</code> is for forcing the random function to give the same number every time we try to recompile it.



```python
# These are the libraries will be used for this lab.

import torch
from torch.utils.data import Dataset
torch.manual_seed(1)
```

<!--Empty Space for separating topics-->


<h2 id="Simple_Dataset">Simple dataset</h2>


Let us try to create our own dataset class.



```python
# Define class for dataset

class toy_set(Dataset):
    
    # Constructor with defult values 
    def __init__(self, length = 100, transform = None):
        self.len = length
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform
     
    # Getter
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)     
        return sample
    
    # Get Length
    def __len__(self):
        return self.len
```

Now, let us create our <code>toy_set</code> object, and find out the value on index 1 and the length of the inital dataset



```python
# Create Dataset Object. Find out the value on index 1. Find out the length of Dataset Object.

our_dataset = toy_set()
print("Our toy_set object: ", our_dataset)
print("Value on index 0 of our toy_set object: ", our_dataset[0])
print("Our toy_set length: ", len(our_dataset))
```

As a result, we can apply the same indexing convention as a <code>list</code>,
and apply the fuction <code>len</code> on the <code>toy_set</code> object. We are able to customize the indexing and length method by <code>def \__getitem\_\_(self, index)</code> and <code>def \__len\_\_(self)</code>.


Now, let us print out the first 3 elements and assign them to x and y:



```python
# Use loop to print out first 3 elements in dataset

for i in range(3):
    x, y=our_dataset[i]
    print("index: ", i, '; x:', x, '; y:', y)
```

The dataset object is an Iterable; as a result, we  apply the loop directly on the dataset object



```python
for x,y in our_dataset:
    print(' x:', x, 'y:', y)
```

<!--Empty Space for separating topics-->


<h3>Practice</h3>


Try to create an <code>toy_set</code> object with length <b>50</b>. Print out the length of your object.



```python
# Practice: Create a new object with length 50, and print the length of object out.

# Type your code here
```

Double-click <b>here</b> for the solution.

<!-- 
my_dataset = toy_set(length = 50)
print("My toy_set length: ", len(my_dataset))
-->


<!--Empty Space for separating topics-->


<h2 id="Transforms">Transforms</h2>


You can also create a class for transforming the data. In this case, we will try to add 1 to x and multiply y by 2:



```python
# Create tranform class add_mult

class add_mult(object):
    
    # Constructor
    def __init__(self, addx = 1, muly = 2):
        self.addx = addx
        self.muly = muly
    
    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.addx
        y = y * self.muly
        sample = x, y
        return sample
```

<!--Empty Space for separating topics-->


Now, create a transform object:.



```python
# Create an add_mult transform object, and an toy_set object

a_m = add_mult()
data_set = toy_set()
```

Assign the outputs of the original dataset to <code>x</code> and <code>y</code>. Then, apply the transform <code>add_mult</code> to the dataset and output the values as <code>x\_</code> and <code>y\_</code>, respectively:



```python
# Use loop to print out first 10 elements in dataset

for i in range(10):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = a_m(data_set[i])
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
```

As the result, <code>x</code> has been added by 1 and y has been multiplied by 2, as <i>\[2, 2] + 1 = \[3, 3]</i> and <i>\[1] x 2 = \[2]</i>


<!--Empty Space for separating topics-->


We can apply the transform object every time we create a new <code>toy_set object</code>? Remember, we have the constructor in toy_set class with the parameter <code>transform = None</code>.
When we create a new object using the constructor, we can assign the transform object to the parameter transform, as the following code demonstrates.



```python
# Create a new data_set object with add_mult object as transform

cust_data_set = toy_set(transform = a_m)
```

This applied <code>a_m</code> object (a transform method) to every element in <code>cust_data_set</code> as initialized. Let us print out the first 10 elements in <code>cust_data_set</code> in order to see whether the <code>a_m</code> applied on <code>cust_data_set</code>



```python
# Use loop to print out first 10 elements in dataset

for i in range(10):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
```

The result is the same as the previous method.


<!--Empty Space for separating topics-->



```python
# Practice: Construct your own my_add_mult transform. Apply my_add_mult on a new toy_set object. Print out the first three elements from the transformed dataset.

# Type your code here.
```

Double-click <b>here</b> for the solution.

<!-- 
class my_add_mult(object):   
    def __init__(self, add = 2, mul = 10):
        self.add=add
        self.mul=mul
        
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.add
        y = y + self.add
        x = x * self.mul
        y = y * self.mul
        sample = x, y
        return sample
        
       
my_dataset = toy_set(transform = my_add_mult())
for i in range(3):
    x_, y_ = my_dataset[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
    
 -->


<!--Empty Space for separating topics-->


<h2 id="Compose">Compose</h2>


You can compose multiple transforms on the dataset object. First, import <code>transforms</code> from <code>torchvision</code>:



```python
# Run the command below when you do not have torchvision installed
# !mamba install -y torchvision

from torchvision import transforms
```

Then, create a new transform class that multiplies each of the elements by 100:



```python
# Create tranform class mult

class mult(object):
    
    # Constructor
    def __init__(self, mult = 100):
        self.mult = mult
        
    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mult
        y = y * self.mult
        sample = x, y
        return sample
```

Now let us try to combine the transforms <code>add_mult</code> and <code>mult</code>



```python
# Combine the add_mult() and mult()

data_transform = transforms.Compose([add_mult(), mult()])
print("The combination of transforms (Compose): ", data_transform)

```

The new <code>Compose</code> object will perform each transform concurrently as shown in this figure:


<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/chapter%201/1.3.1_trasform.png" width="500" alt="Compose PyTorch">



```python
data_transform(data_set[0])
```


```python
x,y=data_set[0]
x_,y_=data_transform(data_set[0])
print( 'Original x: ', x, 'Original y: ', y)

print( 'Transformed x_:', x_, 'Transformed y_:', y_)
```

Now we can pass the new <code>Compose</code> object (The combination of methods <code>add_mult()</code> and <code>mult</code>) to the constructor for creating <code>toy_set</code> object.



```python
# Create a new toy_set object with compose object as transform

compose_data_set = toy_set(transform = data_transform)
```

Let us print out the first 3 elements in different <code>toy_set</code> datasets in order to compare the output after different transforms have been applied:



```python
# Use loop to print out first 3 elements in dataset

for i in range(3):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
    x_co, y_co = compose_data_set[i]
    print('Index: ', i, 'Compose Transformed x_co: ', x_co ,'Compose Transformed y_co: ',y_co)
```

Let us see what happened on index 0. The original value of <code>x</code> is <i>\[2, 2]</i>, and the original value of <code>y</code> is \[1]. If we only applied <code>add_mult()</code> on the original dataset, then the <code>x</code> became <i>\[3, 3]</i> and y became <i>\[2]</i>. Now let us see what is the value after applied both <code>add_mult()</code> and <code>mult()</code>. The result of x is <i>\[300, 300]</i> and y is <i>\[200]</i>. The calculation which is equavalent to the compose is <i> x = (\[2, 2] + 1) x 100 = \[300, 300], y = (\[1] x 2) x 100 = 200</i>


<h3>Practice</h3>


Try to combine the <code>mult()</code> and <code>add_mult()</code> as <code>mult()</code> to be executed first. And apply this on a new <code>toy_set</code> dataset. Print out the first 3 elements in the transformed dataset.



```python
# Practice: Make a compose as mult() execute first and then add_mult(). Apply the compose on toy_set dataset. Print out the first 3 elements in the transformed dataset.

# Type your code here.
```

Double-click <b>here</b> for the solution.

<!--
my_compose = transforms.Compose([mult(), add_mult()])
my_transformed_dataset = toy_set(transform = my_compose)
for i in range(3):
    x_, y_ = my_transformed_dataset[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
-->


<a href="https://dataplatform.cloud.ibm.com/registration/stepone?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0110ENSkillsNetwork20647811-2021-01-01&context=cpdaas&apps=data_science_experience%2Cwatson_machine_learning"><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork/Template/module%201/images/Watson_Studio.png"/></a>


<h2>About the Authors:</h2> 

<a href="https://www.linkedin.com/in/joseph-s-50398b136/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0110ENSkillsNetwork20647811-2021-01-01">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.


Other contributors: <a href="https://www.linkedin.com/in/michelleccarey/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0110ENSkillsNetwork20647811-2021-01-01">Michelle Carey</a>, <a href="https://www.linkedin.com/in/jiahui-mavis-zhou-a4537814a?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0110ENSkillsNetwork20647811-2021-01-01">Mavis Zhou</a>


## Change Log

| Date (YYYY-MM-DD) | Version | Changed By | Change Description                                          |
| ----------------- | ------- | ---------- | ----------------------------------------------------------- |
| 2020-09-21        | 2.0     | Shubham    | Migrated Lab to Markdown and added to course repo in GitLab |


<hr>


## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>

