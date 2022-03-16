# These are the libraries will be used for this lab.

import torch
from torch.utils.data import Dataset
torch.manual_seed(1) # The torch.manual_seed() is for forcing the random function to give the same number every time we try to recompile it.


# Define class for dataset

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
        if self.transform: # None이 아니면
            sample = self.transform(sample) # 변환 적용
        return sample

    # Get Length
    def __len__(self): #메소드 재정의
        return self.len

# Create Dataset Object. Find out the value on index 1. Find out the length of Dataset Object.

our_dataset = toy_set()
print("Our toy_set object: ", our_dataset)
print("Value on index 0 of our toy_set object: ", our_dataset[0])
print("Our toy_set length: ", len(our_dataset))


# Create tranform class add_mult
#변환 클래스
class add_mult(object):

    # Constructor
    def __init__(self, addx=1, muly=2):
        self.addx = addx
        self.muly = muly

    # Executor
    def __call__(self, sample):
        x = sample[0] # x
        y = sample[1] # y
        x = x + self.addx  # x+1
        y = y * self.muly # y*=2
        sample = x, y
        return sample

# Create an add_mult transform object, and an toy_set object
a_m = add_mult() #
data_set = toy_set()

# Use loop to print out first 10 elements in dataset

for i in range(10):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = a_m(data_set[i]) # x는 +1 y는 *2
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)

#같은 작업을 간편하게 하기(데이터 셋 클래스에 transform을 만들어 매개변수로 넘겨준다)
# Create a new data_set object with add_mult object as transform
cust_data_set = toy_set(transform = a_m)

# Use loop to print out first 10 elements in dataset

for i in range(10):
    x, y = data_set[i] # transform이 None
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i] #transofrm이  x+1, y*=2
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)

# Run the command below when you do not have torchvision installed
# !mamba install -y torchvision
# Compose은 여러개의 transform을 차례대로 적용할 때 사용

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

# Combine the add_mult() and mult()
data_transform = transforms.Compose([add_mult(), mult()]) # add_mult한 후 mult() 적용
print("The combination of transforms (Compose): ", data_transform)

data_transform(data_set[0])
x,y=data_set[0]
x_,y_=data_transform(data_set[0])
print( 'Original x: ', x, 'Original y: ', y)
print( 'Transformed x_:', x_, 'Transformed y_:', y_)

compose_data_set = toy_set(transform = data_transform) #마찬가지로 transform에 넣으면 바로 적용
# Use loop to print out first 3 elements in dataset

for i in range(3):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
    x_co, y_co = compose_data_set[i]
    print('Index: ', i, 'Compose Transformed x_co: ', x_co ,'Compose Transformed y_co: ',y_co)



