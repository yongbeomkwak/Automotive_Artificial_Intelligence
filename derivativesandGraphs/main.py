# These are the libraries will be useing for this lab.

import torch
import matplotlib.pylab as plt

# Create a tensor x

x = torch.tensor(2.0, requires_grad = True)
#required_grad =True: 계산완료 후 .backward()를 통해 모든 변화도를 자동으로 계산
print("The tensor x: ", x)

# Create a tensor y according to y = x^2
y = x ** 2
print("The result of y = x^2: ", y)

# Take the derivative. Try to print out the derivative at the value x = 2
y.backward()# y=x**2 -> y=2x
print("The dervative at x = 2: ", x.grad)# y=2x -> y=2*2
print('data:',x.data)
print('grad_fn:',x.grad_fn)
print('grad:',x.grad)
print("is_leaf:",x.is_leaf)
print("requires_grad:",x.requires_grad)
print('data:',y.data)
print('grad_fn:',y.grad_fn)
print('grad:',y.grad)
print("is_leaf:",y.is_leaf)
print("requires_grad:",y.requires_grad)


#편 미분
# Calculate f(u, v) = v * u + u^2 at u = 1, v = 2
u = torch.tensor(1.0,requires_grad=True)
v = torch.tensor(2.0,requires_grad=True)
f = u * v + u ** 2
print("The result of v * u + u^2: ", f)
# Calculate the derivative with respect to u

f.backward()  # uv+u*2
print("The partial derivative with respect to u: ", u.grad) # v+2u -> 2+2(1)=4
# Calculate the derivative with respect to v
print("The partial derivative with respect to u: ", v.grad) # u -> 1

# Calculate the derivative with multiple values

x = torch.linspace(-10, 10, 10, requires_grad = True)
print("x:",x)
Y = x ** 2
y = torch.sum(x ** 2)
y.backward()
plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()

# Practice: Calculate the derivative of f = u * v + (u * v) ** 2 at u = 2, v = 1
# Type the code here
u=torch.tensor(2.0,requires_grad=True)
v=torch.tensor(1.0,requires_grad=True)
f=u*v+(u*v)**2
f.backward()
print("U.grad",u.grad) #


