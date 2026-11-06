#underfitting and overfitting
# to visulise bias-cariance with polynomial regression

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)
x = np.linspace(0,10,40)
y = np.sin(x)+np.random.normal(0,0.3,size=x.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=42)

plt.scatter(x_train,y_train,label='Train')
plt.scatter(x_test,y_test,label='Test')
plt.legend()
plt.title('sample dataset')
plt.show()

# case 1- unffiting (degree 1)
p1 = Polynomial.fit(x_train,y_train,deg=1)
x_line = np.linspace(0,10,200)

plt.scatter(x_train,y_train,label='Train Data')
plt.plot(x_line,p1(x_line),color='red',label='Degree 1 fit')
plt.legend()
plt.title('underfitting example')
plt.show()

print('train MSE',mean_squared_error(y_train,p1(x_train)))
print('test MSE',mean_squared_error(y_test,p1(x_test)))

##case 2- good fit (degree 4)

p2 = Polynomial.fit(x_train,y_train,deg=4)

plt.scatter(x_train,y_train,label='Train Data')
plt.plot(x_line,p2(x_line),color='green',label='Degree 4 fit')
plt.legend()
plt.title('good fit example')
plt.show()

print('train MSE',mean_squared_error(y_train,p2(x_train)))
print('Test MSE',mean_squared_error(y_test,p2(x_test)))

###case 3:overfitting (degree 15)
p3 = Polynomial.fit(x_train,y_train,deg=15)

plt.scatter(x_train,y_train,label='Train Data')
plt.plot(x_line,p3(x_line),color='purple',label='degree 15 fit')
plt.ylim(-2,2)
plt.legend()
plt.title('overfitting exxample')
plt.show()
print('Train MSE',mean_squared_error(y_train,p3(x_train)))
print('Test MSE',mean_squared_error(y_test,p3(x_test)))

#bias-variance curve across degrees

degrees = range(1,10)
train_errors = []
test_errors = []

for d in degrees:
    p = Polynomial.fit(x_train,y_train,deg=4)
    train_errors.append(mean_squared_error(y_train,p(x_train)))
    test_errors.append(mean_squared_error(y_test,p(x_test)))

plt.plot(degrees,train_errors,marker='o',label='Train Error')
plt.plot(degrees,test_errors,marker='o',label='Test Error')

plt.xlabel('polynomial degree')
plt.ylabel('mean square error')
plt.title('underfitting vs overfitting trade-off')
plt.legend()
plt.show()