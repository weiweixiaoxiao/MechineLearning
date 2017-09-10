# coding:utf-8
import numpy as np

#非线性函数与其导数 compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
#covert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1 - output)

class Data(object):
    def __init__(self,data):
        self.data = data        
    def calculate(self):
        alpha = 0.75
        input_dim = 8
        hidden_dim = 10
        output_dim = 1
        v = np.matrix(np.random.random([input_dim,hidden_dim]))  #输入到第一层隐藏层的权重     
        w = np.matrix(np.random.random([hidden_dim,output_dim]))  #隐藏层到输出的权重
        r = np.matrix(np.random.random([hidden_dim,1]))
        l = np.matrix(np.random.random([output_dim,1]))
        #权重更新
        v_updata = np.matrix(np.random.random([input_dim,hidden_dim]))  #输入到第一层隐藏层的权重
        w_updata = np.matrix(np.random.random([hidden_dim,output_dim]))   #隐藏层到输出的权重
        r_updata = np.matrix(np.random.random([hidden_dim,1]))
        l_updata = np.matrix(np.random.random([output_dim,1]))
        maxnum = 1000
        for i in range(maxnum):
            for i in self.data:
                X0 = i[:-1]
                X = np.matrix(X0)
                Y0 = i[-1]
                Y = np.matrix(Y0)
           
                b = sigmoid(v.T * X.T-r) #隐藏层输出，输出层输入            
                y = sigmoid(b.T * w - l) #输出层输出
                g = y*(1-y)*(Y-y) #1*1
                e = b.T*(1-b)*(w*g).T  #1*10
         
                v_updata =  alpha*X.T*e #8*10
                w_updata = alpha*b*g    #10*1
                r_updata = -alpha*e.T
                l_updata = -alpha*g
                v = v + v_updata
                w = w + w_updata
                r = r + r_updata
                l = l + l_updata
                #print("y == ",y)
                #print("Y== ",Y)
                error = 1/2*(y - Y)*(y-Y)
                print("error == ",error)     
               
D = np.array([
    [1, 1, 1, 1, 1, 1, 0.697, 0.460, 1],
    [3, 1, 1, 3, 3, 1, 0.593, 0.042, 0],
    [2, 1, 2, 1, 1, 1, 0.774, 0.376, 1],
    [1, 1, 2, 1, 1, 1, 0.608, 0.318, 1],
    [1, 2, 1, 1, 2, 2, 0.403, 0.237, 1],
    [2, 2, 1, 1, 2, 1, 0.437, 0.211, 1],
    [2, 2, 2, 2, 2, 1, 0.666, 0.091, 0],
    [3, 1, 1, 1, 1, 1, 0.556, 0.215, 1],
    [1, 3, 3, 1, 3, 2, 0.243, 0.267, 0],
    [3, 3, 3, 3, 3, 1, 0.245, 0.057, 0],
    [2, 2, 1, 2, 2, 2, 0.481, 0.149, 1],
    [3, 1, 1, 3, 3, 2, 0.343, 0.099, 0],
    [1, 2, 1, 2, 1, 1, 0.639, 0.161, 0],
    [3, 2, 2, 2, 1, 1, 0.657, 0.198, 0],
    [2, 1, 1, 1, 1, 1, 0.634, 0.264, 1],
    [2, 2, 1, 1, 2, 2, 0.360, 0.370, 0],
    [1, 1, 2, 2, 2, 1, 0.719, 0.103, 0]])

f = Data(D)
f.calculate()