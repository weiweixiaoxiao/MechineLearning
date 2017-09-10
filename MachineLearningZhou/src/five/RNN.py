# -*- coding:utf-8 -*-
import copy,numpy as np
np.random.seed(0) #设定随机数生成的种子

#非线性函数与其导数 compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
#covert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1 - output)

#training dataset generation
int2binary = {} #这一行声明了一个查找表，这个表是一个实数与对应二进制表示的映射。二进制表将会是我们网路的输入与输出，所以这个查找表将会帮助我们实现转化为其二进制表示
binary_dim = 8  #设置二进制数的最大长度，如果一切都调试好了，可以把它调整为一个非常大的数

largest_number = pow(2,binary_dim) #计算跟二进制最大长度对应的可以表示的最大十进制数
#生成了十进制数转二进制数的查找表，并将其复制到int2binary里面，虽然说这一步不是必须的，但是这样理解会更方便
binary = np.unpackbits(
    np.array([range(largest_number)],dtype = np.uint8).T,axis = 1)
for i in range(largest_number):
    int2binary[i] = binary[i]

#input variables
alpha = 0.1
input_dim = 2   #要把两个数加起来，所以一次输入两位字符，网络需要两个输入
hidden_dim = 16
output_dim = 1  #只是预测和的值，也就是一个数，如此只需一个输出

# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim))-1
synapse_1 = 2*np.random.random((hidden_dim,output_dim))-1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim))-1

#这里存储权值更新，在我们积累了一个权值更新以后，再去更新权值
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

#training logic
for j in range(10000):
    #generate a simple addition problem(a+b=c)
    a_int = np.random.randint(largest_number/2) #int version 随机生成一个在范围内的加法问题，所以我们生成一个在0到最大值一半之间的整数。如果我们允许网络的表示超过这个范围，那么把两个数加起来就有可能溢出
    a = int2binary[a_int] #binary encoding 查找a_int对应的二进制表示，然后存进啊里面
    b_int = np.random.randint(largest_number/2) #int version 随机生成一个在范围内的加法问题，所以我们生成一个在0到最大值一半之间的整数。如果我们允许网络的表示超过这个范围，那么把两个数加起来就有可能溢出
    b = int2binary[a_int] #binary encoding 查找a_int对应的二进制表示，然后存进啊里面
    #true answer
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    #where we'll store our best guess(binary encoded)
    d = np.zeros_like(c) #初始化一个空的二进制数组，用来存储神经网络的预测值（便于我们后面输出）
    
    overallError = 0 #重置误差值
    
    #这两个list会每个时刻不断记录layer2的导数值与layer1的值
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim)) #在0时刻没有之前的隐含层，初始化一个全为0
    
    #moving along the positions in the binary encoding
    #这个循环是遍历二进制数字
    for position in range(binary_dim):
        #generate input and output
        x = np.array([[a[binary_dim-position-1],b[binary_dim-position-1]]])
        y = np.array([[c[binary_dim-position-1]]]).T
        
        #hidden layer(input~+prev_hidden)
        layer_1 = sigmoid(np.dot(x,synapse_0)+np.dot(layer_1_values[-1],synapse_h))
        
        #output layer(new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))
        
        #计算预测误差
        layer_2_error = y - layer_2
        #把导数的值存起来，即把每个时刻的导数值都保存着 
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])#计算误差的绝对值，并把他们加起来，这样我们就得到一个误差的标量（用来衡量传播）最后会得到所有二进制位的误差的总和
        
        #decode estimate so we can print it out
        d[binary_dim-position-1] = np.round(layer_2[0][0])
        
        #store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))
        
    future_layer_1_delta = np.zeros(hidden_dim)
    
    #我们已经完成了所有的正向传播，并且已经计算了输出层的导数，并将其存入在一个列表中了，现在我们需要做的就是反向传播，从最后一个时间点开始，反向一直到第一个
    for position in range(binary_dim):
        x = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]  #从列表取出当前的隐含层
        prev_layer_1 = layer_1_values[-position-2] #从列表取出前一个的隐含层
        #从列表中取出当前输出层的误差
        layer_2_delta = layer_2_deltas[-position-1]
        #error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T)+\
                         layer_2_delta.dot(synapse_1.T)*sigmoid_output_to_derivative(layer_1))
        
        #let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += x.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
        
    synapse_0 += synapse_0_update*alpha
    synapse_1 += synapse_1_update*alpha
    synapse_h += synapse_h_update*alpha
    
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
    # print out progress  
    if(j % 1000 == 0):  
        print ("Error:" + str(overallError))
        print ("Pred:" + str(d))
        print ("True:" + str(c))  
        out = 0  
        for index,x in enumerate(reversed(d)):  
            out += x*pow(2,index)  
        print (str(a_int) + " + " + str(b_int) + " = " + str(out))
        print ("------------" )
        
        
    















































