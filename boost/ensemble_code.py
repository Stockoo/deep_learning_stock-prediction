# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:02:36 2018

@author: kimhongji
2/27 :
    mlp cnn rnn 각각 csv 파일 불러와서 읽는거 
    * rnn csv 못읽어옴 ㅠ 데이터 갯수가 조금씩 달라서 일단 갯수만 임의로 맞춰둠 
    
3/22 : 
    선형 데이터로 받아서 사이킷런의 Linear regression API 이용해 봤다. 
        -> 하드로 한번더 코딩해 봐야겠다 
        
04/03 : 
    MLP, CNN 은 데이터 SET 이 맞아서 같이 학습 시키면 어느정도 결과 나옴.
    RNN 데이터 SET 이 달라서 학습시키면 학습이 잘 안됨 
"""

from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import nsml

nsml.bind()
#==========================adaboost 코드=======================================
A_model = linear_model.LinearRegression()
"""
#----------------여기서 데이터 맞춰줘야함 -----------------------
"""
testY = np.loadtxt('../MLP/testy_shinhan.csv', delimiter=',')
MLP_pre = np.loadtxt('../MLP/MLP_shinhan.csv', delimiter=',')
CNN_pre = np.loadtxt('../CNN/CNN_shinhan.csv', delimiter=',')
RNN_pre = np.loadtxt('../RNN/RNN_shinhan.csv', delimiter=',')

size = len(testY)
#A_trainX = np.c_[MLP_pre,RNN_pre,CNN_pre] 
A_trainX = np.c_[MLP_pre[0:] ,(CNN_pre)[0:],(RNN_pre)[0:]]/10000
#A_trainX = np.c_[MLP_pre ,CNN_pre/10000]
A_trainY = testY/10000
A_trainY = np.reshape(A_trainY,(-1,1))
"""
#--------------------------------------------------------
"""

input_num = 3

A_train_size = int(len(A_trainY) * 0.7)
A_test_size = int(len(A_trainY) * 0.7)
A_trainX, A_testX = np.array(A_trainX[0:A_train_size]), np.array(A_trainX[A_test_size:])
A_trainY, A_testY = np.array(A_trainY[0:A_train_size]), np.array(A_trainY[A_test_size:])

X = tf.placeholder(tf.float32, shape=[None, input_num],name="A_x")
Y = tf.placeholder(tf.float32, shape=[None, 1])


W1 = tf.Variable(tf.random_normal([input_num, 10]), name='weight')
b1 = tf.Variable(tf.random_normal([10]), name='bias')
l1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([10, 8]), name='weight')
b2 = tf.Variable(tf.random_normal([8]), name='bias')
l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)

W3 = tf.Variable(tf.random_normal([8, 5]), name='weight')
b3 = tf.Variable(tf.random_normal([5]), name='bias')
l3 = tf.nn.relu(tf.matmul(l2, W3) + b3)

W4 = tf.Variable(tf.random_normal([5, 1]), name='weight')
b4 = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.add(tf.matmul(l3,W4),b4)

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#cost = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(hypothesis,Y),Y))) #MAPE


# Minimize
optimizer = tf.train.AdamOptimizer(learning_rate=0.01 ,name="A_optimizer")
train = optimizer.minimize(cost)

# Launch the graph in a session.

sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(10001):
    cost_val,_ = sess.run([cost,train], feed_dict={X: A_trainX, Y: A_trainY})
    if step % 1000 == 0:
        print(step, "Cost: ", cost_val)

#test 하는 부분
prediction = sess.run(hypothesis,feed_dict={X: A_testX})


#1. MAPE
ERROR_ans=0
ERROR=np.zeros(len(prediction))

for i in range(len(prediction)):
     ERROR[i]=np.abs((prediction[i]-A_testY[i]))/np.abs(A_testY[i])
     ERROR_ans+=ERROR[i]
print(">>MAPE Accuracy %.2f" % (ERROR_ans*100/len(prediction))) 

'''
saver = tf.train.Saver()
save_path = saver.save(sess, './model_saver_test/An_model/An')   
'''
#print prediction graph, real graph
plt.plot(prediction,label='pred')
plt.plot(A_testY,label='real')  
plt.legend()
plt.show()
