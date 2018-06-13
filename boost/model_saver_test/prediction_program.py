# -*- coding: utf-8 -*-
"""
Created on Wed May  9 12:02:43 2018

@author: kimhongji
"""
import numpy as np
import tensorflow as tf
import talib as ta

time_len = 10            #몇일뒤 예측 날짜 인지 
predict_len = 2          #talib parameter
train_index = 1         #nan 제거
M_input_n = 12

xy = np.loadtxt('shinhan_test.csv', delimiter=',')
    
t_open = np.transpose(xy[:,[0]])
t_high =  np.transpose(xy[:,[1]])#x,y 행렬 전치 해줌 ( 그래야 talib가 먹힘 )
t_low = np.transpose(xy[:,[2]])
t_vol = np.transpose(xy[:,[3]])
t_y = np.transpose(xy)

open = np.array(t_open[0],dtype='float')
high = np.array(t_high[0],dtype='float')
low = np.array(t_low[0],dtype='float')
volume = np.array(t_vol[0],dtype='float')
close = np.array(t_y[0],dtype='float')


b_upper, b_middle, b_lower = ta.BBANDS(close)                            #bollinger ban
ma = ta.MA(close,timeperiod=predict_len)                                 #moving average
dmi = ta.DX(high,low,close,timeperiod=predict_len)                       #direct movement index
macd, macdsignal, macdhist = ta.MACD(close,fastperiod=predict_len, slowperiod=predict_len*2,signalperiod=4)                         #moving average convergence/divergence
slowk, slowd = ta.STOCH(high, low, close, fastk_period=predict_len, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)  #Stochastic
mom = ta.MOM(close,timeperiod=predict_len)
rsi = ta.RSI(close,timeperiod=predict_len*2)                             #Relative Strength Index
cci = ta.CCI(high,low,close,timeperiod=predict_len*2)
sma = ta.SMA(close,timeperiod=predict_len)                               #Simple Moving average
wma = ta.WMA(close,timeperiod=predict_len)


MLP_x = np.c_[close,ma,b_upper, b_middle, b_lower,macd, macdsignal, macdhist ,dmi,cci,sma,wma]/10000
CNN_x = np.c_[ma,b_upper, b_middle, b_lower,dmi,cci,sma,wma,mom,close]
RNN_x = np.c_[b_upper, b_middle, b_lower, rsi, sma, wma]/100000



#MLP restore=======================
M_X = tf.placeholder(tf.float32, [None, M_input_n], name="M_X")#10개의 값이 10일치 가 1-set
M_Y = tf.placeholder(tf.float32, [None, 1])

#총 3개의 hidden layer 와 입력 출력 layer 가 있다
#layer 1 : 입력 레이어 열의 개수가 input_n개로 들어가 15개의 전달 값을 갖는다.
M_W1 = tf.Variable(tf.random_uniform([M_input_n, 15],-1.,1.))
M_b1 = tf.Variable(tf.random_normal([15]))
M_L1 = tf.nn.relu(tf.matmul(M_X, M_W1)+M_b1)

#layer 2 : 은닉1 레이어 열의 개수가 15개로 들어가 10개의 전달 값을 갖는다. 
M_W2 = tf.Variable(tf.random_normal([15, 10]))
M_b2 = tf.Variable(tf.random_normal([10]))
M_L2 = tf.nn.relu(tf.matmul(M_L1, M_W2)+M_b2)

#layer 3 : 은닉3 레이어 열의 개수가 10개로 들어가 6개의 전달 값을 갖는다. 
M_W3 = tf.Variable(tf.random_normal([10, 6]))
M_b3 = tf.Variable(tf.random_normal([6]))
M_L3 = tf.nn.relu(tf.matmul(M_L2, M_W3)+M_b3)

#layer 5 :출력 레이어 열의 개수가 6개로 들어가 4개의 전달 값을 갖는다. 
M_W5 = tf.Variable(tf.random_normal([6, 1]))
M_b5 = tf.Variable(tf.random_normal([1]))
M_model = tf.add(tf.matmul(M_L3,M_W5),M_b5,name="M_model")

#====SAVER/RESTORE======================
saver = tf.train.Saver()
sess=tf.Session()

saver.restore(sess,tf.train.latest_checkpoint("./MLP_model/"))
tf.reset_default_graph()
# Now, let's access and create placeholders variables and
# create feed-dict to feed new data
M_prediction  = sess.run(M_model,feed_dict ={M_X:MLP_x})
#=================================










#CNN restore=======================
C_X = tf.placeholder(tf.float32, [10, 10], name="C_X")#10*10
C_X_img = tf.reshape(C_X, [-1, 1, 10, 10])#292일 10개 종목 data를 이미지 처럼
C_Y = tf.placeholder(tf.float32, [None, 1])#마지막 결과 Y 하나로
C_K =tf.placeholder(tf.int32)

#layer1ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
C_W1 = tf.Variable(tf.random_normal([2, 2,10, 120], stddev=0.01))#10 -> 100
C_L1 = tf.nn.conv2d(C_X_img, C_W1, strides=[1, 1, 1, 1], padding='SAME')
C_L1 = tf.nn.relu(C_L1)
C_L1 = tf.nn.max_pool(C_L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')#pooling 끝나면 10/5 -> 2로


#layer2ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
C_W2 = tf.Variable(tf.random_normal([2, 2, 120, 200], stddev=0.01))#100 -> 200로 만들어 주는 필터
C_L2 = tf.nn.conv2d(C_L1, C_W2, strides=[1, 1, 1, 1], padding='SAME')
C_L2 = tf.nn.relu(C_L2)
C_L2 = tf.nn.max_pool(C_L2, ksize=[1, 2, 2, 1],strides=[1, 5, 5, 1], padding='SAME')#pooling 끝나면 2/2 -> 1로

C_L2_flat = tf.reshape(C_L2, [-1,200])#fully connected로 만들어주기


# Final FC 200inputs -> 1 outputsㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
C_W3 = tf.get_variable("C_W3", shape=[200,1], initializer=tf.contrib.layers.xavier_initializer())
C_b = tf.Variable(tf.random_normal([1]))
C_logits = tf.add(tf.matmul(C_L2_flat,C_W3), C_b,name="C_model")#1*512 - 512*1 해서 1*1

#====SAVER/RESTORE======================
saver = tf.train.Saver()
sess=tf.Session()

saver.restore(sess,tf.train.latest_checkpoint("./CNN_model/"))
tf.reset_default_graph()
# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

C_prediction1  = sess.run(C_logits,feed_dict ={C_X:CNN_x[0:10]})
C_prediction2  = sess.run(C_logits,feed_dict ={C_X:CNN_x[10:-1]})
#===================================










#RNN restore=======================
R_seq_length = 10         #10일 데이치를 넣어 
R_after = 9               #그로부터 10일뒤 종가 예측
R_data_dim = 6           #지표로 변환하여 6개가 들어감
R_hidden_dim = 10         #lstm에서 output은 10개가 나옴
R_output_dim = 1          #FC를 통해 10개에서 1개로 

R_X = tf.placeholder(tf.float32, [None, R_seq_length, R_data_dim] , name="R_X")
R_Y = tf.placeholder(tf.float32, [None, 1])

R_dataX = []
_x = RNN_x[10: -1]/10
R_dataX.append(_x)
    

# build a LSTM network
R_cell = tf.contrib.rnn.BasicLSTMCell(num_units=R_hidden_dim, state_is_tuple=True, activation=tf.tanh) #lstm 셀 생성
R_outputs, R_states = tf.nn.dynamic_rnn(R_cell, R_X, dtype=tf.float32) #cell output node
#마지막 lstm output(10개)을 FC를 통해 한개로 출력
R_Y_pred = tf.contrib.layers.fully_connected(R_outputs[:, -1], R_output_dim, activation_fn=None) 

#====SAVER/RESTORE======================
saver = tf.train.Saver()
sess=tf.Session()

saver.restore(sess,tf.train.latest_checkpoint("./RNN_model/"))
tf.reset_default_graph()
# Now, let's access and create placeholders variables and
# create feed-dict to feed new data
R_prediction  = sess.run(R_Y_pred,feed_dict ={R_X:R_dataX})
#===================================







DATA= np.c_[M_prediction[19]*100000 ,C_prediction2*100, R_prediction*1000000]/10000
#=======Ansemble============================
X = tf.placeholder(tf.float32, shape=[None, 3],name="A_x")
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_normal([3, 10]), name='weight')
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

3
#====SAVER/RESTORE======================
saver = tf.train.Saver()
sess=tf.Session()

saver.restore(sess,tf.train.latest_checkpoint("./An_model/"))
tf.reset_default_graph()
# Now, let's access and create placeholders variables and
# create feed-dict to feed new data
A_prediction  = sess.run(hypothesis,feed_dict ={X:DATA})
#===================================

print('\n\n\n\n...')
print('loding')
print('...')
print('--------------------Result------------------------')
A_prediction = A_prediction
print('|shinhan 의 10일뒤 close 값은' )
print('|Prediction value : ',A_prediction[0]*10000,'입니다')

if A_prediction[0]*10000 > close[0] :
   print('|10일 전의 close 보다 상승하였습니다')
else :
   print('|10일 전의 close 보다 하락하였습니다')
print('---------------------------------------------------')

