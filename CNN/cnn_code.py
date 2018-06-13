# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:22:12 2018

@author: yooop
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import talib as ta 
import csv
#Constrcnt dataset
def del_null_line(data):
    output=[]
    date=[]
    for i in data:
        i=i.split(",")
        try:
            date.append(i[0])
            output.append(list(map(float,i[1:])))
        except ValueError as e:
            print(e)
    #print(date)    
    return date,output

def read_file(filename):
    f = open(filename, 'r').read()
    data = f.split('\n')[:-2]
    raw_data=[]
    
    info=data[0].split(",")
    date,raw_data=del_null_line(data)
    
    return info, raw_data

def train(C_sess,C_optimizer,trainx,trainy):
    for i in range(10):
        for j in range(len(trainx)):       
            C_sess.run(C_optimizer, feed_dict={C_X: trainx[j,:,:], C_Y: trainy[j:j+1]}) #testX(10*15) 한 개, testY(1*1) 한 개씩 들어가도록 함
    print("train끝")
    return

def accuracy(prediction,testy):
    CNN_prediction = np.array(prediction) #예측값, list를 array로 변환
    C_testY = testy.copy()
    CNN_pred = CNN_prediction.copy()
    plt.plot(C_testY,label='Real') 
    plt.plot(prediction,label='Pred')
    plt.legend()
    plt.show()
    vari_rate = 0
    RMSE = C_sess.run(rmse, feed_dict={targets: testy, predictions:prediction}) 
    print(">>RMSE Accuracy: %.6f" %(RMSE))
    ERROR_ans=0
    ERROR=np.zeros(len(CNN_pred))
    for i in range(len(CNN_pred)):
        ERROR[i]=np.abs((CNN_pred[i]-testy[i]))/np.abs(testy[i])
        ERROR_ans+=ERROR[i]
    print(">>MAPE Accuracy: %.6f" % (ERROR_ans*100/len(CNN_pred))) 
    
    for i in range(len(testy)) :
        vari = testy[i:i+1]*vari_rate
        if(testy[i:i+1]+vari <= testy[i+10:i+11]) :#10일 후 종가가 전 종가에 + 제한 범위를 더한 것 보다 클 때마 1로 (증가 클 때)
            C_testY[i:i+1] = 1
        elif(testy[i:i+1]-vari>testy[i+10:i+11]): #10일 후 종가가 전 종가에 - 제한 범위를 뺀 거 보다 작아질 때만  -1로 (감소 클 때)
            C_testY[i:i+1] = -1
        else :                                    #떨어졌을 때
            C_testY[i:i+1] = 0
        
    for i in range(len(CNN_pred)) :
        vari = testy[i:i+1]*vari_rate
        if(testy[i:i+1]+vari <= CNN_pred[i+10:i+11]) :  #전날의 종가에 비해 올랐을 때
            CNN_prediction[i:i+1] = 1
        elif(testy[i:i+1]-vari>CNN_pred[i+10:i+11]):
            CNN_prediction[i:i+1] = -1
        else :                                    #떨어졌을 때
            CNN_prediction[i:i+1] = 0

    #정확도 측정
    is_correct = tf.equal(CNN_prediction, C_testY)
    C_accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('>>CNN Accuracy: %.6f' %C_sess.run(C_accuracy*100, feed_dict={predictions: CNN_prediction, targets: C_testY}))
    return

a, raw_data = read_file('../../code/data/shinhan.csv')
#a, raw_data = read_file('./data/hyundai.csv')
xy = np.array(raw_data)
##############################################
x = np.c_[xy[:, 0:-2], xy[:,[-2]]] #numpy로 데이터 합치기 
y = xy[:, [-1]]
t_y = np.transpose(y)
close = np.array(t_y[0],dtype='float')  
t_open = np.transpose(x[:,[0]])
t_high =  np.transpose(x[:,[1]])#x,y 행렬 전치 해줌 ( 그래야 talib가 먹힘 )
t_low = np.transpose(x[:,[2]])
t_vol = np.transpose(x[:,[3]])
high = np.array(t_high[0],dtype='float')
low = np.array(t_low[0],dtype='float')
volume = np.array(t_vol[0],dtype='float')
#-----------------------------------------

#TA- data ( 5종목의 각각 3,1,1,3,2개의 값을 가짐)
#변수들 선언 
predict_len = 5  #몇일뒤 예측 날짜 인지 
train_index = 32 #몇번째 부터 넣을지(nan을 제외한 부분을 위해 )

#TA-Lib data 전처기 과정--------------------------------
b_upper, b_middle, b_lower = ta.BBANDS(close) #bollinger ban
ma = ta.MA(close,timeperiod=predict_len)         #moving average
dmi = ta.DX(high,low,close,timeperiod=predict_len)#direct movement index
macd, macdsignal, macdhist = ta.MACD(close,fastperiod=predict_len, slowperiod=predict_len*2,signalperiod=4) #moving average convergence/divergence
slowk, slowd = ta.STOCH(high, low, close, fastk_period=predict_len, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0) #Stochastic
mom = ta.MOM(close,timeperiod=predict_len)
rsi = ta.RSI(close,timeperiod=predict_len*2) #상대강도지수(0으로 갈수록 상향이라는)
cci = ta.CCI(high,low,close,timeperiod=predict_len*2)
sma = ta.SMA(close,timeperiod=predict_len)#30으로 나와있긴 하다
wma = ta.WMA(close,timeperiod=predict_len)#30으로 나와있

#x = np.c_[open,high,low, volume, close]
x = np.c_[ma,b_upper, b_middle, b_lower,dmi,cci,sma,wma,mom,close]
y = y/100

#===========================CNN 모델 구성 =====================================
# parameters
C_seq_length = 10      #10일 데이치를 넣어 
C_after = 9     #그로부터 15일뒤 종가 예측
C_iterations = 10        #학습 반복 횟수

C_x = x[train_index: -1]            #nan이 아닌 부분부터
C_y = y[train_index: -1]          #x시작에 맞게 nan이 아닌 부분부터
C_learning_rate = 0.0015

#X는 10일치 데이터set, Y는 그로부터 15일 뒤 종가데이터로 나누어 저장
C_dataX = []
C_dataY = []
for C_i in range(0, len(C_y) - C_seq_length - C_after):
    _x = C_x[C_i:C_i + C_seq_length]
    _y = C_y[C_i + C_seq_length + C_after]  # Next close price
    C_dataX.append(_x)
    C_dataY.append(_y)   
    
# train/test split (talib) 
C_train_size = int(len(C_dataY) * 0.7)
C_trainX, C_testX = np.array(C_dataX[0:C_train_size]), np.array(C_dataX[C_train_size:len(C_dataX)])
C_trainY, C_testY = np.array(C_dataY[0:C_train_size]), np.array(C_dataY[C_train_size:len(C_dataY)])

set_size1 = int(len(C_dataY) * 0.3)#30%까지 data갯수
set_size2 = set_size1 + int(len(C_dataY) * 0.3)#60%까지 data갯수
set_size3 = set_size2 + int(len(C_dataY) * 0.4)#100%까지 data갯수

train_size1 = int(set_size1 * 0.8)#80%(30%중 train크기)
train_size2 = int(set_size1 + (set_size2-set_size1)*0.8)#80%(30%중 train크기)
train_size3 = int(set_size2 + (set_size3-set_size2)*0.8)#80%(40%중 train크기)

C_trainX1, C_testX1 = np.array(C_dataX[0:train_size1]), np.array(C_dataX[train_size1:set_size1])
C_trainY1, C_testY1 = np.array(C_dataY[0:train_size1]), np.array(C_dataY[train_size1:set_size1])

C_trainX2, C_testX2 = np.array(C_dataX[set_size1:train_size2]), np.array(C_dataX[train_size2:set_size2])
C_trainY2, C_testY2 = np.array(C_dataY[set_size1:train_size2]), np.array(C_dataY[train_size2:set_size2])

C_trainX3, C_testX3 = np.array(C_dataX[set_size2:train_size3]), np.array(C_dataX[train_size3:])
C_trainY3, C_testY3 = np.array(C_dataY[set_size2:train_size3]), np.array(C_dataY[train_size3:])

#그래프 초기화 하고 시작 
tf.reset_default_graph()
#앞이 날짜 / 뒤가 지표 갯수
# input place holders
C_X = tf.placeholder(tf.float32, [10, 10], name="C_X")#10*10
C_X_img = tf.reshape(C_X, [-1, 1, 10, 10])#292일 10개 종목 data를 이미지 처럼
C_Y = tf.placeholder(tf.float32, [None, 1])#마지막 결과 Y 하나로
C_K =tf.placeholder(tf.int32)

#layer1ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
C_W1 = tf.Variable(tf.random_normal([2, 2,10, 120], stddev=0.01))#10 -> 100
C_L1 = tf.nn.conv2d(C_X_img, C_W1, strides=[1, 1, 1, 1], padding='SAME')
C_L1 = tf.nn.relu(C_L1)
C_L1 = tf.nn.max_pool(C_L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')#pooling 끝나면 10/5 -> 2로
print(C_L1)

#layer2ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
C_W2 = tf.Variable(tf.random_normal([2, 2, 120, 200], stddev=0.01))#100 -> 200로 만들어 주는 필터
C_L2 = tf.nn.conv2d(C_L1, C_W2, strides=[1, 1, 1, 1], padding='SAME')
C_L2 = tf.nn.relu(C_L2)
C_L2 = tf.nn.max_pool(C_L2, ksize=[1, 2, 2, 1],strides=[1, 5, 5, 1], padding='SAME')#pooling 끝나면 2/2 -> 1로
print(C_L2)
C_L2_flat = tf.reshape(C_L2, [-1,200])#fully connected로 만들어주기
print(C_L2_flat)

# Final FC 200inputs -> 1 outputsㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
C_W3 = tf.get_variable("C_W3", shape=[200,1], initializer=tf.contrib.layers.xavier_initializer())
C_b = tf.Variable(tf.random_normal([1]))
C_logits = tf.add(tf.matmul(C_L2_flat,C_W3), C_b,name="C_model")#1*512 - 512*1 해서 1*1

#cost, optimizer
C_cost = tf.reduce_mean(tf.square((C_Y - C_logits)))
C_optimizer = tf.train.AdamOptimizer(learning_rate=C_learning_rate, name="C_optimizer").minimize(C_cost)
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])

#session-----------------------------------------------------------------------
csvF = open("./CNN_shinhan.csv",'w',encoding ='utf-8',newline='')
writer = csv.writer(csvF)
C_init = tf.global_variables_initializer()
C_sess = tf.Session()
C_sess.run(C_init)
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(predictions,targets)))
#train1 & test1--------------------------------------
prediction1 = []
train(C_sess,C_optimizer,C_trainX1,C_trainY1)

for i in range(len(C_testX1)):
    result = C_sess.run(C_logits, feed_dict={C_X: C_testX1[i,:,:]})
    prediction1.extend(result) #값 이어붙임
accuracy(prediction1,C_testY1)

for i in range(len(prediction1)):
    writer.writerow(prediction1[i]*100)
train(C_sess,C_optimizer,C_testX1,C_testY1)


#train2 & test2--------------------------------------
prediction2 = []
train(C_sess,C_optimizer,C_trainX2,C_trainY2)

for i in range(len(C_testX2)):
    result = C_sess.run(C_logits, feed_dict={C_X: C_testX2[i,:,:]})
    prediction2.extend(result)
accuracy(prediction2,C_testY2)

for i in range(len(prediction2)):
    writer.writerow(prediction2[i]*100)
train(C_sess,C_optimizer,C_testX2,C_testY2)


#train3 & test3--------------------------------------
prediction3 = []
train(C_sess,C_optimizer,C_trainX3,C_trainY3)

for i in range(len(C_testX3)):
    result = C_sess.run(C_logits, feed_dict={C_X: C_testX3[i,:,:]})
    prediction3.extend(result)
accuracy(prediction3,C_testY3)

for i in range(len(prediction3)):
    writer.writerow(prediction3[i]*100)
'''
saver = tf.train.Saver()
save_path = saver.save(C_sess, '../boost/model_saver_test/CNN_model/CNN')
'''
csvF.close()