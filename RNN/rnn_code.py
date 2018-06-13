"""
2015722082 정지현

주가지표 사용(Bollinger Band, rsi, sma, wma) -> input 6개 
10일치 데이터 넣고 10일 뒤 예측

RMSE와 MAPE로 정확도 측정
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import talib as ta
import csv

tf.reset_default_graph() 

 # train Parameters
R_seq_length = 10         #10일 데이치를 넣어 
R_after = 9               #그로부터 10일뒤 종가 예측
R_data_dim = 6           #지표로 변환하여 6개가 들어감
R_hidden_dim = 10         #lstm에서 output은 10개가 나옴
R_output_dim = 1          #FC를 통해 10개에서 1개로
R_learning_rate = 0.05   #학습률
R_iterations = 5000       #반복 횟수
R_train_index = 32        #몇번째 부터 넣을지(nan을 제외한 부분을 위해 )
R_predict_len = 5          #talib parameter

#------------------------------------------------------------------------------
#데이터크기를 균일하게 보정
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

#split, null 제거작업
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
        
    return date,output #print(date)
 
def read_file(filename):
    f = open(filename, 'r').read()
    data = f.split('\n')[:-2]
    raw_data=[]
     
    info=data[0].split(",")
    date,raw_data=del_null_line(data)
     
    return info, raw_data
#------------------------------------------------------------------------------
#TRAIN
def R_train(R_sess, R_optimizer, R_trainX, R_trainY):           
    for i in range(R_iterations):
        R_sess.run(R_optimizer, feed_dict={R_X: R_trainX, R_Y: R_trainY})
        
#TEST    
def R_test(R_sess, R_Y_pred, R_testX, R_testY, wr):
    RNN_prediction = R_sess.run(R_Y_pred, feed_dict={R_X: R_testX})
    
    #그래프 출력            
    plt.plot(RNN_prediction, label='pred')
    plt.plot(R_testY, label='real')
    plt.legend()
    plt.show()        

    #예측 결과(선형) 파일에 저장
    for i in range(len(RNN_prediction)) : 
        wr.writerow(RNN_prediction[i]*1000000) 
           
    RMSE(R_testY, RNN_prediction) #RMSE 측정 
    MAPE(R_testY, RNN_prediction) #MAFE 측정
#------------------------------------------------------------------------------
#RMSE
def RMSE(R_testY, RNN_prediction):
    rmse_val = R_sess.run(rmse, feed_dict={targets: R_testY, predictions: RNN_prediction}) 
    print("RMSE:%.6f" % (rmse_val)) 

#MAPE
def MAPE(R_testY, RNN_prediction):    
    ERROR_ans=0
    ERROR=np.zeros(len(RNN_prediction))

    for i in range(len(RNN_prediction)):
        ERROR[i]=np.abs((RNN_prediction[i]-R_testY[i]))/np.abs(R_testY[i])
        ERROR_ans+=ERROR[i]
    
    print("MAPE:%.6f" % (ERROR_ans*100/len(RNN_prediction)))          
#------------------------------------------------------------------------------
#파일을 불러와 x,y로 나눔
#a, raw_data = read_file('../data/amore.csv')
#a, raw_data = read_file('./data/lg.csv')
#a, raw_data = read_file('./data/hyundai.csv')
#a, raw_data = read_file('./data/samsung_fire.csv')
#a, raw_data = read_file('./data/lotte_chemical.csv')
#a, raw_data = read_file('./data/lg_display.csv')
a, raw_data = read_file('../data/shinhan.csv')

xy = np.array(raw_data)
x = np.c_[xy[:, 0:-1]] #numpy로 데이터 합치기 
y = xy[:, [-1]]/1000
#------------------------------------------------------------------------------
#x,y 행렬 전치 해줌 ( 그래야 talib가 먹힘 )
t_open = np.transpose(x[:,[0]])
t_high =  np.transpose(x[:,[1]])
t_low = np.transpose(x[:,[2]])
t_vol = np.transpose(x[:,[3]])
t_y = np.transpose(y)

#array로 변환
open1 = np.array(t_open[0],dtype='float')
high = np.array(t_high[0],dtype='float')
low = np.array(t_low[0],dtype='float')
volume = np.array(t_vol[0],dtype='float')
close = np.array(t_y[0],dtype='float')
#------------------------------------------------------------------------------
#주가데이터를 지표데이터로 변환(TA-Lib 이용)
b_upper, b_middle, b_lower = ta.BBANDS(close) #bollinger ban
rsi = ta.RSI(close, timeperiod=R_predict_len*2) #Relative Strength Index
sma = ta.SMA(close, timeperiod=5) #Simple Moving average
wma = ta.WMA(close, timeperiod=5) #Weigth Moving average
#------------------------------------------------------------------------------
#x를 지표데이터 6개 묶음으로 변경
x = np.c_[b_upper, b_middle, b_lower, rsi, sma, wma]/1000 
y=y/1000
       
#X,Y
R_x = x[R_train_index: -1]            #nan이 아닌 부분부터
R_y = y[R_train_index: -1]          #x시작에 맞게 nan이 아닌 부분부터
#------------------------------------------------------------------------------
#build a dataset
#dataX에는 10일치 데이터, dataY는 그로부터 10일 뒤 종가 데이터 저장
R_dataX = []
R_dataY = []
for R_i in range(0, len(R_y) - R_seq_length - R_after):
    _x = R_x[R_i:R_i + R_seq_length]
    _y = R_y[R_i + R_seq_length + R_after]
    R_dataX.append(_x)
    R_dataY.append(_y)   
#------------------------------------------------------------------------------
# 전체 data 범위를 step1, step2, step3로 나눔
set_size1 = int(len(R_dataY) * 0.3)
set_size2 = set_size1 + int(len(R_dataY) * 0.3)
set_size3 = set_size2 + int(len(R_dataY) * 0.4)

#train/test를 8:2 비율로 나눔
train_size1 = int(set_size1 * 0.8)
train_size2 = int(set_size1 + (set_size2-set_size1)*0.8)
train_size3 = int(set_size2 + (set_size3-set_size2)*0.8)

R_trainX1, R_testX1 = np.array(R_dataX[0:train_size1]), np.array(R_dataX[train_size1:set_size1])
R_trainY1, R_testY1 = np.array(R_dataY[0:train_size1]), np.array(R_dataY[train_size1:set_size1])

R_trainX2, R_testX2 = np.array(R_dataX[set_size1:train_size2]), np.array(R_dataX[train_size2:set_size2])
R_trainY2, R_testY2 = np.array(R_dataY[set_size1:train_size2]), np.array(R_dataY[train_size2:set_size2])

R_trainX3, R_testX3 = np.array(R_dataX[set_size2:train_size3]), np.array(R_dataX[train_size3:])
R_trainY3, R_testY3 = np.array(R_dataY[set_size2:train_size3]), np.array(R_dataY[train_size3:])
#------------------------------------------------------------------------------
# input place holders
R_X = tf.placeholder(tf.float32, [None, R_seq_length, R_data_dim] , name="R_X")
R_Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
R_cell = tf.contrib.rnn.BasicLSTMCell(num_units=R_hidden_dim, state_is_tuple=True, activation=tf.tanh) #lstm 셀 생성
R_outputs, R_states = tf.nn.dynamic_rnn(R_cell, R_X, dtype=tf.float32) #cell output node
#마지막 lstm output(10개)을 FC를 통해 한개로 출력
R_Y_pred = tf.contrib.layers.fully_connected(R_outputs[:, -1], R_output_dim, activation_fn=None) 

# cost, optimizer
R_cost = tf.reduce_sum(tf.square(R_Y_pred - R_Y))
R_optimizer = tf.train.AdamOptimizer(R_learning_rate, name="R_optimizer").minimize(R_cost)
#------------------------------------------------------------------------------
# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
#------------------------------------------------------------------------------
#session
R_init = tf.global_variables_initializer()
R_sess = tf.Session()
R_sess.run(R_init)

###############################################################################
fp = open('./RNN_shinhan.csv','w',encoding='utf-8',newline='')
wr = csv.writer(fp)

print('----------STEP1----------');
R_train(R_sess, R_optimizer, R_trainX1, R_trainY1) #TRATIN1
R_test(R_sess, R_Y_pred, R_testX1, R_testY1, wr) #TEST1
print('----------STEP2----------');
R_train(R_sess, R_optimizer, R_trainX2, R_trainY2) #TRATIN2
R_test(R_sess, R_Y_pred, R_testX2, R_testY2, wr) #TEST2
print('----------STEP3----------');
R_train(R_sess, R_optimizer, R_trainX3, R_trainY3) #TRATIN3
R_test(R_sess, R_Y_pred, R_testX3, R_testY3, wr) #TEST3

'''
saver = tf.train.Saver()
save_path = saver.save(R_sess, '../boost/model_saver_test/RNN_model/RNN')   
'''
fp.close()

