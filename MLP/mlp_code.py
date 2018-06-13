"""
hongji
3/19
:선형 데이터 -> csv
:각모델마다의 정확성 -> 각자 제일 잘 나오는 방식으로
 (MLP : 영향 정도로 0.08 방법 적용)
 
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import talib as ta 
import csv

#전역변수 선언
time_len = 10            #몇일뒤 예측 날짜 인지 
predict_len = 5          #talib parameter
M_learning_rate = 0.01   #MLP model learning rate
train_index = 31         #nan 제거
input_n = 12             #모델의 input data 갯수

##----------Read CSV file -------------
def del_null_line(data):
    output=[]
    date=[]
    for i in data:
        i=i.split(",")
        try:
            date.append(i[0])
            output.append(list(map(float,i[1:])))
        except ValueError as e:
            print('')   
    return date,output

def read_file(filename):
    f = open(filename, 'r')
    data = f.read().split('\n')[:-2]
    raw_data=[]    
    f.close()
    
    info=data[0].split(",")
    date,raw_data=del_null_line(data)
    
    return info, raw_data

a, raw_data = read_file('../data/shinhan.csv')
#hyundai_heavy hyundai lg_elec amore lg_chemical kia_car sk_telecom shinhan naver sk_hynix

xy = np.array(raw_data)
##--------- Accuracy function --------------
def acc(MLP_prediction,testY,testX):
    
    #1. MAPE
    ERROR_ans=0
    ERROR=np.zeros(len(MLP_prediction))
    for i in range(len(MLP_prediction)):
        bi = MLP_prediction[i]-testY[i]
        ERROR[i]=np.abs((bi))/np.abs(testY[i])
        ERROR_ans+=ERROR[i]
    print(">>MAPE Accuracy %.2f" % (ERROR_ans*100/len(MLP_prediction))) 
    
    #print prediction graph, real graph
    plt.plot(MLP_prediction,label='pred')
    plt.plot(testY,label='real')  
    plt.legend()
    plt.show()
    
##--------- Train function -------------
def train(M_sess, M_optimizer, trainX, trainY):
    for step in range(10000):
        M_sess.run(M_optimizer, feed_dict={M_X: trainX, M_Y: trainY})
    
    #2.MSE
    print(">>MSE Accuracy", M_sess.run([M_cost], feed_dict={M_X: trainX, M_Y: trainY}))
    
##================ main function ======================
x = np.c_[xy[:, 0:-1]] #x data : stock data( open, high, low, volume )
y = xy[:, [-1]]/1000   #y data : close 

#---------------for Talib -----------------
t_high =  np.transpose(x[:,[1]])
t_low = np.transpose(x[:,[2]])
t_vol = np.transpose(x[:,[3]])
t_y = np.transpose(y)

high = np.array(t_high[0],dtype='float')
low = np.array(t_low[0],dtype='float')
volume = np.array(t_vol[0],dtype='float')
close = np.array(t_y[0],dtype='float')
#-----------------------------------------

#-------------TA-Lib function 이용-----------------
b_upper, b_middle, b_lower = ta.BBANDS(close)                            #bollinger ban
ma = ta.MA(close,timeperiod=predict_len)                                 #moving average
dmi = ta.DX(high,low,close,timeperiod=predict_len)                       #direct movement index
macd, macdsignal, macdhist = ta.MACD(close,fastperiod=predict_len, 
        slowperiod=predict_len*2,signalperiod=4)                         #moving average convergence/divergence
slowk, slowd = ta.STOCH(high, low, close, fastk_period=predict_len, 
        slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)  #Stochastic
mom = ta.MOM(close,timeperiod=predict_len)
rsi = ta.RSI(close,timeperiod=predict_len*2)                             #Relative Strength Index
cci = ta.CCI(high,low,close,timeperiod=predict_len*2)
sma = ta.SMA(close,timeperiod=predict_len)                               #Simple Moving average
wma = ta.WMA(close,timeperiod=predict_len)                               #Weigth Moving average

x = np.c_[close,ma,b_upper, b_middle, b_lower,macd, macdsignal, macdhist ,dmi,cci,sma,wma]/100
y=y/100
#x = talib_ data /100 ( data 범위 조정 )
#y = close data / 100 ( data 범위 조정 )

x = x[train_index+9:len(x)-time_len-1]  
y = y[train_index+time_len+9:len(y)-1]  #time_index 만큼 뒤의 data와 매치해 학습

#ex) 30,60,100
set_size1 = int(len(y) * 0.3)               #30%
set_size2 = set_size1 + int(len(y) * 0.3)   #30%
set_size3 = set_size2 + int(len(y) * 0.4)   #40%
#set_size4 = set_size3 + int(len(y) * 0.3)   #30%

#ex)
train_size1 = int(set_size1 * 0.8)                       #train rate = 30%중 80%
train_size2 = int(set_size1 + (set_size2-set_size1)*0.8) #train rate = 30%중 80%
train_size3 = int(set_size2 + (set_size3-set_size2)*0.8) #train rate = 40%중 80%
#train_size4 = int(set_size3 + (set_size4-set_size3)*0.8) #train rate = 30%중 80%


#1. train / test set
trainX1, testX1 = np.array(x[0:train_size1]), np.array(x[train_size1:set_size1])
trainY1, testY1 = np.array(y[0:train_size1]), np.array(y[train_size1:set_size1])

#2. train / test set
trainX2, testX2 = np.array(x[set_size1:train_size2]), np.array(x[train_size2:set_size2])
trainY2, testY2 = np.array(y[set_size1:train_size2]), np.array(y[train_size2:set_size2])

#3. train / test set
trainX3, testX3 = np.array(x[set_size2:train_size3]), np.array(x[train_size3:set_size3])
trainY3, testY3 = np.array(y[set_size2:train_size3]), np.array(y[train_size3:set_size3])

#4. train / test set
#trainX4, testX4 = np.array(x[set_size3:train_size4]), np.array(x[train_size4:])
#trainY4, testY4 = np.array(y[set_size3:train_size4]), np.array(y[train_size4:])

#========================MLP 모델/학습시작======================================
###########
# 신경망 모델 구성
#input_n - 15 - 10 - 6 - 1(0,1 모델)
###########

M_X = tf.placeholder(tf.float32, [None, input_n], name="M_X")#10개의 값이 10일치 가 1-set
M_Y = tf.placeholder(tf.float32, [None, 1])

#총 3개의 hidden layer 와 입력 출력 layer 가 있다
#layer 1 : 입력 레이어 열의 개수가 input_n개로 들어가 15개의 전달 값을 갖는다.
M_W1 = tf.Variable(tf.random_uniform([input_n, 15],-1.,1.))
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

M_cost = tf.reduce_mean(tf.square((M_Y - M_model)))
M_optimizer = tf.train.AdamOptimizer(M_learning_rate, name="M_optimizer").minimize(M_cost)

#########
# 신경망 모델 학습
#########

M_init = tf.global_variables_initializer()
M_sess = tf.Session()
M_sess.run(M_init)

#total y graph
plt.plot(y)
plt.show()

print("1 :",set_size1,"2: ",set_size2-set_size1, "3: ",set_size3-set_size2)
print("total data: ",set_size3)

#------------test1 --------------------------
train(M_sess, M_optimizer,trainX1, trainY1)

MLP_prediction =M_sess.run(M_model, feed_dict={M_X: testX1}) #Prediction(Hypothesis)
target = M_sess.run(M_Y, feed_dict={M_Y: testY1})            #실제 data (y)
acc(MLP_prediction,testY1,testX1)


#file open - csv 로 data 저장 
fa = open('./MLP_shinhan.csv','w',encoding='utf-8',newline='')
ft = open('./testy_shinhan.csv','w',encoding='utf-8',newline='')
wr = csv.writer(fa)
wt = csv.writer(ft)

for i in range(len(MLP_prediction)) :
    wr.writerow(MLP_prediction[i]*100000)
for i in range(len(testY1)) :
    wt.writerow(testY1[i]*100000)
    
#------------test2---------------------------
train(M_sess, M_optimizer,trainX2, trainY2)

MLP_prediction =M_sess.run(M_model, feed_dict={M_X: testX2}) #Prediction(Hypothesis)
target = M_sess.run(M_Y, feed_dict={M_Y: testY2})            #실제 data (y)
acc(MLP_prediction,testY2,testX2)


for i in range(len(MLP_prediction)) :
    wr.writerow(MLP_prediction[i]*100000)
for i in range(len(testY2)) :
    wt.writerow(testY2[i]*100000)
    
#------------test3---------------------------
train(M_sess, M_optimizer,trainX3, trainY3)

MLP_prediction =M_sess.run(M_model, feed_dict={M_X: testX3}) #Prediction(Hypothesis)
target = M_sess.run(M_Y, feed_dict={M_Y: testY3})            #실제 data (y)
acc(MLP_prediction,testY3,testX3)


for i in range(len(MLP_prediction)) :
    wr.writerow(MLP_prediction[i]*100000)
for i in range(len(testY3)) :
    wt.writerow(testY3[i]*100000)
'''
saver = tf.train.Saver()
save_path = saver.save(M_sess, '../boost/model_saver_test/MLP_model/MLP',global_step=10000)
'''
#파일 close 
fa.close()
ft.close()
