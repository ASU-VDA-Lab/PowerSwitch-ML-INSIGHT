import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#data = pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw1_2000.csv", on_bad_lines='warn')
data1 = pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw3_130_1p12p_leak5.csv", on_bad_lines='warn')
data2 = pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw3_160_1p12p_leak15m.csv", on_bad_lines='warn')
data3 = pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw3_190_1p12p_leak25.csv", on_bad_lines='warn')

X1 = data1.iloc[:,:10].values
Y1 = data1.iloc[:,10].values
X2 = data2.iloc[:,:10].values
Y2 = data2.iloc[:,10].values
X3 = data3.iloc[:,:10].values
Y3 = data3.iloc[:,10].values

scale1 = 100/X1.shape[0]
X_train,X_test1,Y_train,Y_test1 = train_test_split(X1,Y1,test_size=scale1,random_state=2)
scale2 = 100/X2.shape[0]
X_train,X_test2,Y_train,Y_test2 = train_test_split(X2,Y2,test_size=scale2,random_state=1)
scale3 = 100/X3.shape[0]
X_train,X_test3,Y_train,Y_test3 = train_test_split(X3,Y3,test_size=scale3,random_state=2)

sc = StandardScaler()

#Scales for model_weights_spice3stage_comb_4leak_lesscap_l14.h5
Mean = [5.69560637e+01,5.69409706e+01,5.69693950e+01,3.00000000e+00,1.70866429e+02,1.20030900e-11,4.89514629e-02,4.89386368e-02,4.89627919e-02,1.52482150e-02]
Scale = [4.25376425e+01,4.25097842e+01,4.25426741e+01,1.00000000e+00,3.59250915e+01,6.85413193e-12,3.61484886e-02,3.61248146e-02,3.61527644e-02,1.08478307e-02]

sc.mean_ = np.array(Mean)
sc.scale_ = np.array(Scale)
X_test11 = sc.transform(X_test1)
X_test22 = sc.transform(X_test2)
X_test33 = sc.transform(X_test3)

model = Sequential()
model.add(Dense(64, input_dim=10, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.load_weights('/home/vgopal18/python/ML_data/model_weights_spice3stage_comb_4leak_lesscap_l14.h5')

Y_predict1 = model.predict(X_test11)
Y_predict2 = model.predict(X_test22)
Y_predict3 = model.predict(X_test33)
f = plt.figure(figsize=(9, 6))

#plt.scatter(Y_test1,Y_predict1, label = 'Mempool')
plt.scatter(Y_test1,Y_predict1, label = '130 Switches', s=90)
plt.scatter(Y_test2,Y_predict2, label = '160 Switches', s=90)
plt.scatter(Y_test3,Y_predict3, label = '190 Switches', s=90)
#plt.scatter(Y_test5,Y_predict5, label = 'Ariane')
#plt.scatter(Y_test6,Y_predict6, label = 'AES')

t = []
#t.append(np.min(np.minimum(Y_test1, np.minimum(Y_test2, np.minimum(Y_test3, np.minimum(Y_test4, np.minimum(Y_test5, Y_test6)))))))
#t.append(np.max(np.maximum(Y_test1, np.maximum(Y_test2, np.maximum(Y_test3, np.maximum(Y_test4, np.minimum(Y_test5, Y_test6)))))))
t.append(np.min(np.minimum(Y_test1, np.minimum(Y_test2, Y_test3))))
t.append(np.max(np.maximum(Y_test1, np.maximum(Y_test2, Y_test3)+0.0005)))

plt.plot(t, t, color='red', linestyle='--', label='x=y', linewidth=4)

plt.xlabel('Actual $I_{rush}$ (mA)', fontsize=34)
plt.ylabel('Predicted $I_{rush}$ (mA)', fontsize=34)
#plt.title('Max Inrush I Comparison', fontsize=16)
scale_x = 1e-3
scale_y = 1e-3

ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
plt.gca().xaxis.set_major_formatter(ticks_x)
ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
plt.gca().yaxis.set_major_formatter(ticks_y)
plt.xticks(fontsize = 40)
plt.yticks(fontsize = 40)
plt.legend(['130 Switches','160 Switches','190 Switches'],fontsize = 36, markerscale=4)
#plt.legend(['Mempool','Amber','jpeg','Ravensha','Ariane','AES'],fontsize = 24)
plt.show()
