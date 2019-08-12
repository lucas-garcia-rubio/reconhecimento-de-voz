import os
from scipy.fftpack import fft
from scipy.io import wavfile
from matplotlib.pyplot import plot
import numpy as np
from scipy import signal
#%%
#------------------------------------------------------------------------------
#Criando listas com o nome dos arquivos .wav

arq_yes = os.listdir('D:\Audios\yes')
#arq_no = os.listdir('D:\Audios\yesmarvin')
arq_on = os.listdir('D:\Audios\yesbed')
arq_off = os.listdir('D:\Audios\off')
arq_one = os.listdir('D:\Audios\yesstop')
arq_two = os.listdir('D:\Audios\yestwo')
arq_three = os.listdir('D:\Audios\yeszero')

#%% Função pad - Preencher com 0's

pad1d = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0])))


#%%
#------------------------------------------------------------------------------
#Preenchendo as listas com os valores absolutos da Transformada de Fourier

amostras = 20000
l = 0 #cada linha da matriz X
X = np.zeros(((len(arq_yes) + len(arq_on)
            + len(arq_off) + len(arq_one) + len(arq_two) + len(arq_three)), amostras))
#X = np.zeros((3500, amostras))
#controle = 0 #controlar os primeiros 100 arquivos

for name in arq_yes:
    print('D:\Audios\yes\\' + name)
    rate, data = wavfile.read('D:\Audios\yes\\' + name)
    data = pad1d(data, amostras)
    f, t, Sxx = signal.stft(data, rate, nperseg = 1000, noverlap = 50)
    Sxx = np.abs(Sxx)
    aux = Sxx.flatten()
    aux = np.log(aux+1)
    X[l] = pad1d(np.transpose(aux), amostras)
    l=l+1
    #controle = controle +1
    #if (controle == 500):
    #    break

#controle = 0
"""for name in arq_no:
    print('D:\Audios\yesno\\' + name)
    rate, data = wavfile.read('D:\Audios\yesmarvin\\' + name)
    data = pad1d(data, amostras)
    f, t, Sxx = signal.stft(data, rate, nperseg = 1000, noverlap = 50)
    Sxx = np.abs(Sxx)
    aux = Sxx.flatten()
    aux = np.log(aux+1)
    X[l] = pad1d(np.transpose(aux), amostras)
    l=l+1
    #controle = controle +1
    #if (controle == 500):
    #    break
"""
#controle = 0
for name in arq_on:
    rate, data = wavfile.read('D:\Audios\yesbed\\' + name)
    data = pad1d(data, amostras)
    f, t, Sxx = signal.stft(data, rate, nperseg = 1000, noverlap = 50)
    Sxx = np.abs(Sxx)
    aux = Sxx.flatten()
    aux = np.log(aux+1)
    X[l] = pad1d(np.transpose(aux), amostras)
    l=l+1
    #controle = controle +1
    #if (controle == 500):
    #    break
    
#controle = 0    
for name in arq_off:
    rate, data = wavfile.read('D:\Audios\off\\' + name)
    data = pad1d(data, amostras)
    f, t, Sxx = signal.stft(data, rate, nperseg = 1000, noverlap = 50)
    Sxx = np.abs(Sxx)
    aux = Sxx.flatten()
    aux = np.log(aux+1)
    X[l] = pad1d(np.transpose(aux), amostras)
    l=l+1
    #controle = controle +1
    #if (controle == 500):
    #    break
    
#controle = 0
for name in arq_one:
    rate, data = wavfile.read('D:\Audios\yesstop\\' + name)
    data = pad1d(data, amostras)
    f, t, Sxx = signal.stft(data, rate, nperseg = 1000, noverlap = 50)
    Sxx = np.abs(Sxx)
    aux = Sxx.flatten()
    aux = np.log(aux+1)
    X[l] = pad1d(np.transpose(aux), amostras)
    l=l+1
    #controle = controle +1
    #if (controle == 500):
    #    break

#controle = 0
for name in arq_two:
    rate, data = wavfile.read('D:\Audios\yestwo\\' + name)
    data = pad1d(data, amostras)
    f, t, Sxx = signal.stft(data, rate, nperseg = 1000, noverlap = 50)
    Sxx = np.abs(Sxx)
    aux = Sxx.flatten()
    aux = np.log(aux+1)
    X[l] = pad1d(np.transpose(aux), amostras)
    l=l+1
    #controle = controle +1
    #if (controle == 500):
    #    break
    
#controle = 0
for name in arq_three:
    rate, data = wavfile.read('D:\Audios\yeszero\\' + name)
    data = pad1d(data, amostras)
    f, t, Sxx = signal.stft(data, rate, nperseg = 1000, noverlap = 50)
    Sxx = np.abs(Sxx)
    aux = Sxx.flatten()
    aux = np.log(aux+1)
    X[l] = pad1d(np.transpose(aux), amostras)
    l=l+1
    #controle = controle +1
    #if (controle == 500):
    #    break
    


# MARVIN - 0 YES - 1 BED - 2 OFF - 3 STOP - 4 TWO - 5 ZERO - 6

#%%
#------------------------------------------------------------------------------
# Aprendizado

y = []

for i in arq_yes:
    y.append('1')
    
# =============================================================================
# for i in arq_no:
#     y.append('0')
# =============================================================================
    
for i in arq_on:
    y.append('2')
    
for i in arq_off:
    y.append('3')
    
for i in arq_one:
    y.append('4')
    
for i in arq_two:
    y.append('5')
    
for i in arq_three:
    y.append('6')
    
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

#%%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#%%
from sklearn.neural_network import MLPClassifier as mlp
clf = mlp(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', batch_size=200,
          learning_rate_init=0.00001, max_iter = 400, verbose=1)
clf.fit(X_train, y_train)

#%%
y_train_pred = clf.predict(X_train_std)
y_test_pred = clf.predict(X_test_std)

#%%
from sklearn.metrics import accuracy_score
print(accuracy_score(y_train_pred, y_train))
print(accuracy_score(y_test_pred, y_test))

#%%
#print(ppn.coef_)
#print(ppn.intercept_)


















