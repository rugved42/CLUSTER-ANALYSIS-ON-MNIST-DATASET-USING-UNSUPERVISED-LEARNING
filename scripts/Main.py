#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import util_mnist_reader
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix


X_train, y_train = util_mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = util_mnist_reader.load_mnist('data/fashion', kind='t10k')

x_train_sc=scale(X_train.astype(np.float64))
x_test_sc=scale(X_test.astype(np.float64))

kmeans = KMeans(n_clusters=10, init='random',max_iter=500,random_state=0).fit(X_train)
a=kmeans.labels_
y_pred=kmeans.predict(X_test)

c_m2=confusion_matrix(y_test,y_pred)
metrics.normalized_mutual_info_score(y_test, y_pred,average_method='geometric')


# In[ ]:


#PART 2: AutoEncoder With K-Means 
from keras.layers import Input, Conv2D, MaxPooling2D,UpSampling2D
from sklearn.cluster import KMeans
from keras.models import Model
from keras import datasets
from matplotlib import pyplot as plt
import numpy as np
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from sklearn import metrics
from sklearn.metrics import confusion_matrix

(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()

X_train=X_train.astype(np.float32)
X_test=X_test.astype(np.float32)
X_train = X_train / np.max(X_train)
X_test = X_test / np.max(X_test)
X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))  

input_img = Input(shape=(28, 28, 1))

x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
ada=optimizers.Adagrad(lr=0.005)
autoencoder.compile(optimizer=ada, loss='binary_crossentropy')
autoencoder_train=autoencoder.fit(X_train, X_train,epochs=5,verbose=1,validation_data=(X_test, X_test))
autoencoder.save_weights('autoencoder1.h5')

def plotGraph(history):
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('AutoEncoder loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
plotGraph(autoencoder_train)

input_img1 = Input(shape=(28, 28, 1))

x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img1)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
encoded1 = MaxPooling2D((2, 2), padding='same')(x)
autoencoder1 = Model(input_img1, encoded1)

for l1, l2 in zip(autoencoder1.layers[0:8], autoencoder.layers[0:8]):
    l1.set_weights(l2.get_weights())
autoencoder1.get_weights()[0][1]

decoded_imgs2 = autoencoder1.predict(X_train)
decoded_imgs2=decoded_imgs2.reshape(-1,512)

decoded_imgs3 = autoencoder1.predict(X_test)
decoded_imgs3=decoded_imgs3.reshape(-1,512)

kmeans = KMeans(n_clusters=10, init='random',max_iter=500,random_state=0).fit(decoded_imgs2)
a=kmeans.labels_
y_pred=kmeans.predict(decoded_imgs3)
c_m2=confusion_matrix(y_test,y_pred)
print(c_m2)
acc1=metrics.normalized_mutual_info_score(y_test, y_pred)
print('Accuracy: ', acc1)


# In[ ]:


#Part_3: Autoencoder with GMM
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras import datasets
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.mixture import GaussianMixture
get_ipython().run_line_magic('matplotlib', 'inline')
(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()

X_train = np.reshape(X_train, (len(X_train),784))
X_test = np.reshape(X_test, (len(X_test), 784))  
X_train=X_train.astype(np.float32)
X_test=X_test.astype(np.float32)
X_train = X_train / np.max(X_train)
X_test = X_test / np.max(X_test)


X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))  

input_img = Input(shape = (28, 28, 1))
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
ada=optimizers.Adagrad(lr=0.005)
autoencoder.compile(optimizer=ada, loss='binary_crossentropy')
autoencoder_train = autoencoder.fit(X_train, X_train,epochs=8,verbose=1,validation_data=(X_test, X_test))
autoencoder.save_weights('autoencoder.h5')

def plotGraph(history):
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('AutoEncoder loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
plotGraph(autoencoder_train)

input_img1 = Input(shape=(28, 28, 1))

x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img1)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
encoded1 = MaxPooling2D((2, 2), padding='same')(x)
autoencoder1 = Model(input_img1, encoded1)

for l1, l2 in zip(autoencoder1.layers[0:8], autoencoder.layers[0:8]):
    l1.set_weights(l2.get_weights())
autoencoder1.get_weights()[0][1]

decoded_imgs2 = autoencoder1.predict(X_train)
decoded_imgs2=decoded_imgs2.reshape(-1,512)

decoded_imgs3 = autoencoder1.predict(X_test)
decoded_imgs3=decoded_imgs3.reshape(-1,512)

gmm = GaussianMixture(n_components=10, max_iter=500 ,random_state=0).fit(decoded_imgs2)
y_pred = gmm.predict(decoded_imgs3)
c_m2 = confusion_matrix(y_test, y_pred)
print(c_m2)
acc2 = metrics.normalized_mutual_info_score(y_test, y_pred ,average_method='geometric')
print('Accuracy: ', acc2)

