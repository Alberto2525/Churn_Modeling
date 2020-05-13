#Importing the libraries
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Importing the dataset
dataset = pd.read_csv('Churn_Modeling.csv')
X = dataset.iloc[:,3:13]
y = dataset.iloc[:,13].values

#Enconding the dataset
labelEncoder = LabelEncoder()
X['Gender'] = labelEncoder.fit_transform(X['Gender'])
X = pd.get_dummies(X,drop_first = True).values

#Splitting the dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)

#Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Artificial Network
classifier = tf.keras.Sequential()
classifier.add(tf.keras.layers.Dense(input_shape = (11,),units = 6,activation = 'relu'))
classifier.add(tf.keras.layers.Dense(units = 6,activation = 'relu'))
classifier.add(tf.keras.layers.Dense(units = 1,activation = 'sigmoid'))

#Compiling the Neural Network
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

#Fitting the Neural Network, after a few tries the best number of epochs was 20
history = classifier.fit(X_train,y_train,batch_size = 10,epochs = 20,validation_data = (X_test,y_test))

#Plotting the training
plt.plot(history.history['loss'],label = 'Loss')
plt.plot(history.history['val_loss'],label = 'Val_Loss')
plt.legend()
plt.show()

#Plotting the training
plt.plot(history.history['accuracy'],label = 'Accuracy')
plt.plot(history.history['val_accuracy'],label = 'Val_Acurracy')
plt.legend()
plt.show()