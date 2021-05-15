# ***************************************************************************************************************************
# Camada de entrada: Características extraídas
# Camada oculta 1: 256 neurônios densos
# Camada oculta 2: 256 neurônios densos
# Camada oculta 3: 256 neurônios densos
# Camada de saída: softmax para 5 classes

# Parâmetros: 170,245

# ***************************************************************************************************************************
# Import libraries, modules, py files, etc

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from bibs import auxiliar as ax
from bibs.dataManipulator import DataManipulator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ***************************************************************************************************************************
# global variables
num_classes = 5
classes = ['Z', 'O', 'F', 'S', 'N']

# ***************************************************************************************************************************
# Data Manipulator
path_data = "data.csv"
dataManipulator = DataManipulator(path_data)
titles, data = dataManipulator.read_data(h_titles=True, type=float)
ndata = np.asarray(data, dtype=np.float32)
X, y = ndata[0:, 1:], ndata[0:, 0]
y = np.asarray(y, dtype=np.int32)
print('Class label counts: ', np.bincount(y))
print('X.shape: ', X.shape)
print('y.shape: ', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print('X_train.shape: ', X_train.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape)
print('y_test.shape: ', y_test.shape)

# ***************************************************************************************************************************
# Data Normalization
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print('X_train mean', np.mean(X_train))
print('X_train standard deviation', np.std(X_train))

# ***************************************************************************************************************************
# Criando a rede
model = tf.keras.models.Sequential()

# Camada de entrada
model.add(tf.keras.Input(shape=(X.shape[1],), name='Entrada'))

# Camadas ocultas
## Camada 1
model.add(tf.keras.layers.Dense(units=256,
                                name='camadaOculta1',
                                activation='selu',
                                kernel_initializer="he_uniform",
                                kernel_regularizer=tf.keras.regularizers.l1(l=0.01),
                                bias_regularizer=tf.keras.regularizers.l1(0.01)))

# Camada de Saída
model.add(tf.keras.layers.Dense(units=num_classes,
                                name='camadaSaida',
                                activation='softmax',
                                kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.)))

opt = tf.keras.optimizers.RMSprop(learning_rate=0.11)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

# Callback para salvar os pesos com maior accuracy
checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.hdf5", monitor='sparse_categorical_accuracy', verbose=1, save_best_only=True, mode='max', period=1)
callbacks_list = [checkpoint]

# Treinando rede
training = model.fit(X_train, y_train, epochs=500, batch_size=5, callbacks=callbacks_list, verbose=1)

print("\n==== PESO FINAL ====")
print("\n==== AVALIANDO ====")
print("Avaliando o treino")
train_loss, train_accuracy = model.evaluate(X_train, y_train)
print("Avaliando o teste")
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\n==== RESULTADOS ====")
print("Train accuracy: {}".format(train_accuracy))
print("Train loss: {}".format(train_loss))
print("Test accuracy: {}".format(test_accuracy))
print("Test loss: {}".format(test_loss))
print("====================")

y_test_pred = model.predict(X_test)
ax.plot_results(training, y_test_pred, y_test, classes)

# Carrega os pesos da rede com maior accuracy
model.load_weights("best_model.hdf5")

print("\n==== MELHOR PESO ====")
print("\n==== AVALIANDO ====")
print("Avaliando o treino")
train_loss, train_accuracy = model.evaluate(X_train, y_train)
print("Avaliando o teste")
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\n==== RESULTADOS ====")
print("Train accuracy: {}".format(train_accuracy))
print("Train loss: {}".format(train_loss))
print("Test accuracy: {}".format(test_accuracy))
print("Test loss: {}".format(test_loss))
print("====================")