# ***************************************************************************************************************************
# Camada de entrada: Características extraídas
# Camada ocultta: 128 neurônios densos
# Camada de saída: softmax para 5 classes

# Parâmetros: 19,333

# ***************************************************************************************************************************
# Import libraries, modules, py files, etc

import numpy as np
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from bibs import metrics as mt
from tabulate import tabulate
from bibs.dataManipulator import DataManipulator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ***************************************************************************************************************************
# global variables
BATCH_SIZE = 5
FOLDS = 10
EPOCHS = 500
NEURONS = [64, 128, 256, 512, 1024, 2048, 4096]
ACTIVATION = "selu"

classes = ['Z', 'O', 'F', 'S', 'N']
num_classes = len(classes)

table = []

# ***************************************************************************************************************************
# Data Manipulator
path_data = "../data.csv"
dataManipulator = DataManipulator(path_data)
titles, data = dataManipulator.read_data(h_titles=True, type=float)
ndata = np.asarray(data, dtype=np.float32)
inputs, targets = ndata[0:, 1:], ndata[0:, 0]
targets = np.asarray(targets, dtype=np.int32)
print('Class label counts: ', np.bincount(targets))
print('inputs.shape: ', inputs.shape)
print('targets.shape: ', targets.shape)

# ***************************************************************************************************************************
# Data Normalization
scaler = StandardScaler()
scaler.fit(inputs)
inputs = scaler.transform(inputs)

print('inputs mean', np.mean(inputs))
print('inputs standard deviation', np.std(inputs))

titles = ["Neurons Units"]
for i in range(FOLDS):
    titles.append("K" + str(i + 1))
titles.append("Mean")
titles.append("Std")

# Roda para testar com todas as quantidades de neurônios
for interations in range(len(NEURONS)):
    # Define per-fold score containers
    loss_per_fold = []
    acc_per_fold = []

    # Define the K-fold Cross Validator
    #kfold = GroupKFold(n_splits=10)
    kfold = KFold(n_splits=FOLDS, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        # ******************************************************************************************************************
        # Criando a rede
        name = str(NEURONS[interations]) + "-" + ACTIVATION
        model = tf.keras.models.Sequential(name=name)

        # Camada de entrada

        model.add(tf.keras.Input(shape=(inputs.shape[1],), name='Entrada'))

        # Camadas ocultas
        ## Camada 1
        model.add(tf.keras.layers.Dense(units=NEURONS[interations],
                                        name='camadaOculta',
                                        activation=ACTIVATION,
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

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        file = str(NEURONS[interations]) + "-" + ACTIVATION + ".hdf5"
        # Callback para salvar os pesos com maior accuracy
        checkpoint = tf.keras.callbacks.ModelCheckpoint(file, monitor='sparse_categorical_accuracy', verbose=1,
                                                        save_best_only=True, mode='max', period=1)
        callbacks_list = [checkpoint]

        # Fit data to model
        history = model.fit(inputs[train], targets[train],
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            callbacks=callbacks_list,
                            verbose=1)

        # Carrega os pesos da rede com maior accuracy
        model.load_weights(file)

        # Generate generalization metrics
        scores = model.evaluate(inputs[test], targets[test], verbose=0)
        # Avalia o modelo
        y_test_pred = model.predict(inputs[test], verbose=1)
        y_pred = []
        for i in y_test_pred:
            y_pred.append(i.argmax())
        y_pred = np.asarray(y_pred)
        matrix = confusion_matrix(y_pred, targets[test])
        metrics = mt.CategoricalMetrics(matrix)
        metrics.extract_all_metrics()

        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')

        loss_per_fold.append(scores[0])
        acc_per_fold.append(metrics.get_accuracy())

        # Increase fold number
        fold_no = fold_no + 1

    # == Provide average scores ==
    print(
        '------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Score per fold')
    # == Provide average scores ==
    print('------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Scores per fold')

    acc_l = [NEURONS[interations]]

    # Accuracy
    for c in acc_per_fold:
        acc_l.append(round(c, 4) * 100)
    acc_l.append(round(np.mean(acc_per_fold), 4)*100)
    acc_l.append(round(np.std(acc_per_fold), 4)*100)

    table.append(acc_l)

print(tabulate(table, titles))

# Save the results
csvPath = ACTIVATION + '.csv'
if os.path.exists(csvPath):
            os.remove(csvPath)
with open(csvPath, mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=';', lineterminator='\n')
    employee_writer.writerow(titles)
    employee_writer.writerows(table)