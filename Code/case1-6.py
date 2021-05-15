# ***************************************************************************************************************************
# Camada de entrada: Características extraídas
# Camada 1: 256 neurônios densos
# Camada de saída: softmax para 5 classes

# Parâmetros: 38,661

# ***************************************************************************************************************************
# Import libraries, modules, py files, etc

import csv
import numpy as np
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

allClasses = ['Z', 'O', 'N', 'F', 'S']
classesA = ['F']
classesB = ['S']
classesC = []

case = "case4"

classes = []
classes.extend(classesA)
classes.extend(classesB)

if classesC:
    classes.extend(classesC)
    num_classes = 3
else:
    num_classes = 2

# ***************************************************************************************************************************
# Data Manipulator
path_data = "data.csv"
dataManipulator = DataManipulator(path_data)
titles, data = dataManipulator.read_data(h_titles=True, type=float)
ndata = np.asarray(data, dtype=np.float32)

# Prepara os dados
delete = []
for i, c in enumerate(allClasses):
    # Seleciona itens que serão excluídos
    if not c in classes:
        for j, data in enumerate(ndata):
            if data[0] == i:
                delete.append(j)
    # Altera o rótulo das que serão usadas
    else:
        if c in classesA:
            for data in ndata:
                if data[0] == i:
                    data[0] = 0
        elif c in classesB:
            for data in ndata:
                if data[0] == i:
                    data[0] = 1
        elif c in classesC:
            for data in ndata:
                if data[0] == i:
                    data[0] = 2
#Exclui as classes que não serão usadas
if delete:
    ndata = np.delete(ndata, delete, 0)


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

while(True):

    # Define per-fold score containers
    acc_per_fold = []
    precision_per_fold = []
    sensitivity_per_fold = []
    specificity_per_fold = []
    f1_per_fold = []
    g_meam_per_fold = []

    # Define the K-fold Cross Validator
    #kfold = GroupKFold(n_splits=10)
    kfold = KFold(n_splits=FOLDS, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        # ******************************************************************************************************************
        # Criando a rede
        model = tf.keras.models.Sequential()

        # Camada de entrada

        model.add(tf.keras.Input(shape=(inputs.shape[1],), name='Entrada'))

        # Camadas ocultas
        ## Camada 1
        model.add(tf.keras.layers.Dense(units=512,
                                        name='camadaOculta1',
                                        activation='elu',
                                        kernel_initializer="he_uniform",
                                        kernel_regularizer=tf.keras.regularizers.l1(l=0.01),
                                        bias_regularizer=tf.keras.regularizers.l1(0.01)))

        # Camada de Saída
        model.add(tf.keras.layers.Dense(units=num_classes,
                                        name='camadaSaida',
                                        activation='softmax',
                                        kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.)))

        opt = tf.keras.optimizers.RMSprop(learning_rate=0.11)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        model.summary()

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Callback para salvar os pesos com maior accuracy
        checkpoint = tf.keras.callbacks.ModelCheckpoint(case + ".hdf5", monitor='sparse_categorical_accuracy', verbose=1,
                                                        save_best_only=True, mode='max', period=1)
        callbacks_list = [checkpoint]

        # Fit data to model
        history = model.fit(inputs[train], targets[train],
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            callbacks=callbacks_list,
                            verbose=1)

        # Carrega os pesos da rede com maior accuracy
        model.load_weights(case + ".hdf5")


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
            f'Score for fold {fold_no}: Accuracy of {metrics.get_accuracy()*100}%')

        acc_per_fold.append(metrics.get_accuracy())
        precision_per_fold.append(metrics.get_precision_macro())
        sensitivity_per_fold.append(metrics.get_sensitivity_macro())
        specificity_per_fold.append(metrics.get_specificity_macro())
        f1_per_fold.append(metrics.get_f1_macro())
        g_meam_per_fold.append(metrics.get_g_mean_macro())

        # Increase fold number
        fold_no = fold_no + 1

    # == Provide average scores ==
    print('------------------------------------------------------------------------------------------------------------------------------------------------')
    print('Scores per fold')

    titles = ["Metric"]
    for i in range(FOLDS):
        titles.append("K" + str(i+1))
    titles.append("Mean")
    titles.append("Std")

    acc_l = ["Acc"]
    pre_l = ["Pre"]
    sen_l = ["Sen"]
    spe_l = ["Spe"]
    f1_l = ["F1"]
    gm_l = ["GM"]

    table = []
    # Accuracy
    for c in acc_per_fold:
        acc_l.append(round(c, 4) * 100)
    acc_l.append(round(np.mean(acc_per_fold), 4)*100)
    acc_l.append(round(np.std(acc_per_fold), 4)*100)

    # Precision
    for c in precision_per_fold:
        pre_l.append(round(c, 4) * 100)
    pre_l.append(round(np.mean(precision_per_fold), 4)*100)
    pre_l.append(round(np.std(precision_per_fold), 4)*100)

    # Sensitivity
    for c in sensitivity_per_fold:
        sen_l.append(round(c, 4) * 100)
    sen_l.append(round(np.mean(sensitivity_per_fold), 4)*100)
    sen_l.append(round(np.std(sensitivity_per_fold), 4)*100)

    # Specificity
    for c in specificity_per_fold:
        spe_l.append(round(c, 4) * 100)
    spe_l.append(round(np.mean(specificity_per_fold), 4)*100)
    spe_l.append(round(np.std(specificity_per_fold), 4)*100)

    # F1
    for c in f1_per_fold:
        f1_l.append(round(c, 4) * 100)
    f1_l.append(round(np.mean(f1_per_fold), 4)*100)
    f1_l.append(round(np.std(f1_per_fold), 4)*100)

    # G-mean
    for c in g_meam_per_fold:
        gm_l.append(round(c, 4) * 100)
    gm_l.append(round(np.mean(g_meam_per_fold), 4)*100)
    gm_l.append(round(np.std(g_meam_per_fold), 4)*100)

    table.append(acc_l)
    table.append(pre_l)
    table.append(sen_l)
    table.append(spe_l)
    table.append(f1_l)
    table.append(gm_l)

    #print(tabulate(table, titles, tablefmt="latex"))

    print(tabulate(table, titles))

    # Save the results
    csvPath = "mainResults\\" + case + " - " + str(acc_l[len(acc_l) - 2]) + '.csv'
    if os.path.exists(csvPath):
        os.remove(csvPath)
    with open(csvPath, mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=';', lineterminator='\n')
        employee_writer.writerow(titles)
        employee_writer.writerows(table)