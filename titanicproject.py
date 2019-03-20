import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cluster import KMeans
from sklearn import metrics
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def preprocess():

    # load data
    data = pd.read_csv('train.csv', header=0)

    labels = data['Survived'].as_matrix()

    data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

    # pclass attribute to vector so that we can compare its content
    pclass = data['Pclass'].as_matrix()
    pclass_vector = np_utils.to_categorical(pclass, num_classes=4)[:, 1:]

    # sex attribute to vector so that we can compare its content
    att_sex = {'male': 0, 'female': 1}
    sex = [att_sex[sx] for sx in data['Sex'].tolist()]
    sex_vector = np_utils.to_categorical(sex , num_classes=2)

    # adds mean_age to null cells
    mean_age = int(data['Age'].mean())
    age = data['Age'].as_matrix()
    age[np.isnan(age)] = mean_age

    # scales the numeric data
    numeric = data[['SibSp', 'Parch', 'Fare']].as_matrix()
    numeric = preprocessing.scale(numeric)

    input = np.concatenate((numeric, age.reshape((len(age), 1)), sex_vector , pclass_vector), axis=1)

    # returns an array with 6 attributes as input & an array of the labels as output
    return input, labels




def RF_algorithm(train_data, val_data, train_label, val_label):

    # Random Forest is being trained for different values of number of trees and criterion

    for criterion in ['gini', 'entropy']:
        for n_trees in [5, 10, 15, 20, 25, 30]:

            rf = RandomForestClassifier(criterion= criterion, n_estimators=n_trees, random_state=10)
            rf.fit(train_data, train_label)

            accuracy = rf.score(val_data, val_label)
            prediction = rf.predict(val_data)
            metrics = precision_recall_fscore_support(val_label, prediction, average='binary')

            print('criterion:',criterion, 'n_estimators:', n_trees)
            print('Accuracy:', accuracy)
            print('Precision:', metrics[0])
            print('Recall:', metrics[1])
            print('F1-score:', metrics[2])


    return

def KNN_algorithm(train_data, val_data, train_label, val_label):

    # KNN classifier is being trained for different values of 'k' and weights

    for neighbors in [1, 2, 3, 4, 5, 6, 7, 8]:
        for weights in ['uniform', 'distance']:
            knn = KNeighborsRegressor(n_neighbors=neighbors, weights=weights, metric='manhattan').fit(train_data, train_label)

            accuracy = knn.score(val_data, val_label)
            prediction = knn.predict(val_data)
            prediction [prediction >= 0.5] = 1
            prediction [prediction < 0.5] = 0
            metrics = precision_recall_fscore_support(val_label, prediction, average='binary')

            print('n_neighbors:', neighbors, 'weights:', weights)
            print('Accuracy:', accuracy)
            print('Precision:', metrics[0])
            print('Recall:', metrics[1])
            print('F1-score:', metrics[2])

    return


def SVM_algorithm(train_data, val_data, train_label, val_label):

    # SVM is being traind for different values of kernel function and c parameter


    for kernel in ['linear', 'rbf', 'poly']:
        for c in [0.1, 1, 10]:

            svm_model = SVC(kernel=kernel, C=c)
            svm_model.fit(train_data, train_label)
            accuracy = svm_model.score(val_data, val_label)
            prediction = svm_model.predict(val_data)
            prediction[prediction >= 0.5] = 1
            prediction[prediction < 0.5] = 0
            metrics = precision_recall_fscore_support(val_label, prediction, average='binary')

            print('Kernel:', kernel, 'C param:', c)
            print('Accuracy:', accuracy)
            print('Precision:', metrics[0])
            print('Recall:', metrics[1])
            print('F1-score:', metrics[2])



    return


def neural_model(input, learn_rate=0.0001):
    #create a neural model with 4 layers, learning rate=0.0001 and uses adam algorithm as optimizer

    input_layer = Input(shape=(input,))

    layer1 = Dense(50, activation='elu',kernel_initializer='glorot_uniform')(input_layer)
    dropout1 = Dropout(0.0)(layer1)

    layer2 = Dense(30, activation='elu', kernel_initializer='glorot_uniform')(dropout1)
    dropout2 = Dropout(0.0)(layer2)

    layer3 = Dense(10, activation='elu',kernel_initializer='glorot_uniform')(dropout2)
    dropout3 = Dropout(0.0)(layer3)

    output_layer = Dense(1, activation='sigmoid')(dropout3)

    neural_model = Model(inputs=input_layer, outputs=output_layer)

    adam_optimizer = Adam(lr=learn_rate, decay=1e-5)

    neural_model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

    neural_model.summary()
    return neural_model


def neural_network(train_data, val_data, train_label, val_label):

    st = 'titanic_mlp'
    input = train_data.shape[1]
    nm = neural_model(input)

    nnm_json = nm.to_json()

    with open("model/" + st + ".json", "w") as json_file:
        json_file.write(nnm_json)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        path = "path" + st + '.h5'

        model_checkpoint = ModelCheckpoint(path, monitor='val_acc',verbose=1, save_best_only=True, save_weights_only=True)

        history = nm.fit(train_data, train_label, validation_data=(val_data, val_label), epochs=1000, batch_size=64, shuffle=True, callbacks=[early_stopping, model_checkpoint])

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Neural Network Model')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    return




def k_means(input, labels):

    # k-means clustering algorithm for different values of k

    for k in [2, 3, 4, 5, 6, 7]:

        k_means = KMeans(n_clusters=k, random_state=10)

        k_means.fit(input)

        # clustering metrics
        homogeneity = metrics.homogeneity_score(labels, k_means.labels_)
        completeness = metrics.completeness_score(labels, k_means.labels_),
        silhouette_score = metrics.silhouette_score(input, k_means.labels_, metric='euclidean', sample_size=200)
        inertia = k_means.inertia_

        print('Number of clusters:', k)
        print('Homogeneity:', homogeneity)
        print('Completeness:', completeness[0])
        print('Silhouette score:', silhouette_score)
        print('Inertia:', inertia)

    return


if __name__ == '__main__':

    input, labels = preprocess()
    train_data, val_data, train_label, val_label = train_test_split(input, labels, test_size=0.2, random_state=10)

#print('Random Forest Algorithm')
#RF_algorithm(train_data, val_data, train_label, val_label)
#print('KNN algorithm')
#KNN_algorithm(train_data, val_data, train_label, val_label)
#print('SVM algorithm')
#SVM_algorithm(train_data, val_data, train_label, val_label)
#print('Neural Network')
#neural_network(train_data, val_data, train_label, val_label)
#print('K-means algorithm')
#k_means(input, labels)
