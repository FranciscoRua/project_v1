# Francisco Rua
# Sistema de Visualização do espaço topológico de uma rede neuronal convolucional

import ctypes
import os
from PIL import Image
import tensorflow as tf
import numpy as np
from matplotlib import pyplot
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from models import basemodel, basemodel_adam, deepmodel, batchmodel, model_Drop_and_DataAug, bestmodel

'''from importlib import reload
reload(keras.models)'''

epochs = 15
learning_rate = 0.001


def load_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # redefinir canal de forma a ter um canal único
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def prep_dados(train, test):
    # converter inteiros para floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalizar para um intervalo entre 0 e 1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm


def app_run(trainX, trainY, testX, testY, epochs):
    print(
        "\n\n\nBem Vindos à aplicação que permite a Visualização do Espaço Topológico de uma Rede Neuronal Convolucional")
    print("--> Para tal, utilizar-se-ão modelos que serão treinados para classificar digitos escritos manualmente")
    print("\nDataset utilizado - MNIST Dataset")

    print(
        "\nFoi implementado um modelo base para esta aplicação. É nesse modelo que se farão alterações conforme as escolhas feitas pelo utilizador")
    print("Learning Rate - 0.001")
    print("Epochs - 15")
    print("Loss function - categorical_crossentropy")
    print("SGD como optimizador")
    menuchoice = int(input('''
    *************MENU*************
    Escolha uma das opções abaixo:

    1 - Utilizar Modelo Predefinido
    2 - Alterar número de epochs
    3 - Alterar valor de Learning Rate (SGD optimizer)
    4 - Alterar optimizer para Adam
    5 - Aprofundar modelo com a introdução de mais camadas
    6 - Utilizar técnica de Batch Normalization
    7 - Utilizar técnicas de Data Augmentation e Dropout
    8 - Alterar modelo predefino aplicando os 4, 5, 6 e 7 
        (Learning Rate e Número de Epochs predefinido)

    0 - Sair

        Introduzir aqui:  '''))

    if menuchoice == 0:
        exit()
    if menuchoice == 1:
        model = basemodel(learning_rate)
        model.summary()
        modelname = 'models/basemodel.h5'
        graficname = 'grafics/grafic_basemodel.png'
        tsnename = 'tsne/tsne_basemodel.png'

        model_work(modelname, graficname, tsnename, model, trainX, trainY, testX, testY, epochs)
        del menuchoice, model, modelname, graficname, tsnename
        app_run(trainX, trainY, testX, testY, epochs)

    if menuchoice == 2:
        eps = int(input('''
        1 - Diminuir para 5 epochs
        2 - Aumentar para 45 epochs
        
        0 - Voltar
        
        Introduzir aqui: '''))
        if eps == 0:
            del menuchoice, eps
            app_run(trainX, trainY, testX, testY, epochs)
        if eps == 1:
            model = basemodel(learning_rate)
            model.summary()
            modelname = 'models/basemodel_menosEpochs.h5'
            graficname = 'grafics/grafic_basemodel_menosEpochs.png'
            tsnename = 'tsne/tsne_basemodel_menosEpochs.png'

            model_work(modelname, graficname, tsnename, model, trainX, trainY, testX, testY, 5)
            del menuchoice, model, modelname, graficname, tsnename, eps
            app_run(trainX, trainY, testX, testY, epochs)
        if eps == 2:
            model = basemodel(learning_rate)
            model.summary()
            modelname = 'models/basemodel_maisEpochs.h5'
            graficname = 'grafics/grafic_basemodel_maisEpochs.png'
            tsnename = 'tsne/tsne_basemodel_maisEpochs.png'

            model_work(modelname, graficname, tsnename, model, trainX, trainY, testX, testY, 45)
            del menuchoice, model, modelname, graficname, tsnename, eps
            app_run(trainX, trainY, testX, testY, epochs)

    if menuchoice == 3:
        lr = int(input('''
                1 - Diminuir para 0.00001
                2 - Aumentar para 0.1

                0 - Voltar

                Introduzir aqui: '''))
        if lr == 0:
            del menuchoice, lr
            app_run(trainX, trainY, testX, testY, epochs)
        if lr == 1:
            model = basemodel(0.00001)
            model.summary()
            modelname = 'models/basemodel_menosLR.h5'
            graficname = 'grafics/grafic_basemodel_menosLR.png'
            tsnename = 'tsne/tsne_basemodel_menosLR.png'

            model_work(modelname, graficname, tsnename, model, trainX, trainY, testX, testY, epochs)
            del menuchoice, model, modelname, graficname, tsnename, lr
            app_run(trainX, trainY, testX, testY, epochs)
        if lr == 2:
            model = basemodel(0.1)
            model.summary()
            modelname = 'models/basemodel_maisLR.h5'
            graficname = 'grafics/grafic_basemodel_maisLR.png'
            tsnename = 'tsne/tsne_basemodel_maisLR.png'

            model_work(modelname, graficname, tsnename, model, trainX, trainY, testX, testY, epochs)
            del menuchoice, model, modelname, graficname, tsnename, lr
            app_run(trainX, trainY, testX, testY, epochs)

    if menuchoice == 4:
        adam_lr = int(input('''
                1 - Learning_Rate = 0.00001
                2 - Learning_Rate = 0.001
                3 - Learning_Rate = 0.1
                
                0 - Voltar

                Introduzir aqui: '''))
        if adam_lr == 0:
            del menuchoice, adam_lr
            app_run(trainX, trainY, testX, testY, epochs)
        if adam_lr == 1:
            model = basemodel_adam(0.00001)
            model.summary()
            modelname = 'models/basemodel_adam_menosLR.h5'
            graficname = 'grafics/grafic_basemodel_adam_menosLR.png'
            tsnename = 'tsne/tsne_basemodel_adam_menosLR.png'

            model_work(modelname, graficname, tsnename, model, trainX, trainY, testX, testY, epochs)
            del menuchoice, model, modelname, graficname, tsnename, adam_lr
            app_run(trainX, trainY, testX, testY, epochs)
        if adam_lr == 2:
            model = basemodel_adam(0.001)
            model.summary()
            modelname = 'models/basemodel_adam.h5'
            graficname = 'grafics/grafic_basemodel_adam.png'
            tsnename = 'tsne/tsne_basemodel_adam.png'

            model_work(modelname, graficname, tsnename, model, trainX, trainY, testX, testY, epochs)
            del menuchoice, model, modelname, graficname, tsnename, adam_lr
            app_run(trainX, trainY, testX, testY, epochs)
        if adam_lr == 3:
            model = basemodel_adam(0.1)
            model.summary()
            modelname = 'models/basemodel_adam_maisLR.h5'
            graficname = 'grafics/grafic_basemodel_adam_maisLR.png'
            tsnename = 'tsne/tsne_basemodel_adam_maisLR.png'

            model_work(modelname, graficname, tsnename, model, trainX, trainY, testX, testY, epochs)
            del menuchoice, model, modelname, graficname, tsnename, adam_lr
            app_run(trainX, trainY, testX, testY, epochs)

    if menuchoice == 5:
        model = deepmodel()
        model.summary()
        modelname = 'models/deepmodel.h5'
        graficname = 'grafics/grafic_deepmodel.png'
        tsnename = 'tsne/tsne_deepmodel.png'

        model_work(modelname, graficname, tsnename, model, trainX, trainY, testX, testY, epochs)
        del menuchoice, model, modelname, graficname, tsnename
        app_run(trainX, trainY, testX, testY, epochs)

    if menuchoice == 6:
        model = batchmodel()
        model.summary()
        modelname = 'models/batchmodel.h5'
        graficname = 'grafics/grafic_batchmodel.png'
        tsnename = 'tsne/tsne_batchmodel.png'

        model_work(modelname, graficname, tsnename, model, trainX, trainY, testX, testY, epochs)
        del menuchoice, model, modelname, graficname, tsnename
        app_run(trainX, trainY, testX, testY, epochs)

    if menuchoice == 7:
        model = model_Drop_and_DataAug()
        model.summary()
        modelname = 'models/model_Drop_and_DataAug.h5'
        graficname = 'grafics/grafic_model_Drop_and_DataAug.png'
        tsnename = 'tsne/tsne_model_Drop_and_DataAug.png'

        model_work(modelname, graficname, tsnename, model, trainX, trainY, testX, testY, epochs)
        del menuchoice, model, modelname, graficname, tsnename
        app_run(trainX, trainY, testX, testY, epochs)

    if menuchoice == 8:
        model = bestmodel()
        model.summary()
        modelname = 'models/bestmodel.h5'
        graficname = 'grafics/grafic_bestmodel.png'
        tsnename = 'tsne/tsne_bestmodel.png'

        model_work(modelname, graficname, tsnename, model, trainX, trainY, testX, testY, epochs)
        del menuchoice, model, modelname, graficname, tsnename
        app_run(trainX, trainY, testX, testY, epochs)
    else:
        del menuchoice
        print("Caracter inválido! Introduzir opção novamente.\n\n")
        app_run(trainX, trainY, testX, testY, epochs)


def model_work(modelname, graficname, tsnename, model, trainX, trainY, testX, testY, epochs):
    model = model_fit(modelname, graficname, model, trainX, trainY, testX, testY, epochs)

    loss, score = model.evaluate(testX, testY, verbose=0)
    ctypes.windll.user32.MessageBoxW(0,
                                     "Este modelo tem uma precisão de {:.2f}% com erro de {:.2f}%"
                                     .format((score * 100.0), (loss * 100)),
                                     "Precisão do modelo",
                                     0)

    imgshow = Image.open(graficname)
    imgplot = pyplot.imshow(imgshow)
    pyplot.axis('off')
    pyplot.tight_layout()
    pyplot.show()

    tsne_func(model, testX, testY, tsnename)
    digit = digit_choice()
    while digit != '':
        img = load_image(digit)
        pred = model.predict(img)
        ctypes.windll.user32.MessageBoxW(0,
                                         "Nesta imagem está representado um {} com {:.2f}% de confiança."
                                         .format(np.argmax(pred), 100 * np.max(pred)),
                                         "Previsão do modelo",
                                         0)
        del img, pred
        digit = digit_choice()


def model_fit(modelname, graficname, model, trainX, trainY, testX, testY, epochs):
    if not os.path.exists('./{}'.format(modelname)):

        history = model.fit(trainX, trainY, epochs=epochs, batch_size=32, validation_data=(testX, testY),
                            validation_split=0.4)

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)
        pyplot.figure(figsize=(8, 8))

        pyplot.subplot(1, 2, 1)
        pyplot.plot(epochs_range, acc, label='Training Accuracy')
        pyplot.plot(epochs_range, val_acc, label='Validation Accuracy')
        pyplot.legend(loc='lower right')
        pyplot.title('Training and Validation Accuracy')

        pyplot.subplot(1, 2, 2)
        pyplot.plot(epochs_range, loss, label='Training Loss')
        pyplot.plot(epochs_range, val_loss, label='Validation Loss')
        pyplot.legend(loc='upper right')
        pyplot.title('Training and Validation Loss')
        # pyplot.show()
        pyplot.savefig(graficname)
        pyplot.close()
        model.save_weights(modelname)
    else:
        model.load_weights('./{}'.format(modelname))

    return model


def tsne_func(model, testX, testY, tsnename):
    if not os.path.exists('./{}'.format(tsnename)):
        w = model.predict(testX)
        tsne = TSNE(n_components=2, verbose=1, init='pca')
        transformed_w = tsne.fit_transform(w)
        color_map = np.argmax(testY, axis=1)
        plot_tsne(transformed_w, color_map, tsnename)

    imgshow = Image.open(tsnename)
    imgplot = pyplot.imshow(imgshow)
    pyplot.axis('off')
    pyplot.tight_layout()
    pyplot.show()


def plot_tsne(t, cmap, tsnename):
    pyplot.figure(figsize=(10, 10))
    for cl in range(10):
        indices = np.where(cmap == cl)
        indices = indices[0]
        pyplot.scatter(t[indices, 0], t[indices, 1], label=cl)
    pyplot.legend()
    # pyplot.show()
    pyplot.savefig(tsnename)
    pyplot.close()


def digit_choice():
    choice = int(input('''Introduza um digito que pretenda usar para classificação (de 0 a 9).
    De seguida é carregada uma imagem do respetivo digito, escrito manualmente através da ferramenta 'Paint'.
    Com a respetiva imagem, o sistema vai tentar prever de que digito se trata, através do modelo previamente treinado.

    Caso queira voltar ao menu principal introduza '-1'


    Introduzir aqui:  '''))

    if choice == 0:
        digito = 'digits_test/zero.png'
        return digito
    if choice == 1:
        digito = 'digits_test/um.png'
        return digito
    if choice == 2:
        digito = 'digits_test/dois.png'
        return digito
    if choice == 3:
        digito = 'digits_test/tres.png'
        return digito
    if choice == 4:
        digito = 'digits_test/quatro.png'
        return digito
    if choice == 5:
        digito = 'digits_test/cinco.png'
        return digito
    if choice == 6:
        digito = 'digits_test/seis.png'
        return digito
    if choice == 7:
        digito = 'digits_test/sete.png'
        return digito
    if choice == 8:
        digito = 'digits_test/oito.png'
        return digito
    if choice == 9:
        digito = 'digits_test/nove.png'
        return digito
    if choice == -1:
        digito = ''
        return digito
    else:
        del choice
        print("Caracter inválido! Introduzir opção novamente.\n\n")
        digit_choice()


def load_image(filename):
    img = tf.keras.preprocessing.image.load_img(filename, color_mode='grayscale', target_size=(28, 28))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = 255 - img
    img = img / 255.0
    imgshow = Image.open(filename)
    pyplot.imshow(imgshow)
    pyplot.axis('off')
    pyplot.show()
    return img


def run():
    trainX, trainY, testX, testY = load_dataset()

    trainX, testX = prep_dados(trainX, testX)

    app_run(trainX, trainY, testX, testY, epochs)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()
