{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNNsCswUMbYf5LP6WrIIi2J"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "KpfWdHFBxBFU"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import tensorflow\n",
        "from tensorflow import keras\n",
        "from keras.models import model_from_json\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "base = pd.read_csv('wine.csv')"
      ],
      "metadata": {
        "id": "1HzXhgxaxCqU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "entradas = base[base.columns[1:]].to_numpy()\n",
        "saidas = base[base.columns[0]]\n",
        "\n",
        "valoresSaida = np.empty((178, 1), dtype=int)\n",
        "\n",
        "for i in range(178):\n",
        "   valoresSaida[i]= saidas[i]-1\n",
        "\n",
        "entradasTrain, entradasTest, valoresSaidaTrain, valoresSaidaTest = train_test_split(entradas, valoresSaida, test_size=0.3)\n",
        "\n",
        "print(\"Quantidade de Dados de Treino:\" ,len(entradasTrain))\n",
        "print(\"Quantidade de Dados de Teste:\" ,len(entradasTest))\n",
        "\n",
        "print(\"Quantidade de Dados de Treino e dos atributos:\" ,entradasTrain.shape)\n",
        "print(\"Quantidade de Dados de Teste e dos atributos\" ,entradasTest.shape)\n",
        "\n",
        "print(\"Quantidade de Saidas de Treino e coluna:\" ,valoresSaidaTrain.shape)\n",
        "print(\"Quantidade de Saidas de Teste e coluna:\" ,valoresSaidaTest.shape)\n",
        "\n",
        "print(\"min: \", valoresSaidaTrain.min())\n",
        "print(\"max: \", valoresSaidaTrain.max())"
      ],
      "metadata": {
        "id": "Uwo8da0fxGLX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "modelo = keras.Sequential([keras.layers.Dropout(0.2),\n",
        "                           keras.layers.Dense(130, activation=tensorflow.nn.relu),\n",
        "                           keras.layers.Dense(70, activation=tensorflow.nn.relu),\n",
        "                           keras.layers.Dense(40, activation=tensorflow.nn.relu),\n",
        "                           keras.layers.Dense(13, activation=tensorflow.nn.relu),\n",
        "                           keras.layers.Dense(3, activation=tensorflow.nn.softmax)\n",
        "                           ])\n",
        "\n",
        "modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')\n",
        "\n",
        "hist=modelo.fit(entradasTrain, valoresSaidaTrain, epochs=70, validation_split=0.3)\n",
        "\n",
        "plt.plot(hist.history['accuracy'])\n",
        "plt.plot(hist.history['val_accuracy'])\n",
        "plt.title('Acurácia por épocas')\n",
        "plt.xlabel('Épocas')\n",
        "plt.ylabel('Acurácia')\n",
        "plt.legend(['Treino', 'Validação'])\n",
        "\n",
        "plt.plot(hist.history['loss'])\n",
        "plt.plot(hist.history['val_loss'])\n",
        "plt.title('Loss por épocas')\n",
        "plt.xlabel('Épocas')\n",
        "plt.ylabel('Loss')\n",
        "plt.legend(['Treino', 'Validação'])\n",
        "\n",
        "model_json = modelo.to_json()\n",
        "with open(\"model.json\", \"w\") as json_file:\n",
        "   json_file.write(model_json)\n",
        "\n",
        "modelo.save_weights(\"model.h5\")\n",
        "print(\"Modelo Salvo\")\n",
        "\n",
        "json_file = open(\"model.json\", \"r\")\n",
        "loaded_model_json = json_file.read()\n",
        "json_file.close()\n",
        "\n",
        "loaded_model = model_from_json(loaded_model_json)\n",
        "loaded_model.load_weights(\"model.h5\")\n",
        "print(\"Modelo Carregado\")\n",
        "\n",
        "loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')\n",
        "\n",
        "perda_teste, acuracia_teste = loaded_model.evaluate(entradasTest, valoresSaidaTest)\n",
        "print(\"Perda do teste: \", perda_teste)\n",
        "print(\"Acurácia do teste: \", acuracia_teste)\n",
        "testes=loaded_model.predict(entradasTest)\n",
        "\n",
        "x=0\n",
        "for i in testes:\n",
        "   print(np.argmax(i), valoresSaidaTest[x])\n",
        "   x=x+1"
      ],
      "metadata": {
        "id": "U7h4aPkVxLn0"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}