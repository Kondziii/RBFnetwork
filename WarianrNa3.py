import math
import random
import matplotlib.pyplot as plot
import numpy as np

class RBF(object):
    def __init__(self, learningRate=0.2, momentum=0.01, epochs=100, inputNodes=1, hiddenNodes=12, outputNodes=1, bias=1):
        self.bias = bias
        self.learningRate = learningRate
        self.momentum = momentum
        self.epochs = epochs
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.radials = np.zeros(self.hiddenNodes)
        self.r = np.zeros(self.hiddenNodes)
        self.radialOutput = np.zeros(self.hiddenNodes + self.bias)
        self.radialOutput[:] = 1
        self.outputWeights = np.zeros(self.hiddenNodes + self.bias)
        for i in range(self.hiddenNodes + self.bias):
            self.outputWeights[i] = random.uniform(-10, 10)
        self.previousWeights = np.array(self.outputWeights)
        self.x = []
        self.y = []
        self.epoch = []
        self.error = []
        self.testingError = []
        self.testError = 0
        self.outFromHiddenLayer = None

    def calculateGaussian(self, distance, r):
        return math.exp(-math.pow(distance, 2) / 2 * math.pow(r, 2))

    def query(self, input):
        self.radialOutput[:] = 1
        for i in range(self.hiddenNodes):
            self.radialOutput[i] = self.calculateGaussian(np.abs(input - self.radials[i]), self.r[i])

        # propagacja w przod warstwy outputu
        output = 0.0
        for i in range(self.hiddenNodes + self.bias):
            output += self.outputWeights[i] * self.radialOutput[i]
        return output

    def trainRadialLayer(self, data):
        # wybor centrow ze zbioru treningowego
        self.radials = data[np.random.choice(data.shape[0], self.hiddenNodes, replace=False)][:, 0]
        # ustawienie promieni dla neuronow radialnych
        for i in range(self.hiddenNodes):
            distances = np.zeros(self.hiddenNodes)
            for j in range(self.hiddenNodes):
                distances[j] = np.abs(self.radials[i] - self.radials[j])
            self.r[i] = max(distances) / math.sqrt(2 * self.hiddenNodes)
        # obliczenie outputu z warstwy ukrytej korzystajac z funkcji gaussa
        self.outFromHiddenLayer = np.zeros((data.shape[0], self.hiddenNodes))
        for d in range(data.shape[0]):
            for i in range(self.hiddenNodes):
                self.outFromHiddenLayer[d][i] = self.calculateGaussian(np.abs(data[d][0] - self.radials[i]), self.r[i])

    def trainLinearLayer(self, data, testData):
        for e in range(self.epochs):
            epochError = 0
            for d in range(data.shape[0]):
                target = data[d][1]
                self.radialOutput[:self.hiddenNodes] = self.outFromHiddenLayer[d]
                # propagacja w przod warstwy outputu
                output = 0.0
                for i in range(self.hiddenNodes + self.bias):
                    output += self.outputWeights[i] * self.radialOutput[i]

                deltaOutput = output - target
                # aktualizacja wag
                for i in range(self.hiddenNodes + self.bias):
                    self.outputWeights[i] = self.outputWeights[i] - (
                            deltaOutput * self.radialOutput[i] * self.learningRate) + \
                                            self.momentum * (self.outputWeights[i] - self.previousWeights[i])
                    self.previousWeights[i] = self.outputWeights[i]

                # obliczenie bledu
                epochError += deltaOutput * deltaOutput / 2
            self.epoch.append(e)
            self.error.append(epochError / data.shape[0])
            self.testingError.append(self.checkAproximateQuality(testData))

    def train(self, data, testData, repeats=5):
        error = float('inf')
        for i in range(repeats):
            self.x.clear()
            self.y.clear()
            self.epoch.clear()
            self.error.clear()
            self.testingError.clear()
            self.trainRadialLayer(data)
            self.trainLinearLayer(data, testData)
            if np.average(self.error) < error:
                error = np.average(self.error)
                err = self.error.copy()
                wei = self.outputWeights.copy()
                rad = self.radials.copy()
                sig = self.r.copy()
                tes = self.testingError.copy()
        self.error = err
        self.outputWeights = wei
        self.radials = rad
        self.r = sig
        self.testError = self.checkAproximateQuality(testData)
        self.testingError = tes
        # wykres aproksymacji z punktami treningowymi
        plot.plot(data[:, 0], data[:, 1], 'ro', label='Training Points')
        plot.title('Chart for '+self.hiddenNodes.__str__()+' radial neurons \n eta= '+self.learningRate.__str__()+ ' ,momentum= '+
                                        self.momentum.__str__())
        plot.xlabel('x')
        plot.ylabel('y')
        for i in range(-4050, 4050):
            self.x.append(i / 1000)
            self.y.append(float(self.query(i / 1000)))
        plot.plot(self.x, self.y, linewidth=7, label='Approximated function')
        plot.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
        plot.xticks(np.arange(-5, 6, 1))
        plot.ylim(-8, 4)
        #plot.savefig(self.momentum.__str__()+'m2train' + self.hiddenNodes.__str__() + '.png')
        plot.show()


        # wykres bledu dla zbioru treningowego
        plot.plot(self.epoch, self.error)
        plot.title('Chart for '+self.hiddenNodes.__str__()+' neurons - errors from training input')
        plot.xlabel('Epochs')
        plot.ylabel('Error')
        plot.ylim(0, max(self.error)+0.05)
        #plot.savefig(self.momentum.__str__()+'m2etrain'+self.hiddenNodes.__str__()+'.png')
        plot.show()

        # wykres bledu dla zbioru testowego
        plot.plot(self.epoch, self.testingError)
        plot.title('Chart for '+self.hiddenNodes.__str__()+' neurons - errors from testing input')
        plot.xlabel('Epochs')
        plot.ylabel('Error')
        plot.ylim(0, max(self.testingError)+0.05)
        #plot.savefig('2etest'+self.hiddenNodes.__str__()+'.png')
        plot.show()

        # wykres aproksymacji z punktami testowymi
        plot.plot(testData[:, 0], testData[:, 1], 'yo', label='Testing Points')
        plot.title('Chart for ' + self.hiddenNodes.__str__() + ' radial neurons - error='+round(self.testError, 5).__str__()+
                   '\n eta= ' + self.learningRate.__str__() + ' ,momentum= ' + self.momentum.__str__())
        plot.xlabel('x')
        plot.ylabel('y')
        plot.plot(self.x, self.y, linewidth=7, label='Approximated function')
        plot.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
        plot.xticks(np.arange(-5, 6, 1))
        plot.ylim(-8, 4)
        #plot.savefig('2test'+self.hiddenNodes.__str__()+'.png')
        plot.show()

    def checkAproximateQuality(self, testData):
        error = 0
        for d in testData:
            input = d[0]
            target = d[1]
            output = self.query(input)
            error += ((output - target) ** 2) / 2
        error /= testData.shape[0]
        return error

# odczytanie pliku approximation_train1
with open('approximation_train1.txt') as f:
    inputsTrain1 = []
    for line in f:
        inputsTrain1.append([float(x) for x in line.split()])
inputsTrain1 = np.array(inputsTrain1)
# odczytanie pliku approximation_train2
with open('approximation_train2.txt') as f:
    inputsTrain2 = []
    for line in f:
        inputsTrain2.append([float(x) for x in line.split()])
inputsTrain2 = np.array(inputsTrain2)
# odczytanie pliku approximation_test
with open('approximation_test.txt') as f:
    inputsTest = []
    for line in f:
        inputsTest.append([float(x) for x in line.split()])
inputsTest = np.array(inputsTest)

rbf = RBF()
rbf.train(inputsTrain1, inputsTest)

# rbf1 = RBF()
# rbf1.train(inputsTrain2, inputsTest)