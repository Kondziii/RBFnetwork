import math
import random
import matplotlib.pyplot as plot
import numpy as np
from sklearn.cluster import KMeans


class RBF(object):
    def __init__(self, learningRate=0.05, momentum=0.05, epochs=20, inputNodes=4, hiddenNodes=20, outputNodes=1,
                 bias=1, classes=3):
        self.learningRate = learningRate
        self.momentum = momentum
        self.epochs = epochs
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.bias = bias
        self.radials = np.zeros((self.inputNodes, self.hiddenNodes))
        self.r = np.zeros(self.hiddenNodes)
        self.outFromHiddenLayer = None
        self.radialOutput = np.zeros(self.hiddenNodes + self.bias)
        self.radialOutput[:] = 1
        self.outputWeights = np.zeros(self.hiddenNodes + self.bias)
        for i in range(self.hiddenNodes + self.bias):
            self.outputWeights[i] = random.uniform(-1 / 2, 1 / 2)
        self.previousWeights = np.array(self.outputWeights)
        self.epoch = []
        self.error = []
        self.testingError = []
        self.trainOutput = []
        self.testOutput = []
        self.mistakeCounter = 0
        self.resultTable = np.zeros((classes, classes))
        self.classes = classes

    def calculateGaussian(self, distance, r):
        return math.exp(-math.pow(distance, 2) / 2 * math.pow(r, 2))

    def trainRadialLayer(self, data):
        # wybor centrow za pomoca algorytmu k-means
        np.random.shuffle(data)
        self.radials = KMeans(init='k-means++', n_clusters=self.hiddenNodes).fit(
            data[:, :self.inputNodes]).cluster_centers_

        # ustawienie promieni dla neuronow radialnych
        for i in range(self.hiddenNodes):
            distances = np.zeros(self.hiddenNodes)
            for j in range(self.hiddenNodes):
                distances[j] = np.linalg.norm(self.radials[i] - self.radials[j])
            self.r[i] = max(distances) / math.sqrt(2 * self.hiddenNodes)

        # obliczenie outputu z warstwy ukrytej
        self.outFromHiddenLayer = np.zeros((data.shape[0], self.hiddenNodes))
        for d, x in enumerate(data[:, :self.inputNodes]):
            for i in range(self.hiddenNodes):
                self.outFromHiddenLayer[d][i] = self.calculateGaussian(np.linalg.norm(x - self.radials[i]), self.r[i])

    def trainLinearLayer(self, data, dataToTest):
        for e in range(self.epochs):
            epochError = 0
            for d in range(data.shape[0]):
                input = data[d][:self.inputNodes]
                target = data[d][self.inputNodes]

                self.radialOutput[:self.hiddenNodes] = self.outFromHiddenLayer[d]

                # propagacja w przod
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

                if e == self.epochs - 1:
                    self.trainOutput.append([input, output, target])

                epochError += deltaOutput * deltaOutput / 2
            self.epoch.append(e)
            self.error.append(epochError / data.shape[0])
            self.testingError.append(self.classify(dataToTest))

    def query(self, input):
        for i in range(self.hiddenNodes):
            self.radialOutput[i] = self.calculateGaussian(np.linalg.norm(input - self.radials[i]), self.r[i])
        output = 0.0
        for i in range(self.hiddenNodes + self.bias):
            output += self.outputWeights[i] * self.radialOutput[i]
        return output

    def classify(self, dataTest, final=False):
        error = 0
        np.random.shuffle(dataTest)
        for d in range(dataTest.shape[0]):
            input = dataTest[d][:self.inputNodes]
            target = dataTest[d][self.inputNodes]
            output = self.query(input)
            error += ((output - target) ** 2) / 2
            if final:
                self.testOutput.append([input, output, target])
        error /= dataTest.shape[0]
        return error

    def train(self, data, dataToTest, repeats=10):
        mis = data.shape[0]
        for r in range(repeats):
            self.error.clear()
            self.epoch.clear()
            self.testingError.clear()
            self.testOutput.clear()
            self.trainOutput.clear()
            self.trainRadialLayer(data)
            self.trainLinearLayer(data, dataToTest)
            self.classify(dataToTest, True)
            if self.calMistakes() < mis:
                mis = self.calMistakes()
                testo = self.testOutput.copy()
                traino = self.trainOutput.copy()
        self.trainOutput = traino
        self.testOutput = testo
        self.displayResults()
        plot.plot(self.epoch, self.error)
        plot.title('Chart for ' + self.hiddenNodes.__str__() + ' neurons - errors from training input')
        plot.xlabel('Epochs')
        plot.ylabel('Error')
        plot.ylim(0, max(self.error) + 0.05)
        # plot.savefig(self.momentum.__str__()+'m2etrain'+self.hiddenNodes.__str__()+'.png')
        # plot.show()

        # wykres bledu dla zbioru testowego
        plot.plot(self.epoch, self.testingError)
        plot.title('Chart for ' + self.hiddenNodes.__str__() + ' neurons - errors from testing input')
        plot.xlabel('Epochs')
        plot.ylabel('Error')
        plot.ylim(0, max(self.testingError) + 0.05)
        # plot.savefig('2etest'+self.hiddenNodes.__str__()+'.png')
        # plot.show()

    def calMistakes(self):
        counter = 0
        for i in self.testOutput:
            if round(i[1], 0) != i[2]:
                counter += 1
        return counter

    def createResultTable(self):
        for i in self.testOutput:
            for j in range(self.classes):
                if i[2] == float(j+1):
                    self.resultTable[j][int(round(i[1], 0)) - 1] += 1

    def displayResults(self):
        print('--------------------------TRAINING POINTS--------------------------')
        for i in self.trainOutput:
            if round(i[1], 0) != i[2]:
                print('Input: ', i[0], 'Outcome: ', round(i[1], 6), '[', round(i[1], 0), ']', 'Target: ', i[2], 'Difference')
            else:
                print('Input: ', i[0], 'Outcome: ', round(i[1], 6), '[', round(i[1], 0), ']', 'Target: ', i[2])

        print('--------------------------TESTING POINTS--------------------------')
        for i in self.testOutput:
            if round(i[1], 0) != i[2]:
                print('Input: ', i[0], 'Outcome: ', round(i[1], 6), '[', round(i[1], 0), ']', 'Target: ', i[2], 'Difference')
            else:
                print('Input: ', i[0], 'Outcome: ', round(i[1], 6), '[', round(i[1], 0), ']', 'Target: ', i[2])
        print('--------------------------RESULTS--------------------------')
        print('Badly classified testing vectors: ', self.calMistakes())
        self.createResultTable()
        print('       1.0 2.0 3.0')
        for i, vec in enumerate(self.resultTable):
            print((i+1).__str__()+'.0: ', vec)

with open('classification_train.txt') as f:
    inputs = []
    for line in f:
        inputs.append([float(x) for x in line.split()])
inputsTrain = np.array(inputs)

with open('classification_test.txt') as f:
    inputs = []
    for line in f:
        inputs.append([float(x) for x in line.split()])
inputsTest = np.array(inputs)

rbf = RBF()
rbf.train(inputsTrain, inputsTest)

# inputsTrain2 = np.delete(inputsTrain, (0,1), 1)
# inputsTest2 = np.delete(inputsTest, (0,1), 1)
# rbf = RBF()
# rbf.train(inputsTrain2, inputsTest2)
