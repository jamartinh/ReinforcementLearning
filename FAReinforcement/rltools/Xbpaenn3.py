# Back-Propagation Neural Network for estimated value learning
#
# Written in Python with Numeric python
#
# Jose Antonio Martin H <jamartin AT dia fi upm es>
import pickle
import time

import numpy as np
from numpy.random import uniform, normal


def GetBinaryCode(value, ranges, max_deep):
    a, b = ranges
    deep = 0
    binary_code = []
    error = 0.0
    while deep < max_deep:
        c = (a + b) / 2.0
        if value >= c:
            binary_code += [1.0]
            a = c
        else:
            binary_code += [-1.0]
            b = c
        deep += 1

    return binary_code


def GetNumFromBinaryCode(listbin, ranges):
    a, b = ranges
    number = a
    for bit in listbin:
        c = (a + b) / 2.0
        if bit > 0:
            a = c
        else:
            b = c
    c = (((a + b) / 2.0) + c) / 2.0
    return c


class NN:
    def DecodeBinaryEncoding(self, values, ranges, deep):
        numbers_list = []
        a = 0
        for i in range(len(ranges)):
            numbers_list += [GetNumFromBinaryCode(values[a:a + deep[i]], ranges[i])]
            a = a + deep[i]
        return numbers_list

    def CreateBinaryEncoding(self, values, ranges, deep):
        input_list = []
        for i in range(len(ranges)):
            input_list += GetBinaryCode(values[i], ranges[i], deep[i])

        return input_list

    def RandomSample(self, new_pattern):
        x = uniform(-1.0, 1.0, (self.nsamples, self.ninputs))
        x = self.ScaleValue(x, -1.0, 1.0, self.lbounds_in, self.ubounds_in)
        pat = []

        for i in range(self.nsamples):
            out = self.update(x[i])
            pattern = [x[i].tolist(), out]
            pat.append(pattern)

        pat.extend(new_pattern)

        return pat

    def ScaleValue(self, x, from_a, from_b, to_a, to_b):
        maxv = to_b
        minv = to_a
        maximo = from_b
        minimo = from_a
        return (minv) + (((x - minimo) / (maximo - minimo)) * ((maxv) - (minv)))

    def sigmoid(self, x):
        # return sign(x)
        # return x / (1.0 + abs(x))
        return np.tanh(x)

    # derivative of our sigmoid function
    def dsigmoid(self, y):
        # return sign(y)
        # return 1.0/((1.0+abs(y))*(1.0+abs(y)))
        return 1.0 - y * y

    def __init__(self, input_ranges, nhidden, output_ranges, deep_in, deep_out, nsamples = 20):
        # number of input, hidden, and output nodes

        self.deep_in = deep_in
        self.deep_out = deep_out
        self.input_ranges = input_ranges
        self.output_ranges = output_ranges

        self.ninputs = len(input_ranges)
        self.nsamples = nsamples
        self.out_values = None

        self.lbounds_in = []
        self.ubounds_in = []

        self.lbounds_out = []
        self.ubounds_out = []

        for r in input_ranges:
            self.lbounds_in.append(r[0])
            self.ubounds_in.append(r[1])

        for r in output_ranges:
            self.lbounds_out.append(r[0])
            self.ubounds_out.append(r[1])

        self.lbounds_in = np.array(self.lbounds_in, np.float32)
        self.ubounds_in = np.array(self.ubounds_in, np.float32)

        self.lbounds_out = np.array(self.lbounds_out, np.float32)
        self.ubounds_out = np.array(self.ubounds_out, np.float32)

        self.ni = np.sum(deep_in) + 1  # +1 for bias node
        self.nh = nhidden
        self.no = np.sum(deep_out)

        # activations for nodes
        self.ai = np.ones(self.ni, np.float32)
        self.ah = np.ones(self.nh, np.float32)
        self.ao = np.ones(self.no, np.float32)

        # create weights
        self.wi = np.random.uniform(-2.0001, 2.0001, (self.ni, self.nh))
        self.wo = np.random.uniform(-2.0001, 2.0001, (self.nh, self.no))

        # last change in weights for momentum
        self.ci = np.zeros((self.ni, self.nh), np.float32)
        self.co = np.zeros((self.nh, self.no), np.float32)

        # approx deltas
        self.output_deltas = self.dsigmoid(self.ao) * (0 * self.ao)
        self.hidden_deltas = self.dsigmoid(self.ah) * np.dot(self.wo, self.output_deltas)

    def SaveW(self, filename):
        W = [self.wi, self.wo]
        pickle.dump(W, open(filename, 'w'))

    def LoadW(self, filename):
        W = pickle.load(open(filename, 'r'))
        self.wi = W[0]
        self.wo = W[1]

    def update(self, inputs):

        inputs = self.CreateBinaryEncoding(inputs, self.input_ranges, self.deep_in)

        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.ai[0:self.ni - 1] = np.array(inputs)

        # hidden activations
        self.ah = self.sigmoid(np.dot(self.ai, self.wi))
        self.ao = self.sigmoid(np.dot(self.ah, self.wo))
        self.out_values = self.DecodeBinaryEncoding(self.ao, self.output_ranges, self.deep_out)

        return self.out_values

    def backPropagate(self, targets_real, N, M):

        # targets_real = self.out_values  + 0.3* (array(targets_real) - self.out_values)

        targets_binary = self.CreateBinaryEncoding(targets_real, self.output_ranges, self.deep_out)

        if len(targets_binary) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        od = self.dsigmoid(self.ao) * (targets_binary - self.ao)
        self.output_deltas += (od - self.output_deltas)

        # calculate error terms for hidden
        error = np.dot(self.wo, self.output_deltas)
        hd = self.dsigmoid(self.ah) * error
        self.hidden_deltas += (hd - self.hidden_deltas)

        # update output weights
        change = self.output_deltas * np.reshape(self.ah, (self.ah.shape[0], 1))
        self.wo = self.wo + N * change + M * self.co
        self.co = change

        # update input weights
        change = self.hidden_deltas * np.reshape(self.ai, (self.ai.shape[0], 1))
        self.wi = self.wi + N * change + M * self.ci
        self.ci = change

        # calculate error
        # error = np.sum(0.5 * (targets_binary - self.ao) ** 2)
        error = sum(0.5 * (self.out_values - np.array(targets_real)) ** 2)
        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def singletrain(self, inputs, targets_real, N = 0.5, M = 0.1):
        self.update(inputs)
        error = self.backPropagate(targets_real, N, M)
        return error

    def fast_train(self, patterns, iterations = 100, N = 0.5, M = 0.1):
        # N: learning rate
        # M: momentum factor
        if np.random.random() < 0.00:
            pat = self.RandomSample(patterns)
            self.train(pat, 1, 0.9, 0.1)
        else:
            for i in range(iterations):
                p = patterns[0]
                self.update(p[0])
                self.backPropagate(p[1], 0.25, 0.01)

    def train(self, patterns, iterations = 100, N = 0.1, M = 0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                # print p
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error += self.backPropagate(targets, N, M)
            if i % 100 == 0 and i != 0:
                print('error ' + str(error))


def demoavg():
    # create a network with two input, two hidden, and one output nodes
    rang_input = [[-10, 12], [-10, 12]]
    rang_output = [[-10, 12], [-10, 12]]
    deep_in = [8, 8]
    deep_out = [8, 8]

    a = time.clock()
    n = NN(rang_input, 6, rang_output, deep_in, deep_out)

    targets = n.CreateBinaryEncoding([7, 7], rang_output, deep_out)
    print('prueba targets ', targets)
    # return
    # train it with some patterns
    print("Starting single step training")
    for i in range(50000):
        error = 0.0
        # for 1
        inputs = [-5, -5]
        targets = [normal(-5, 0.01), normal(5, 0.01)]  # noisy input
        error += n.singletrain(inputs, targets, N = 0.5, M = 0.1)

        # for 5
        inputs = [0, 0]
        targets = [normal(0, 0.01), normal(0, 0.01)]  # noisy input
        error += n.singletrain(inputs, targets, N = 0.5, M = 0.1)

        # for 10
        inputs = [10, 10]
        targets = [normal(-10, 0.01), normal(10, 0.01)]  # noisy input
        error += n.singletrain(inputs, targets, N = 0.5, M = 0.1)

        if i % 100 == 0 and i != 0:
            print('error ' + str(error))
            # if error < 1:
            #        break
    # test it



    # n.test(pat)
    # for 1
    print("[-5,5]===>", n.update([-5, 5]))
    print("[0,0]===>", n.update([0, 0]))
    print("[-10,10]===>", n.update([-10, 10]))

    b = time.clock()
    print("Total time for Back Propagation Trainning ", b - a)


if __name__ == '__main__':
    demoavg()
