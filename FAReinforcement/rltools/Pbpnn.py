# Back-Propagation Neural Networks
#
# Written in Python with Numeric python
#
# Jose Antonio Martin H <jamartin AT dia fi upm es>
import time
from functools import reduce

import numpy as np
from numpy import *
from numpy.random import uniform


class NN:
    def sigmoid(self, x):
        return np.tanh(x)
        # return 1.0 / (1.0 + exp(-x))

    # derivative of our sigmoid function
    def dsigmoid(self, y):
        return 1.0 - y**2
        # return y * (1-y)

    def __init__(self, input_ranges, nhidden, output_ranges):
        # number of input, hidden, and output nodes and ranges
        self.input_ranges = input_ranges
        self.output_ranges = output_ranges

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

        self.lbounds_in = array(self.lbounds_in, float)
        self.ubounds_in = array(self.ubounds_in, float)

        self.lbounds_out = array(self.lbounds_out, float)
        self.ubounds_out = array(self.ubounds_out, float)

        self.ni = len(input_ranges) + 1  # +1 for bias node
        self.nh = nhidden
        self.no = len(output_ranges)

        # true scale output
        self.ro = ones((self.no), float)

        # activations for nodes
        self.ai = ones(self.ni, float)
        self.ah = ones(self.nh, float)
        self.ao = ones(self.no, float)

        # create weights
        self.wi = uniform(-2.0, 2.0, (self.ni, self.nh))
        self.wo = uniform(-2.0, 2.0, (self.nh, self.no))

        # last change in weights for momentum
        self.ci = zeros((self.ni, self.nh), float)
        self.co = zeros((self.nh, self.no), float)

    def RescaleInputs(self, s):
        ret_s = array(s)
        return self.ScaleValue(ret_s, self.lbounds_in, self.ubounds_in, -1.0, 1.0)

    def RescaleOutputs(self, o):
        return self.ScaleValue(o, -1.0, 1.0, self.lbounds_out, self.ubounds_out)

    def ScaleValue(self, x, from_a, from_b, to_a, to_b):
        maxv = to_b
        minv = to_a
        maximo = from_b
        minimo = from_a
        return (minv) + (((x - minimo) / (maximo - minimo)) * ((maxv) - (minv)))

    def SaveW(self, filename):
        W = [self.wi, self.wo]
        pickle.dump(W, open(filename, 'w'))

    def LoadW(self, filename):
        W = pickle.load(open(filename, 'r'))
        self.wi = W[0]
        self.wo = W[1]

    def update(self, real_inputs):

        # rescales to the input interval [-1,1]
        inputs = self.RescaleInputs(real_inputs)

        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.ai[0:self.ni - 1] = inputs

        ai = self.ai
        wi = self.wi
        wo = self.wo
        f = self.sigmoid
        # hidden activations
        net = dot(self.ai, self.wi)
        self.ah = self.sigmoid(net)

        # output activations
        net = dot(self.ah, self.wo)
        self.ao = self.sigmoid(net)

        self.ao = reduce(lambda i, o: f(dot(i, o)), [ai, wi, wo])
        self.ro = self.RescaleOutputs(self.ao)
        return self.ro

    def backPropagate(self, real_targets, N, M):

        targets = self.ScaleValue(real_targets, self.lbounds_out, self.ubounds_out, -1.0, 1.0)

        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = self.dsigmoid(self.ao) * (targets - self.ao)

        # calculate error terms for hidden
        error = dot(self.wo, output_deltas)
        hidden_deltas = self.dsigmoid(self.ah) * error
        change = hidden_deltas * reshape(self.ai, (self.ai.shape[0], 1))
        self.wi += N * change + M * self.ci
        self.ci = change

        # update output weights
        change = output_deltas * reshape(self.ah, (self.ah.shape[0], 1))
        self.wo += N * change + M * self.co
        self.co = change
        error = np.sum(0.5 * (targets - self.ao) ** 2)

        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def arraytest(self, inputs, targets):
        N = inputs.shape[0]
        for i in range(N):
            print(inputs[i], "->", self.update(inputs[i]))

    def singletrain(self, inputs, targets, N = 0.5, M = 0.1):
        self.update(inputs)
        return self.backPropagate(targets, N, M)

    def fast_train(self, patterns, iterations = 100, N = 0.5, M = 0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            p = patterns[0]
            self.update(p[0])
            self.backPropagate(p[1], N, M)

    def train(self, patterns, iterations = 100, N = 0.5, M = 0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error += self.backPropagate(targets, N, M)
            if i % 100 == 0 and i != 0:
                print('error ' + str(error))

    def array_train(self, inputs, targets, iterations = 100, N = 0.5, M = 0.1):
        # N: learning rate
        # M: momentum factor
        nelems = inputs.shape[0]
        for i in range(iterations):
            error = 0.0
            for j in range(nelems):
                self.update(inputs[j])
                error += self.backPropagate(targets[j], N, M)
            if i % 100 == 0 and i != 0:
                print('error ' + str(error))


def demoArr():
    # create a network with two input, two hidden, and one output nodes

    print("Array Tests")

    rang_input = [[1, 6]]
    rang_output = [[1, 26]]

    inputs = arange(1, 6)
    targets = inputs ** 2

    print(inputs)
    print(targets)

    a = time.clock()
    n = NN(rang_input, 6, rang_output)

    n.array_train(inputs, targets, 1000)
    # test it
    n.arraytest(inputs, targets)

    b = time.clock()
    print("Total time for Back Propagation Trainning ", b - a)


def demo():
    # create a network with two input, two hidden, and one output nodes
    pat = []
    rang_input = [[1, 6]]
    rang_output = [[1, 26]]

    for i in range(1, 6):
        inputs = [float(i)]
        targets = [float(i) ** 2]
        pat.append([inputs, targets])

    print(pat)

    a = time.clock()
    n = NN(rang_input, 6, rang_output)

    n.train(pat, 1000)
    # test it
    n.test(pat)

    b = time.clock()
    print("Total time for Back Propagation Trainning ", b - a)


def demoavg():
    # create a network with two input, two hidden, and one output nodes
    rang_input = [[1, 6]]
    rang_output = [[1, 26]]

    a = time.clock()
    n = NN(rang_input, 3, rang_output)

    # train it with some patterns
    print("Starting single step training")
    for i in range(10000):
        error = 0.0
        # for 1
        for j in range(1, 6):
            inputs = [float(j)]
            targets = [float(j) ** 2]
            error += n.singletrain(inputs, targets, N = 0.1, M = 0.1)

        if i % 100 == 0 and i != 0:
            print('error ' + str(error))

    # n.test(pat)
    # for 1
    for j in range(1, 6):
        input = [float(j)]
        print("[%d]===>" % j, n.update([input]))

    b = time.clock()
    print("Total time for Back Propagation Trainning ", b - a)


if __name__ == '__main__':
    # demoArr()
    # demo()
    demoavg()
