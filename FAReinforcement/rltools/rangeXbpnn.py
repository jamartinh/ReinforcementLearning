# Back-Propagation Neural Networks
#
# Written in Python with Numeric python
#
# Jose Antonio Martin H <jamartin AT dia fi upm es>
import pickle
import time

from numpy import *
from numpy.random import *


class NN:
    def sigmoid(self, x):
        return tanh(x)
        # return 1.0 / (1.0 + exp(-x))

    # derivative of our sigmoid function
    def dsigmoid(self, y):
        return 1.0 - y * y
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

        self.lbounds_in = array(self.lbounds_in, float32)
        self.ubounds_in = array(self.ubounds_in, float32)

        self.lbounds_out = array(self.lbounds_out, float32)
        self.ubounds_out = array(self.ubounds_out, float32)

        self.ni = len(input_ranges) + 1  # +1 for bias node
        self.nh = nhidden
        self.no = len(output_ranges)

        # activations for nodes
        self.ai = ones((self.ni), float32)
        self.ah = ones((self.nh), float32)
        self.ao = ones((self.no), float32)

        # true scale output
        self.ro = ones((self.no), float32)

        # create weights
        # self.wi = uniform(-2.,2.,(self.ni, self.nh))
        # self.wo = uniform(-2.,2.,(self.nh, self.no))
        self.wi = random.uniform(-0.00001, 0.00001, (self.ni, self.nh))
        self.wo = random.uniform(-0.00001, 0.00001, (self.nh, self.no))

        # last change in weights for momentum
        self.ci = zeros((self.ni, self.nh), float32)
        self.co = zeros((self.nh, self.no), float32)

        # approx deltas
        self.output_deltas = self.dsigmoid(self.ao) * (0 * self.ao)
        self.hidden_deltas = self.dsigmoid(self.ah) * dot(self.wo, self.output_deltas)

        # estimates learning rate
        self.est_lr = 0.9

    def RescaleInputs(self, s):
        ret_s = array(s)
        return self.ScaleValue(ret_s, self.lbounds_in, self.ubounds_in, -1.0, 1.0)

    def RescaleOutputs(self, o):
        # return self.ScaleValue(o,-1.0,1.0,self.lbounds_out,self.ubounds_out)
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
        # print real_inputs,'->',inputs
        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.ai[0:self.ni - 1] = inputs

        # hidden activations
        net = dot(transpose(self.wi), self.ai)
        self.ah = tanh(net)

        # output activations
        net = dot(transpose(self.wo), self.ah)
        self.ao = tanh(net)

        # rescales to the output interval from [-1,1] to output_ranges
        self.ro = self.RescaleOutputs(self.ao)
        return self.ro

    def backPropagate(self, real_targets, N, M):

        targets = self.ScaleValue(real_targets, self.lbounds_out, self.ubounds_out, -1.0, 1.0)

        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        od = self.dsigmoid(self.ao) * (targets - self.ao)
        # self.output_deltas = self.output_deltas + self.est_lr * (od - self.output_deltas) #+ uniform(-.01,.01)
        self.output_deltas = od + self.est_lr * (self.output_deltas - od)  # + uniform(-.01,.01)
        # self.output_deltas = od

        # calculate error terms for hidden
        error = dot(self.wo, self.output_deltas)
        hd = self.dsigmoid(self.ah) * error
        # self.hidden_deltas = self.hidden_deltas + self.est_lr * (hd - self.hidden_deltas ) #+ uniform(-.01,.01)
        self.hidden_deltas = hd + self.est_lr * (self.hidden_deltas - hd)  # + uniform(-.01,.01)
        # self.hidden_deltas = hd


        # update output weights
        change = self.output_deltas * reshape(self.ah, (self.ah.shape[0], 1))
        self.wo = self.wo + N * change + M * self.co
        self.co = change

        # update input weights
        change = self.hidden_deltas * reshape(self.ai, (self.ai.shape[0], 1))
        self.wi = self.wi + N * change + M * self.ci
        self.ci = change

        # calculate error
        error = sum(0.5 * (targets - self.ao) ** 2)
        # error = sum(0.5* (real_targets-self.ro)**2)

        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def arraytest(self, inputs, targets):
        N = inputs.shape[0]
        for i in range(N):
            print(inputs[i], "->", self.update(inputs[i]))

    def singletrain(self, inputs, targets):
        self.update(inputs)
        return self.backPropagate(targets, 0.5, 0.1)

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
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0 and i != 0:
                print('error ' + str(error))

    def arraytrain(self, inputs, targets, iterations = 100, N = 0.5, M = 0.1):
        # N: learning rate
        # M: momentum factor
        nelems = inputs.shape[0]
        for i in range(iterations):
            error = 0.0
            for j in range(nelems):
                self.update(inputs[j])
                error = error + self.backPropagate(targets[j], N, M)
            if i % 100 == 0 and i != 0:
                print('error ' + str(error))


def demoArr():
    # create a network with two input, two hidden, and one output nodes

    print("Array Tests")

    rang_input = [[1, 5]]
    rang_output = [[1, 25]]

    inputs = arange(1, 6)
    targets = inputs ** 2

    print(inputs)
    print(targets)

    a = time.clock()
    n = NN(rang_input, 10, rang_output)

    n.arraytrain(inputs, targets, 1000)
    # test it
    n.arraytest(inputs, targets)

    b = time.clock()
    print("Total time for Back Propagation Trainning ", b - a)


def demo():
    # create a network with two input, two hidden, and one output nodes
    pat = []
    rang_input = [[1, 5]]
    rang_output = [[1, 25]]

    for i in range(1, 6):
        inputs = [i]
        targets = [i ** 2]
        pat.append([inputs, targets])

    print(pat)

    a = time.clock()
    n = NN(rang_input, 10, rang_output)

    n.train(pat, 1000)
    # test it
    n.test(pat)

    b = time.clock()
    print("Total time for Back Propagation Trainning ", b - a)


def demoavg():
    # create a network with two input, two hidden, and one output nodes
    rang_input = [[1, 20], [1, 20]]
    rang_output = [[1, 20], [1, 20]]

    a = time.clock()
    n = NN(rang_input, 10, rang_output)

    # train it with some patterns
    print("Starting single step training")
    for i in range(1000):
        error = 0.0
        # for 1
        inputs = [1, 3]
        targets = [normal(5, 0.5), normal(5, 0.5)]  # noisy input
        error = error + n.singletrain(inputs, targets)

        # for 5
        inputs = [10, 2]
        targets = [normal(8, 0.5), normal(8, 0.5)]  # noisy input
        error = error + n.singletrain(inputs, targets)

        # for 10
        inputs = [17, 14]
        targets = [normal(15, 0.5), normal(15, 0.5)]  # noisy input
        error = error + n.singletrain(inputs, targets)

        if i % 100 == 0 and i != 0:
            print('error ' + str(error))
            # if error < 1:
            #        break
    # test it



    # n.test(pat)
    # for 1
    print("[1]===>", n.update([1, 3]))
    print("[10]===>", n.update([10, 2]))
    print("[17]===>", n.update([17, 14]))

    b = time.clock()
    print("Total time for Back Propagation Trainning ", b - a)


if __name__ == '__main__':
    demoArr()
    demo()
    demoavg()
