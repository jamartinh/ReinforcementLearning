from FAInterface import FARL
from Xbpaenn3 import NN


class NeuroQ(FARL):
    def __init__(self, nactions, input_ranges, nhidden, output_ranges, deep_in, deep_out, alpha = 0.3):
        self.nactions = nactions
        self.alpha = alpha

        # Create the function approximator
        self.Net = []
        for i in range(self.nactions):
            self.Net.append(NN(input_ranges, nhidden, output_ranges, deep_in, deep_out))  # the function approximator

        self.trace = []

    def UpdateRewards(self, r):
        for i in range(len(self.trace)):
            self.trace[i][2] += r

    def PushtTaces(self):
        for s, a, v in self.trace:
            self.Update(s, a, v)

    def GetValue(self, s, a = None):
        """ Return the Q value of state (s) for action (a)

        """
        if a is None:
            v = []
            for i in range(self.nactions):
                v.extend(self.Net[i].update(s))
            return v

        v = self.Net[a].update(s)
        if len(v) == 1:
            return v[0]

        return v

    def Addtrace(self, s, a, r):
        self.UpdateRewards(r)
        self.trace.append([s, a, r])

    def ResetTraces(self):
        self.trace = []

    def Update(self, s, a, v):
        """ update action value for action(a)

        """
        # qsa = self.GetValue(s,a)
        # qsa = qsa + self.alpha * (v - qsa)
        inputs = s
        targets = [v]
        pattern = [[inputs, targets]]
        self.Net[a].fast_train(pattern, 1, self.alpha, 0.3)

    def UpdateAll(self, s, v):
        """ update action value for action(a)

        """
        for i in range(self.nactions):
            self.Update(s, i, v[i])

    def HasPopulation(self):
        return False
