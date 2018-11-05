from UserDict import UserDict

class odict(UserDict):
    def __init__(self, dict = None):
        self._keys = []
        UserDict.__init__(self, dict)

    def __delitem__(self, key):
        UserDict.__delitem__(self, key)
        self._keys.remove(key)

    def __setitem__(self, key, item):
        UserDict.__setitem__(self, key, item)
        if key not in self._keys: self._keys.append(key)

    def clear(self):
        UserDict.clear(self)
        self._keys = []

    def copy(self):
        dict = UserDict.copy(self)
        dict._keys = self._keys[:]
        return dict

    def items(self):
        return list(zip(self._keys, list(self.values())))

    def keys(self):
        return self._keys

    def popitem(self):
        try:
            key = self._keys[-1]
        except IndexError:
            raise KeyError('dictionary is empty')

        val = self[key]
        del self[key]

        return (key, val)

    def setdefault(self, key, failobj = None):
        UserDict.setdefault(self, key, failobj)
        if key not in self._keys: self._keys.append(key)

    def update(self, dict):
        UserDict.update(self, dict)
        for key in list(dict.keys()):
            if key not in self._keys: self._keys.append(key)

    def values(self):
        return list(map(self.get, self._keys))


def dict2ts(ts):
    a= str(list(ts.items()))[3:-3]
    a = a.replace(',','')
    a = a.replace("'",'')
    a = a.replace('{','')
    a = a.replace('}','')
    a = a.replace('(','')
    a = a.replace(')','')
    a = a.replace('[[','(')
    a = a.replace('[','(')
    a = a.replace(']]',']')
    a = a.replace(']',')')
    a = a.replace(':','')
    return a


ts = odict()
ts['VERSION'] = 'RL-Glue-3.0'
ts['PROBLEMTYPE'] = 'episodic'
ts['DISCOUNTFACTOR'] = 1.0

ts['OBSERVATIONS']   = odict()
ts['OBSERVATIONS']['DOUBLES']= []
ts['OBSERVATIONS']['DOUBLES'].append(['-1.6' , '1.5'])


ts['ACTIONS']   = odict()
ts['ACTIONS']['INTS']= []
ts['ACTIONS']['INTS'].append(['0' , '2'])

ts['OBSERVATIONS']['DOUBLES'].append(['-0.07' , '0.07'])

ts['REWARDS'] = []
ts['REWARDS'].append(['-1', '0'])
print(ts)
#ts['EXTRA'] = 'Yes, Yea, JAMH is a genius; recognize it!!!'

a = dict2ts(ts)

a+= " EXTRA Yes, Yea, JAMH is a genius; recognize it!!! "
print("Generated")
print(a)
print()
print("manually")
print('VERSION RL-Glue-3.0 PROBLEMTYPE episodic DISCOUNTFACTOR 1.0 OBSERVATIONS DOUBLES (-1.6 1.5) (-0.07 0.07) ACTIONS INTS (0 2)  REWARDS (-1 0)')


