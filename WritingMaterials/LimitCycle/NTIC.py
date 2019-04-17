<<<<<<< HEAD
import dit
import itertools
import numpy as np
from numpy import array as npa
from pprint import pprint


def genAllComb(stateSpace, nVar):
    return list(itertools.product(stateSpace, repeat=nVar))


def genAllSubsets(X):
    subsets = [()]
    for iLen in range(len(X)):
        subset = list(itertools.combinations(X, iLen + 1))
        subsets = subsets + subset
    return subsets


def genRandTranTable(comb, isManyToOne=True):
    comb = npa(comb)
    nComb = comb.shape[0]
    prob = np.ones(nComb) / nComb
    iComb_ = np.random.choice(range(nComb), nComb, replace=isManyToOne, p=prob)
    comb_ = comb[iComb_]
    comb_ = npa2tuple(comb_)
    return comb, comb_, prob


def npa2str(a):
    """
    :param a: numpy array
    :return: string list
    """
    return [''.join(x) for x in a.astype(str)]


def npa2tuple(a):
    return [tuple(x) for x in a]


def conditional_mutual_information(d, X, Y, Z):
    """
    I(X;Y|Z)
    :param d: dit distribution obj
    :param X: vars1
    :param Y: vars2
    :param Z: condition vars
    :return: conditional mutual information
    """
    return dit.shannon.mutual_information(d, X, Y + Z) - dit.shannon.mutual_information(d, X, Z)


def non_trivial_informational_closure(d, Yt, Et, Yt_):
    return dit.shannon.mutual_information(d, Yt_, Yt) - conditional_mutual_information(d, Yt_, Yt, Et)


def freqTable(samples):
    comb, freq = np.unique(samples, return_counts=True, axis=0)
    prob = freq / np.sum(freq)
    return comb, prob


class RandomVariableSet:
    dd = None
    iRV_now = None
    iRV_next = None
    nRV = None
    Yt = None
    Et = None
    Yt_ = None
    Et_ = None

    def __init__(self):
        pass

    def CreateDistFromTimeSeries(self, ts):
        ts = np.hstack([ts[0:-1], ts[1:]])
        comb, prob = freqTable(ts)
        comb = npa2tuple(comb)
        self.__CreateDD(comb, prob)
        return self.dd

    def CreateDistFromTransitionTable(self, rv, rv_, prob):
        """
        :param rv: random variable for now
        :param rv_: transitional state of rv at next time
        :param prob: transition probability
        :return:
        """
        comb = np.hstack([rv, rv_])
        comb = npa2tuple(comb)
        self.__CreateDD(comb, prob)
        return self.dd

    def __CreateDD(self, comb, prob):
        dd = dit.Distribution(comb, prob)
        nRV = len(dd.rvs)
        dd.set_rv_names(range(nRV))
        self.iRV_now = list(range(nRV // 2))
        self.iRV_next = list(range(nRV // 2, nRV))
        self.nRV = nRV // 2
        self.dd = dd

    def setYtEt(self, Yt=None, Et=None):
        if Yt is not None:
            self.Yt = Yt
            self.Et = list(set(range(self.nRV)) - set(Yt))
        else:
            self.Et = Et
            self.Yt = list(set(range(self.nRV)) - set(Et))

        self.Yt_ = [x + self.nRV for x in self.Yt]
        self.Et_ = [x + self.nRV for x in self.Et]
        return self.Yt, self.Et, self.Yt_, self.Et_

    def getYtEt(self):
        return self.Yt, self.Et, self.Yt_, self.Et_

    def ntic(self):
        return non_trivial_informational_closure(self.dd, self.Yt, self.Et, self.Yt_)

    def print(self):
        print(self.dd)
        print('\n--- attributes ---')
        pprint(vars(self))


v = RandomVariableSet()
c = genRandTranTable(genAllComb('01', 3))
v.CreateDistFromTransitionTable(*c)
v.setYtEt(Et=[0, 1, 2])
v.ntic()
vars(v)

print(v.dd)

v.print()
=======
import dit
import itertools
import numpy as np
from numpy import array as npa
from pprint import pprint


def genAllComb(stateSpace, nVar):
    return list(itertools.product(stateSpace, repeat=nVar))


def genAllSubsets(X):
    subsets = [()]
    for iLen in range(len(X)):
        subset = list(itertools.combinations(X, iLen + 1))
        subsets = subsets + subset
    return subsets


def genRandTranTable(comb, isManyToOne=True):
    comb = npa(comb)
    nComb = comb.shape[0]
    prob = np.ones(nComb) / nComb
    iComb_ = np.random.choice(range(nComb), nComb, replace=isManyToOne, p=prob)
    comb_ = comb[iComb_]
    comb_ = npa2tuple(comb_)
    return comb, comb_, prob


def npa2str(a):
    """
    :param a: numpy array
    :return: string list
    """
    return [''.join(x) for x in a.astype(str)]


def npa2tuple(a):
    return [tuple(x) for x in a]


def conditional_mutual_information(d, X, Y, Z):
    """
    I(X;Y|Z)
    :param d: dit distribution obj
    :param X: vars1
    :param Y: vars2
    :param Z: condition vars
    :return: conditional mutual information
    """
    return dit.shannon.mutual_information(d, X, Y + Z) - dit.shannon.mutual_information(d, X, Z)


def non_trivial_informational_closure(d, Yt, Et, Yt_):
    return dit.shannon.mutual_information(d, Yt_, Yt) - conditional_mutual_information(d, Yt_, Yt, Et)


def freqTable(samples):
    comb, freq = np.unique(samples, return_counts=True, axis=0)
    prob = freq / np.sum(freq)
    return comb, prob


class RandomVariableSet:
    dd = None
    iRV_now = None
    iRV_next = None
    nRV = None
    Yt = None
    Et = None
    Yt_ = None
    Et_ = None

    def __init__(self):
        pass

    def CreateDistFromTimeSeries(self, ts):
        ts = np.hstack([ts[0:-1], ts[1:]])
        comb, prob = freqTable(ts)
        comb = npa2tuple(comb)
        self.__CreateDD(comb, prob)
        return self.dd

    def CreateDistFromTransitionTable(self, rv, rv_, prob):
        """
        :param rv: random variable for now
        :param rv_: transitional state of rv at next time
        :param prob: transition probability
        :return:
        """
        comb = np.hstack([rv, rv_])
        comb = npa2tuple(comb)
        self.__CreateDD(comb, prob)
        return self.dd

    def __CreateDD(self, comb, prob):
        dd = dit.Distribution(comb, prob)
        nRV = len(dd.rvs)
        dd.set_rv_names(range(nRV))
        self.iRV_now = list(range(nRV // 2))
        self.iRV_next = list(range(nRV // 2, nRV))
        self.nRV = nRV // 2
        self.dd = dd

    def setYtEt(self, Yt=None, Et=None):
        if Yt is not None:
            self.Yt = Yt
            self.Et = list(set(range(self.nRV)) - set(Yt))
        else:
            self.Et = Et
            self.Yt = list(set(range(self.nRV)) - set(Et))

        self.Yt_ = [x + self.nRV for x in self.Yt]
        self.Et_ = [x + self.nRV for x in self.Et]
        return self.Yt, self.Et, self.Yt_, self.Et_

    def getYtEt(self):
        return self.Yt, self.Et, self.Yt_, self.Et_

    def ntic(self):
        return non_trivial_informational_closure(self.dd, self.Yt, self.Et, self.Yt_)

    def print(self):
        print(self.dd)
        print('\n--- attributes ---')
        pprint(vars(self))


v = RandomVariableSet()
c = genRandTranTable(genAllComb('01', 3))
v.CreateDistFromTransitionTable(*c)
v.setYtEt(Et=[0, 1, 2])
v.ntic()
vars(v)

print(v.dd)

v.print()
>>>>>>> master
