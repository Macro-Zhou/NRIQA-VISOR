
import numpy as np
import math


def srocc(data1, data2):
    dat1 = np.array(data1)
    dat2 = np.array(data2)
    ind1 = np.argsort(dat1)
    ind2 = np.argsort(dat2)
    ind1 = np.argsort(ind1)
    ind2 = np.argsort(ind2)
    D = ind1 - ind2
    Num = len(dat1)
    srcc = 1 - 6 * (np.inner(D, D)) / (Num * (Num * Num - 1))

    return srcc

def pearsoncc(data1, data2):
    dat1 = np.array(data1)
    dat2 = np.array(data2)
    m1 = dat1.mean()
    m2 = dat2.mean()
    plcc = (np.dot((dat1 - m1), (dat2 - m2)) + 1e-6) / (np.sqrt(np.dot((dat1 - m1), (dat1 - m1))
                                                                * np.dot((dat2 - m2), (dat2 - m2))) + 1e-6)

    return plcc

def kendallcc(data1, data2):
    dat1 = np.array(data1)
    dat2 = np.array(data2)
    num = dat1.size
    tau = 0
    for i in range(num):
        for j in range(i):
            r1 = dat1[j] - dat1[i]
            if r1 > 0:
                r1 = 1
            elif r1 < 0:
                r1 = -1
            r2 = dat2[j] - dat2[i]
            if r2 > 0:
                r2 = 1
            elif r2 < 0:
                r2 = -1
            tau += r1 * r2
    krcc = 2 * tau / (num * (num - 1))

    return krcc

def rootMSE(data1, data2):
    dat1 = np.array(data1)
    dat2 = np.array(data2)
    mse = ((dat1 - dat2) * (dat1 - dat2)).mean()
    rmse = np.sqrt(mse)

    return rmse


# a = [1,2,3,4,5,6,7,8]
# b = [2,1,3,4,5,6,8,7]

# rusult in matlab
# srcc = 0.9000
# plcc = 0.9000
# krcc = 0.8571
# rmse = 0.7071

# srcc = srocc(a,b)
# print(srcc)
# plcc = pearsoncc(a,b)
# print(plcc)
# krcc = kendallcc(a,b)
# print(krcc)
# rmse = rootMSE(a,b)
# print(rmse)