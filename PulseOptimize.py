import numpy as np
from scipy.linalg import expm
import math
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from itertools import combinations
import time as t

##################################################


# runs the grape algorithm for a target unitary and returns a 2-d array of control field amplitudes
# parameters:
# tgt - the target unitary operator, a matrix
# H0 - the intrinsic hamiltonian, a matrix
# Hks - a list of the external field hamiltonians, a list of matricies
# numt - the number of discrete time steps
# dt - the length of each time step
def runGrape(tgt, H0, Hks, numt, dt, initType, tolerance, parallel):
    amps = initAmps(len(Hks), numt, dt, initType)

    g = 0
    while True:
        X, P = calcXPs(tgt, amps, H0, Hks, numt, dt)
        stop = checkCondition(tgt, X, P, tolerance)
        if stop:
            break
        else:
            g += 1
            # print("updating amps again: " + str(g))
            updateAmps(amps, P, X, dt, Hks)

    return amps, X

########################################################################
# returns an array of amplitudes in the range (-1,1)
# there is one row for each of the numk controls and one column for each of the numt timesteps
def initAmps(numk, numt, dt, initType):

    if initType == "random":
        return 2 * np.random.random_sample((numk, numt)) - 1
    if initType == "linear":
        myList = []
        for _ in range(numk):
            myList.append(np.linspace(-1 * (numt * dt), numt * dt, numt).tolist())
            # myList.append(np.linspace(0, numt * dt, numt).tolist())
        return np.array(myList) / (numt * dt)
    if initType == "sinusoidal":
        myList = []
        for _ in range(numk):
            myList.append(np.linspace(0, numt / 50, numt).tolist())
        # for i in range(len(myList)):
        # myList[i] = np.sin(myList[i])
        return np.sin(myList)


# calculates all the Xi's and Pi's and returns an array of both
def calcXPs(tgt, amps, H0, Hks, numt, dt):
    x = []
    p = []
    U = []

    for i in range(numt):  # loop over time steps
        mat = H0  # matrix in the exponential
        for k in range(len(Hks)):  # loop over controls and add to hamiltonian
            mat = np.add(mat, amps[k][i] * Hks[k])

        mat = -1 * 1j * dt * mat
        U.append(expm(mat))  # do the matrix exponential
        if i == 0:
            x.append(U[i])
        else:
            x.append(np.matmul(U[i], x[i - 1]))

    for i in range(numt - 1, -1, -1):
        if i == (numt - 1):
            p = [tgt]
        else:
            p = [np.matmul(U[i + 1].conjugate().transpose(), p[0])] + p

    return x, p


#########################################################################################
def calcXPsParallel(tgt, amps, H0, Hks, numt, dt):
    x = []
    p = []
    U = []

    pool = mp.Pool(mp.cpu_count())
    Jmats = [pool.apply(f2, args=(H0, Hks, amps, i, dt, x, U)) for i in range(numt)]
    pool.close()


    for i in range(numt - 1, -1, -1):
        if i == (numt - 1):
            p = [tgt]
        else:
            p = [np.matmul(U[i + 1].conjugate().transpose(), p[0])] + p

    return x, p


# checks if we should stop iterating
# by calculating hilbert schmidt product
def checkCondition(tgt, X, P, tolerance):
    val = HS(tgt, X[-1]) * HS(X[-1], tgt)
    print("Fidelity is: " + str(val))
    if val > (1 - tolerance) and val < (1 + tolerance):
        return True
    else:
        return False


def updateAmps(amps, P, X, dt, Hks):
    for i in range(len(X)):  # loop over time steps

        for k in range(len(Hks)):
            # update the amplitudes based on the derivatives
            derivative = -2 * ((HS(P[i], 1j * dt * np.matmul(Hks[k], X[i]))) * HS(X[i], P[i])).real
            amps[k][i] = amps[k][i] + epsilon * derivative


# Hilbert-Schmidt-Product of two matrices M1, M2
def HS(M1, M2):
    v1 = (np.matmul(M1.conjugate().transpose(), M2)).trace()
    # print(M1.shape[0])
    return v1 / M1.shape[0]



# checks whether array arr is a unitary matrix
###############################################################################
def is_unitary(arr):
    m = np.matrix(arr)
    return np.allclose(np.eye(m.shape[0]), m.H * m)


###################################################################################
def printInitialStuff(targ, H0, Hks, dt, numt, initType, tolerance, tTotal, exec_time):

    HkString = ""
    for k in Hks:
        HkString += (np.array2string(k) + "\n")

    text = "target is: \n" + np.array2string(targ) + "\nH0 is: \n" + np.array2string(H0)\
            + "\nHK's: \n" + HkString + "\ndt is: " + str(dt) + "\nnumt is: " + str(numt)\
            + "\ninitType is: " + initType\
            + "\ntolerance is: " + str(tolerance) + "\nepsilon is: " + str(epsilon)\
            + "\ntTotal is: " + str(tTotal) + "\nexecution time is: " + str(exec_time) + " seconds"
    #print(text + "\n\n\n\n")

    return text

#####################################################################################
def printResults(amps, result_matrix, numt, dt, Hks, labels, text):

    text = text + "\nresult matrix: \n" + np.array2string(result_matrix[-1])

    #print(text)

    all_files = os.listdir("results/")
    highest = 0
    for f in all_files:
        num = int(f[3:])
        if num > highest:
            highest = num

    dir = "results/run" + str(highest+1)
    os.mkdir(dir)

    for k in range(len(amps)):
        plt.plot(np.linspace(0, numt * dt, numt), amps[k])
        plt.title("Control Hamiltonian: " + labels[k])
        plt.xlabel("time")
        plt.ylabel("Amplitude")
        plt.savefig(dir + "/control_" + labels[k] + ".pdf")
        plt.clf()
        #plt.show()

    for k in range(len(amps)):
        #np.savetxt(dir + "/control_" + str(k) + "_amplitudes.txt", amps[k], delimiter='\n')
        np.savetxt(dir + "/" + labels[k] + "_amplitudes.txt", amps[k], delimiter='\n')

    myFile = open(dir + "/results.txt", "w+")
    myFile.write(text)


# n is the number of qubits to make controls for
def get_hks(n):
    if(n<1):
        print("Error, cant make controls for less than 1 qubits")
        exit(-1)

    hks = []
    labels = []
    for i in range(n):
        if(i==0):
            t1 = "X"
            t2 = "Y"
            t3 = "Z"
            matx = Sx
            maty = Sy
            matz = Sz
        else:
            t1 = "I"
            t2 = "I"
            t3 = "I"
            matx = I
            maty = I
            matz = I
            for j in range(i-1):
                t1 += "I"
                t2 += "I"
                t3 += "I"
                matx = np.kron(matx, I)
                maty = np.kron(maty, I)
                matz = np.kron(matz, I)
            t1 += "X"
            t2 += "Y"
            t3 += "Z"
            matx = np.kron(matx, Sx)
            maty = np.kron(maty, Sy)
            matz = np.kron(matz, Sz)

        for j in range(n-i-1):
            t1 += "I"
            t2 += "I"
            t3 += "I"
            matx = np.kron(matx, I)
            maty = np.kron(maty, I)
            matz = np.kron(matz, I)

        hks.append(matx)
        hks.append(maty)
        hks.append(matz)
        labels.append(t1)
        labels.append(t2)
        labels.append(t3)

    print()
    combs = combinations(np.arange(n).tolist(), 2)
    total = 0
    for c in combs:
        total += 1
        if c[0] == 0 or c[1] == 0:
            t = "Z"
            mat = Sz
        else:
            t = "I"
            mat = I
        for k in range(1, n):
            if k == c[0] or k == c[1]:
                t += "Z"
                mat = np.kron(mat, Sz)
            else:
                t += "I"
                mat = np.kron(mat, I)
        hks.append(mat)
        labels.append(t)

    # number of controls should be 3n + n choose 2
    num = 3*n + total
    assert len(hks) == num

    return hks, labels



###################################################################################3
I = np.array([[1, 0], [0, 1]])
Sx = np.array([[0, 1], [1, 0]])
Sy = np.array([[0, -1j], [1j, 0]])
Sz = np.array([[1, 0], [0, -1]])
hadamard = np.array([[1 / (math.sqrt(2)), 1 / (math.sqrt(2))],
                     [1 / (math.sqrt(2)), -1 / (math.sqrt(2))]])
xx_plus_yy = np.add(np.kron(Sx, Sx), np.kron(Sy, Sy))
xy_minus_yx = np.add(np.kron(Sx, Sy), -1 * np.kron(Sy, Sx))
zz = np.kron(Sz, Sz)

##############################################################################
def testMyGrape():
    targ = expm(1j*(np.pi/4)*np.kron(np.kron(Sz, I), Sz))
    w = np.exp(1j*(math.pi/4))
    targ = (1/np.sqrt(8))*np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                     [1, w, np.power(w, 2), np.power(w, 3), np.power(w, 4), np.power(w, 5), np.power(w, 6), np.power(w, 7)],
                     [1, np.power(w, 2), np.power(w, 4), np.power(w, 6), 1, np.power(w, 2), np.power(w, 4), np.power(w, 6)],
                     [1, np.power(w, 3), np.power(w, 6), w, np.power(w, 4), np.power(w, 7), np.power(w, 2), np.power(w, 5)],
                     [1, np.power(w, 4), 1, np.power(w, 4), 1, np.power(w, 4), 1, np.power(w, 4)],
                     [1, np.power(w, 5), np.power(w, 2), np.power(w, 7), np.power(w, 4), w, np.power(w, 6), np.power(w, 3)],
                     [1, np.power(w, 6), np.power(w, 4), np.power(w, 2), 1, np.power(w, 6), np.power(w, 4), np.power(w, 2)],
                     [1, np.power(w, 7), np.power(w, 6), np.power(w, 5), np.power(w, 4), np.power(w, 3), np.power(w, 2), w]], dtype=complex)
    print(targ)
    num_qubits = 3
    H0 = 0 * np.kron(np.kron(I, I), I)
    tTotal = 1
    numt = 1000
    dt = tTotal/numt
    initType = "sinusoidal"
    parallel = False
    tolerance = 1e-4

    global epsilon
    #epsilon = (2/dt)  # the smaller epsilon, the slower the convergence
    epsilon = 2000

    hks, labels = get_hks(num_qubits)

    start = t.process_time()
    amps, result_matrix = runGrape(targ, H0, hks, numt, dt, initType, tolerance, parallel)
    end = t.process_time()

    exec_time = end - start
    print("GRAPE execution time: " + str(exec_time) + " seconds")
    text = printInitialStuff(targ, H0, hks, dt, numt, initType, tolerance, tTotal, exec_time)
    printResults(amps, result_matrix, numt, dt, hks, labels, text)





def main():
    testMyGrape()


if __name__ == '__main__':
    main()

