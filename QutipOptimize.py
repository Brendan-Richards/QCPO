import numpy as np
import os
import sys
import math
import matplotlib.pyplot as plt
from itertools import combinations

###############################################
from qutip import *
import qutip.control.grape as gr
import qutip.control.pulseoptim as cpo
from qutip.ui.progressbar import TextProgressBar
##################################################

################################################################################3####
# returns a new Qobj equal to "state" rotated about "axis" by "angle" on the bloch sphere
# angle is expected in radians
def blochRotate(axis, angle, state):
    xRot = Qobj([[np.cos(angle/2), -1 * 1j * np.sin(angle/2)],
                 [-1 * 1j * np.sin(angle/2), np.cos(angle/2)]])
    yRot = Qobj([[np.cos(angle/2), -1 * np.sin(angle/2)],
                 [ np.sin(angle/2), np.cos(angle/2)]])
    zRot = Qobj([[np.exp(-1*1j*angle/2), 0],
                 [0, np.exp(1j*angle/2)]])

    if axis.lower()=='x':
        return xRot * state
    elif axis.lower()=='y':
        return yRot * state
    elif axis.lower()=='z':
        return zRot * state
    else:
        print("couldn't do rotation, i'm confused about which axis you want to rotate about")

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

        hks.append(Qobj(matx))
        hks.append(Qobj(maty))
        hks.append(Qobj(matz))
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
        hks.append(Qobj(mat))
        labels.append(t)

    # number of controls should be 3n + n choose 2
    num = 3*n + total
    assert len(hks) == num

    return hks, labels

##################################################################################
def printResults(amps, numt, dt, labels):

    all_files = os.listdir("Qutip_results/")
    highest = 0
    for f in all_files:
        num = int(f[3:])
        if num > highest:
            highest = num

    dir = "Qutip_results/run" + str(highest+1)
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

    #myFile = open(dir + "/results.txt", "w+")
    #myFile.write(text)

###################################################################################3
I = np.array([[1,0], [0,1]])
Sx = np.array([[0, 1], [1, 0]])
Sy = np.array([[0, -1j], [1j, 0]])
Sz = np.array([[1, 0], [0, -1]])
hadamard = np.array([[1 / (math.sqrt(2)), 1 / (math.sqrt(2))],
                     [1 / (math.sqrt(2)), -1 / (math.sqrt(2))]])

def testQutipGrape():
    #sys.stdout =
    sys.stdout = open('error file.txt', 'w')
    H_d = Qobj(0 * np.kron(np.kron(I, I), I))
    H_c, labels = get_hks(3)
    U_0 = identity(8)
    #U_targ = hadamard_transform(1)
    #U_targ = Qobj([[np.cos(-10000), 1j*np.sin(-10000)], [1j*np.sin(-10000), np.cos(-10000)]])
    w = np.exp(1j * (math.pi / 4))
    U_targ = Qobj((1 / np.sqrt(8)) * np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, w, np.power(w, 2), np.power(w, 3), np.power(w, 4), np.power(w, 5),
                                         np.power(w, 6), np.power(w, 7)],
                                        [1, np.power(w, 2), np.power(w, 4), np.power(w, 6), 1, np.power(w, 2),
                                         np.power(w, 4), np.power(w, 6)],
                                        [1, np.power(w, 3), np.power(w, 6), w, np.power(w, 4), np.power(w, 7),
                                         np.power(w, 2), np.power(w, 5)],
                                        [1, np.power(w, 4), 1, np.power(w, 4), 1, np.power(w, 4), 1, np.power(w, 4)],
                                        [1, np.power(w, 5), np.power(w, 2), np.power(w, 7), np.power(w, 4), w,
                                         np.power(w, 6), np.power(w, 3)],
                                        [1, np.power(w, 6), np.power(w, 4), np.power(w, 2), 1, np.power(w, 6),
                                         np.power(w, 4), np.power(w, 2)],
                                        [1, np.power(w, 7), np.power(w, 6), np.power(w, 5), np.power(w, 4),
                                         np.power(w, 3), np.power(w, 2), w]], dtype=complex))
    n_ts = 1000
    tTotal = 1000
    amps_low = None
    amps_high = None
    tolerence = 1e-1
    max_iterations = 1e500
    max_runtime = 1e100
    init_options = ["RND", "LIN", "ZERO", "SINE", "SQUARE", "TRIANGLE", "SAW"]
    init_type = init_options[3]

    result = cpo.optimize_pulse(H_d, H_c, U_0, U_targ, num_tslots=n_ts, evo_time=tTotal,
                                amp_lbound=amps_low, amp_ubound=amps_high, fid_err_targ=tolerence,
                                max_iter=max_iterations, max_wall_time=max_runtime,
                                init_pulse_type=init_type, gen_stats=True, log_level=qutip.logging_utils.DEBUG)
    amps = []
    for i in range(len(labels)):
        amps.append(result.final_amps[:, i])
    print(len(amps))
    print(len(amps[0]))
    printResults(amps, n_ts, tTotal/(1.0*n_ts), labels)


testQutipGrape()

def testBlochStuff():
    b = Bloch()
    vec = basis(2, 0)
    print("initial state: ")
    print(vec)
    rotated = blochRotate('x', math.pi/2, vec)
    print("rotated state")
    print(rotated)
    b.add_states(vec)
    b.add_states(rotated)
    b.show()


#testBlochStuff()
