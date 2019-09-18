import numpy as np
import random as rand
from itertools import combinations
from scipy.linalg import expm
import matplotlib.pyplot as plt

class Params:
    def __init__(self):
        self.I = np.array([[1, 0], [0, 1]])
        self.Sx = np.array([[0, 1], [1, 0]])
        self.Sy = np.array([[0, -1j], [1j, 0]])
        self.Sz = np.array([[1, 0], [0, -1]])

        #quantum info parameters
        self.numt = 100
        self.num_qubits = 2
        self.H0 = 0 * np.kron(self.I, self.I)
        #self.H0 = 0 * np.kron(np.kron(I, I), I)
        self.tTotal = 1
        #self.numt = 1000
        self.hks, self.labels = self.get_hks(self.num_qubits)
        self.num_controls = len(self.hks)
        self.dt = self.tTotal/self.numt
        self.initType = "linear"
        self.parallel = False
        self.tolerance = 1e-4
        self.target = np.kron(self.Sx, self.Sx)

        #Geneticc algorithm parameters
        self.pop_size = 100
        self.curr_gen = 1
        self.max_gens = 200000000
        self.stop = False
        self.tourney_size = 4
        self.avg_fitness = 0
        self.halloffame = []
        self.pop = []
        self.new_pop = []
        self.mutation_prob = 0.01
        self.solution_guy = None

    # n is the number of qubits to make controls for
    def get_hks(self, n):
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
                matx = self.Sx
                maty = self.Sy
                matz = self.Sz
            else:
                t1 = "I"
                t2 = "I"
                t3 = "I"
                matx = self.I
                maty = self.I
                matz = self.I
                for j in range(i-1):
                    t1 += "I"
                    t2 += "I"
                    t3 += "I"
                    matx = np.kron(matx, self.I)
                    maty = np.kron(maty, self.I)
                    matz = np.kron(matz, self.I)
                t1 += "X"
                t2 += "Y"
                t3 += "Z"
                matx = np.kron(matx, self.Sx)
                maty = np.kron(maty, self.Sy)
                matz = np.kron(matz, self.Sz)

            for j in range(n-i-1):
                t1 += "I"
                t2 += "I"
                t3 += "I"
                matx = np.kron(matx, self.I)
                maty = np.kron(maty, self.I)
                matz = np.kron(matz, self.I)

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
                mat = self.Sz
            else:
                t = "I"
                mat = self.I
            for k in range(1, n):
                if k == c[0] or k == c[1]:
                    t += "Z"
                    mat = np.kron(mat, self.Sz)
                else:
                    t += "I"
                    mat = np.kron(mat, self.I)
            hks.append(mat)
            labels.append(t)

        # number of controls should be 3n + n choose 2
        num = 3*n + total
        assert len(hks) == num

        return hks, labels


class Individual:
    def __init__(self, parents, p):

        self.fitness = 0.0

        if(len(parents) == 0):
            # make new individual
            self.amps = self.initAmps(p.num_controls, p.numt, p.dt, p.initType)
        elif(len(parents) == 2):
            #do crossover
            self.amps = np.zeros((len(parents[0].amps), len(parents[1].amps[0])))
            self.crossover(parents)
            self.mutate(p.mutation_prob)

        else:
            print("received unexpected number of parents...exiting")
            exit(-1)

    def crossover(self, parents):
        #print("Length: " + str(parents[0].amps.shape))
        for k in range(len(parents[0].amps)):
            cross_point = rand.randint(0, len(parents[0].amps[0]))
            #print("cross_point: " + str(cross_point))
            for m in range(0, cross_point):
                self.amps[k, m] = parents[0].amps[k, m]
            for n in range(cross_point, len(parents[0].amps[0])):
                self.amps[k, n] = parents[1].amps[k, n]
        #print(self.amps)

    def mutate(self, mutation_prob):
        for k in range(len(self.amps)):
            for m in range(0, len(self.amps[0])):
                prob = rand.random()
                #print("prob: " + str(prob))
                if(mutation_prob > prob):
                    #print("mutating...")
                    self.amps[k, m] = self.amps[k, m]*rand.random()



    # returns an array of amplitudes in the range (-1,1)
    # there is one row for each of the numk controls and one column for each of the numt timesteps
    def initAmps(self, numk, numt, dt, initType):

        if initType == "random":
            return 2 * np.random.random_sample((numk, numt)) - 1
        if initType == "linear":
            myList = []
            for _ in range(numk):
                myList.append(np.linspace(-1 * (numt * dt), numt * dt, numt))
                # myList.append(np.linspace(0, numt * dt, numt).tolist())
            return np.array(myList) / (numt * dt)
        if initType == "sinusoidal":
            myList = []
            for _ in range(numk):
                myList.append(np.linspace(0, numt / 50, numt))
            # for i in range(len(myList)):
            # myList[i] = np.sin(myList[i])
            return np.sin(myList)

#########################################################################################
class evolver:
    def __init__(self, params):
        self.p = params

    def eval_fitness(self):
        total_fitness = 0
        for r in range(len(self.p.pop)):
            U = []
            x = []
            for i in range(self.p.numt):  # loop over time steps
                mat = self.p.H0  # matrix in the exponential
                for k in range(self.p.num_controls):  # loop over controls and add to hamiltonian
                    mat = np.add(mat, self.p.pop[r].amps[k][i] * self.p.hks[k])

                mat = -1 * 1j * self.p.dt * mat
                #print(mat)
                U.append(expm(mat))  # do the matrix exponential
                if i == 0:
                    x.append(U[i])
                else:
                    x.append(np.matmul(U[i], x[i - 1]))
            self.p.pop[r].fitness = self.fidelity(self.p.target, x)
            total_fitness += self.p.pop[r].fitness
            if(self.p.pop[r].fitness > 1-self.p.tolerance):
                print("solution found")
                self.p.stop = True
                self.p.solution_guy = self.p.pop[r]
        self.average_fitness = total_fitness/len(self.p.pop)
        print("average population fitness: " + str(self.average_fitness))

    def fidelity(self, tgt, X):
        return self.HS(tgt, X[-1]) * self.HS(X[-1], tgt)

    # Hilbert-Schmidt-Product of two matrices M1, M2
    def HS(self, M1, M2):
        v1 = (np.matmul(M1.conjugate().transpose(), M2)).trace()
        return v1 / M1.shape[0]

    def init_pop(self):
        print("making initial population...")
        for i in range(self.p.pop_size):
            self.p.pop.append(Individual([], self.p))

    def check_condition(self):
        print("in check_condition()")
        print("current generation is: " + str(self.p.curr_gen))
        if(self.p.curr_gen > self.p.max_gens):
            print("reached max generation, stopping")
            exit(-1)

    def next_gen(self):
        print("in next_gen()")
        self.p.new_pop = []
        for i in range(self.p.pop_size):
            self.p.new_pop.append(Individual(self.select_parents(), self.p))
        self.p.pop = self.p.new_pop
        self.p.curr_gen += 1

    def select_parents(self):
        participants = []
        for i in range(self.p.tourney_size):
            participants.append(self.p.pop[rand.randint(0, len(self.p.pop)-1)])
        participants.sort(key=lambda x: x.fitness, reverse=True)

        return participants[0:2]

    def output_results(self):
        winner = self.p.solution_guy
        print("in output_results()")
        for k in range(len(winner.amps)):
            plt.plot(np.linspace(0, self.p.numt * self.p.dt, self.p.numt), winner.amps[k])
            plt.title("Control Hamiltonian: " + self.p.labels[k])
            plt.xlabel("time")
            plt.ylabel("Amplitude")
            plt.savefig("control_" + self.p.labels[k] + ".pdf")
            plt.clf()
            # plt.show()

    def evolve(self):
        self.init_pop()
        self.eval_fitness()
        self.check_condition()
        while not(self.p.stop):
            self.next_gen()
            self.eval_fitness()
            self.check_condition()
        self.output_results()


def main():
    p = Params()

    a = evolver(p)
    a.evolve()
    #test_crossover()


def test_crossover():
    p1 = Individual([], Params())
    p1.amps = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    print("parent 1 amps: ")
    print(p1.amps)
    p2 = Individual([], Params())
    p2.amps = np.array([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])
    print("parent 2 amps: ")
    print(p2.amps)
    parents = np.array([p1, p2])
    a = Individual(parents, Params())


if __name__ == '__main__':
    main()