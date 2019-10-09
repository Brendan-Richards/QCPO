from itertools import combinations
from NetworkStuff import *

class Params:
    def __init__(self):
        self.I = np.array([[1, 0], [0, 1]])
        self.Sx = np.array([[0, 1], [1, 0]])
        self.Sy = np.array([[0, -1j], [1j, 0]])
        self.Sz = np.array([[1, 0], [0, -1]])
        self.hadamard = np.array([[1 / (np.sqrt(2)), 1 / (np.sqrt(2))],
                     [1 / (np.sqrt(2)), -1 / (np.sqrt(2))]])

        #quantum info parameters
        self.numt = 1000
        self.num_qubits = 1
        self.H0 = 0*self.I
        #self.H0 = 0 * np.kron(self.I, self.I)
        #self.H0 = 0 * np.kron(np.kron(I, I), I)
        self.tTotal = 1
        #self.numt = 1000
        self.hks, self.labels = self.get_hks(self.num_qubits)
        self.num_controls = len(self.hks)
        #self.num_controls = 3
        self.dt = self.tTotal/self.numt
        self.initType = "sinusoidal"
        self.parallel = False
        self.tolerance = 1e-4
        #self.target = np.kron(self.Sx, self.Sx)
        self.target = self.hadamard
        self.fourier_amps = [1, 5.4, 6.2, .3, 2.2]
        self.fourier_freqs = [1, 2, 3, 4, 5]
        self.fourier_max_amp = 2
        self.fourier_max_freq = 100
        self.fourier_max_phase = 2*np.pi
        self.fourier_max_terms = 10
        self.num_fourier_terms = 4

        #Genetic algorithm parameters
        self.pop_size = 1000
        self.curr_gen = 1
        self.max_gens = 200000000
        self.stop = False
        self.tourney_size = 10
        self.avg_fitness = 0
        self.halloffame = []
        self.pop = []
        self.new_pop = []
        self.mutation_prob = 0.15
        self.solution_guy = None

        #neural network parameters
        #linux paths
        self.data_location = "/home/brendan/Dropbox/stuffforlinux/python_projects/training_data"
        self.predict_input_location = "/home/brendan/Dropbox/stuffforlinux/python_projects/clusterexpansion_AlGaN"
        #self.predict_input_location = "/home/brendan/Dropbox/stuffforlinux/python_projects/prediction_data"
        #self.predict_input_location = "/home/brendan/Dropbox/stuffforlinux/python_projects/test_prediction"
        self.predict_output_location = "/home/brendan/Dropbox/stuffforlinux/python_projects"
        """
        #windows paths
        self.data_location = "C:\\Users\\Brendan\\Dropbox\\stuffforlinux\\python_projects\\training_data"
        self.predict_input_location = "C:\\Users\\Brendan\\Dropbox\\stuffforlinux\\python_projects\\prediction_data"
        self.predict_output_location = "C:\\Users\\Brendan\\Dropbox\\stuffforlinux\\python_projects"
        """
        self.num_epochs = 1600
        self.num_steps_per_epoch = 50
        self.my_batch_size = 50
        self.learning_rate = .0001
        self.net_train_size = 20000
        self.model_fname = "./neural_net_models/1qubit_hadamard.h5"
        self.model = tf.keras.models.load_model(self.model_fname)


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
        self.mats = []
        self.fourier_amps = []
        self.fourier_freqs = []
        self.fourier_phases = []

        if(len(parents) == 0):
            # make new individual
            self.init_fourier(p)
            self.amps = self.initAmps(p)
        elif(len(parents) == 2):
            #do crossover
            #self.amps = np.zeros((len(parents[0].amps), len(parents[1].amps[0])))
            self.avg_crossover(parents, p)
            self.mutate(p)

        else:
            print("received unexpected number of parents...exiting")
            exit(-1)

    def init_fourier(self, p):
        #self.num_fourier_terms = rand.randint(0, p.fourier_max_terms)
        self.num_fourier_terms = p.num_fourier_terms
        for j in range(p.num_controls):
            t1 = []
            t2 = []
            t3 = []
            for i in range(self.num_fourier_terms):
                t1.append(rand.random() * p.fourier_max_amp)
                t2.append(rand.random() * p.fourier_max_freq)
                t3.append(rand.random() * p.fourier_max_phase)
            self.fourier_amps.append(t1)
            self.fourier_freqs.append(t2)
            self.fourier_phases.append(t3)

    def fourier(self, x, k):
        val = np.zeros(len(x))
        for i in range(len(self.fourier_amps[k])):
            val += (self.fourier_amps[k][i]*np.sin(self.fourier_freqs[k][i]*x + self.fourier_phases[k][i]))
        return val

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

    def avg_crossover(self, parents, p):
        for k in range(len(parents[0].fourier_amps)): # loop through controls
            t1 = []
            t2 = []
            t3 = []
            for i in range(len(parents[0].fourier_amps[0])): #loop through fourier amplitudes
                t1.append((parents[0].fourier_amps[k][i]+parents[1].fourier_amps[k][i])/2.0)
                t2.append((parents[0].fourier_freqs[k][i] + parents[1].fourier_freqs[k][i]) / 2.0)
                t3.append((parents[0].fourier_phases[k][i] + parents[1].fourier_phases[k][i]) / 2.0)
            self.fourier_amps.append(t1)
            self.fourier_freqs.append(t2)
            self.fourier_phases.append(t3)
        self.amps = self.initAmps(p)

    # def mutate(self, mutation_prob):
    #     for k in range(len(self.amps)):
    #         for m in range(0, len(self.amps[0])):
    #             prob = rand.random()
    #             #print("prob: " + str(prob))
    #             if(mutation_prob > prob):
    #                 #print("mutating...")
    #                 self.amps[k, m] = self.amps[k, m]*rand.random()*2

    def mutate(self, p):
        prob = rand.random()
                #print("prob: " + str(prob))
        if(p.mutation_prob > prob):
            mut_type = 0
            #mut_type = rand.randint(0, 2)
            control = rand.randint(0, p.num_controls-1)
            if(mut_type == 0):
                b = rand.randint(0, len(self.fourier_amps[0]) - 1)
                self.fourier_amps[control][b] = rand.random()*p.fourier_max_amp
            elif(mut_type == 1):
                self.fourier_freqs[control][rand.randint(0, len(self.fourier_freqs[0])-1)] = rand.random() * p.fourier_max_freq
            elif(mut_type == 2):
                self.fourier_phases[control][rand.randint(0, len(self.fourier_phases[0])-1)] = rand.random() * p.fourier_max_phase
            self.initAmps(p)


    # returns an array of amplitudes in the range (-1,1)
    # there is one row for each of the numk controls and one column for each of the numt timesteps
    def initAmps(self, p):

        if p.initType == "random":
            return 2 * np.random.random_sample((p.num_controls, p.numt)) - 1
        if p.initType == "linear":
            myList = []
            for _ in range(p.num_controls):
                myList.append(np.linspace(-1 * (p.numt * p.dt), p.numt * p.dt, p.numt))
                # myList.append(np.linspace(0, numt * dt, numt).tolist())
            return np.array(myList) / (p.numt * p.dt)
        if p.initType == "sinusoidal":
            myList = []
            temp = []
            for k in range(p.num_controls):
                myList.append(np.linspace(0, p.tTotal, p.numt))
                temp.append(self.fourier(myList[k], k))
            #for x in temp:
             #   plt.plot(np.linspace(0, p.tTotal, p.numt), x)
             #   plt.show()
            return np.array(temp)

#########################################################################################
class evolver:
    def __init__(self, params):
        self.p = params

    def eval_fitness(self):
        temp = np.array([x.amps.flatten() for x in self.p.pop])

        fitness_array = self.p.model.predict(temp).flatten()
        for r in range(len(self.p.pop)):
            self.p.pop[r].fitness = fitness_array[r]
            if (self.p.pop[r].fitness > 1 - self.p.tolerance and self.p.pop[r].fitness < 1 + self.p.tolerance):
                print("possible solution found")
                self.eval_fitness_final(self.p.pop[r])
        self.average_fitness = np.mean(fitness_array)
        print("average population fitness: " + str(self.average_fitness))

    def eval_fitness_final(self, guy):
        print("checking fitness...")
        U = []
        x = []
        for i in range(self.p.numt):  # loop over time steps
            mat = self.p.H0  # matrix in the exponential
            for k in range(self.p.num_controls):  # loop over controls and add to hamiltonian
                mat = np.add(mat, guy.amps[k][i] * self.p.hks[k])

            mat = -1 * 1j * self.p.dt * mat
            #print(mat)
            U.append(expm(mat))  # do the matrix exponential
            if i == 0:
                x.append(U[i])
            else:
                x.append(np.matmul(U[i], x[i - 1]))
        guy.fitness = self.fidelity(self.p.target, x)
        if(guy.fitness > 1-self.p.tolerance):
            print("solution found")
            self.p.stop = True
            self.p.solution_guy = guy
        print("solution fitness: " + str(guy.fitness))


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
        #print("in check_condition()")
        print("current generation is: " + str(self.p.curr_gen))
        if(self.p.curr_gen > self.p.max_gens):
            print("reached max generation, stopping")
            exit(-1)

    def next_gen(self):
        #print("in next_gen()")
        self.p.new_pop = []
        for i in range(self.p.pop_size):
            self.p.new_pop.append(Individual(self.select_parents(), self.p))
        self.p.pop = self.p.new_pop
        self.p.curr_gen += 1

    def select_parents(self):
        participants = []
        for i in range(self.p.tourney_size):
            participants.append(self.p.pop[rand.randint(0, len(self.p.pop)-1)])

        winners = []

        winners.append(self.find_nearest(participants, 1-self.p.tolerance))
        participants.remove(winners[0])
        winners.append(self.find_nearest(participants, 1 - self.p.tolerance))

        return winners

    def find_nearest(self, a, value):
        fit_vals = [x.fitness for x in a]
        array = np.asarray(fit_vals)
        idx = (np.abs(array - value)).argmin()
        return a[idx]

    def output_results(self):
        winner = self.p.solution_guy
        print("in output_results()")
        for k in range(len(winner.amps)):
            plt.plot(np.linspace(0, self.p.numt * self.p.dt, self.p.numt), winner.amps[k])
            plt.title("Control Hamiltonian: " + self.p.labels[k])
            plt.xlabel("time")
            plt.ylabel("Amplitude")
            plt.savefig("genetic_results/control_" + self.p.labels[k] + ".pdf")
            plt.clf()
            #plt.show()

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

    #create_new_model(p)
    #test_crossover()


def test_crossover():
    p1 = Individual([], Params())
    p1.amps = np.array([[1, 1, 1, 1, 1], [5, 5, 5, 5, 5]])
    print("parent 1 amps: ")
    print(p1.amps)
    p2 = Individual([], Params())
    p2.amps = np.array([[10, 10, 10, 10, 10], [16, 16, 16, 16, 16]])
    print("parent 2 amps: ")
    print(p2.amps)
    parents = np.array([p1, p2])
    a = Individual(parents, Params())
    print("child amps:")
    print(a.amps)


if __name__ == '__main__':
    main()