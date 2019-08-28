import numpy as np
import random as rand

class Params:
    def __init__(self):
        self.num_controls = 10
        self.numt = 12
        self.dt = 12
        self.initType = "dfgsdf"

class Individual:
    def __init__(self, parents, p):

        self.mutation_prob = .15

        if(len(parents) == 0):
            # make new individual
            self.amps = self.initAmps(p.num_controls, p.numt, p.dt, p.initType)
        elif(len(parents) == 2):
            #do crossover
            print(parents)
            print(parents[0].shape)
            self.amps = np.zeros(parents[0].amps.shape[0], parents[0].amps.shape[1])
            self.crossover(parents)
            self.mutate()
        else:
            print("received unexpected number of parents...exiting")
            exit(-1)

    def crossover(self, parents):
        for k in range(len(parents[0].amps)):
            cross_point = rand.randint(0, len(parents[0]))
            for m in range(0, cross_point):
                self.amps[k][m] = parents[0][k][m]
            for n in range(cross_point, len(parents[0].amps)):
                self.amps[k][n] = parents[1][k][n]
        print(self.amps)

    def mutate(self):
        None

    # returns an array of amplitudes in the range (-1,1)
    # there is one row for each of the numk controls and one column for each of the numt timesteps
    def initAmps(self, numk, numt, dt, initType):

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

class evolver:
    def __init__(self):
        self.pop_size = 100
        self.curr_gen = 1
        self.max_gens = 200
        self.stop = False
        self.tourney_size = 5
        self.avg_fitness = 0
        self.halloffame = []
        self.pop = []
        self.new_pop = []

    def eval_fitness(self):
        None

    def init_pop(self):
        None

    def check_condition(self):
        print("in check_condition()")
        print("current generation is: " + str(self.curr_gen))
        if(self.curr_gen > self.max_gens):
            exit(-1)

    def next_gen(self):
        print("in next_gen()")
        self.new_pop = []
        self.curr_gen += 1

    def select_parents(self):
        participants = []
        for i in range(self.tourney_size):
            participants.append(self.pop[rand.randint(0, len(self.pop)-1)])
        participants.sort(key=lambda x: x.fitness, reverse=False)

        return participants[0:2]

    def output_results(self):
        print("in output_results()")

    def evolve(self):
        self.init_pop()
        self.eval_fitness()
        self.check_condition()
        while not(self.stop):
            self.next_gen()
            self.eval_fitness()
            self.check_condition()
        self.output_results()


def main():
    #a = evolver()
    #a.evolve()
    p1 = Individual([], Params())
    p1.amps = [1, 2, 3, 4, 5]
    p2 = Individual([], Params())
    p2.amps = [6, 7, 8, 9, 10]
    parents = np.array([p1, p2])
    a = Individual(parents, Params())


if __name__ == '__main__':
    main()