import numpy as np
import random as rand

class Individual:
    def __init__(self, parents):
        None

    def crossover(self, parents):
        None

    def mutate(self):
        None

class evolver:
    def __init__(self):
        self.pop_size = 100
        self.curr_gen = 1
        self.max_gens = 200
        self.stop = False
        self.tourney_size = 5
        self.avg_fitness = 100000
        self.halloffame = []
        self.pop = []
        self.new_pop = []
        self.max_mat_size = 28
        self.model_location = "/home/brendan/Dropbox/stuffforlinux/python_projects/models/longlonglongmodel.h5"

    def eval_fitness(self):
        None

    def init_pop(self):
        None

    def check_condition(self):
        print("in check_condition()")
        print("current generation is: " + str(self.curr_gen))

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
    a = evolver()
    a.evolve()

main()


