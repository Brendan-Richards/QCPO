import multiprocessing as mp
import time as t
import numpy as np


def f1(a, c):
    for i in range(100000):
        b = 3
    return a*c

def do_parallel():
    for i in range(1000):
        p = mp.Process(target=f1, args=(i, 2))
        p.start()
        p.join()



    #pool = mp.Pool(mp.cpu_count())
    #result = pool.map(f1, np.arange(45))
    #pool.close()

def do_sequence():
    for j in range(1000):
        for i in range(100000):
            b = 3
        d = j*2

def main():

    start = t.process_time()
    do_parallel()
    end = t.process_time()
    parallel = end - start

    start = t.process_time()
    do_sequence()
    end = t.process_time()
    sequence = end-start

    print("parallel time: " + str(parallel) + " seconds")
    print("sequential time: " + str(sequence) + " seconds")

if __name__ == '__main__':
    main()