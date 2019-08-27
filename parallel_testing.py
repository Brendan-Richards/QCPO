import multiprocessing as mp
import numpy as np


def f1(a, b):
    return a+b


pool = mp.Pool(mp.cpu_count())
result = [pool.apply(f1, args=(i, 2)) for i in range(10)]
pool.close()


print(result)