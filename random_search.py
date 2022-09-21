from citylist import CityList
from parameters import *
import numpy as np

class RandomSearch:
    def __init__(self, citylist, pop_size=POPULATION_SIZE):
        self.city = citylist
        self.pop_size = pop_size
        self.n_city = self.city.n_city
        self.population = None
        self.route_length = np.zeros(self.pop_size)
        self.best = None
        self.worst = None
        self.first_generation()

    def random_route(self):
        p = []
        for i in range(self.pop_size):
            new_indvidual = np.arange(self.n_city)
            np.random.shuffle(new_indvidual)
            p.append(new_indvidual)
        self.population = np.array(p, dtype=np.int16)
        return

    def first_generation(self):
        self.random_route()
        self.get_route_length()
        self.best = self.route_length.min()
        self.worst = self.route_length.max()

    def get_route_length(self):
        for i in range(self.pop_size):
            self.route_length[i] = self.city.get_route_length(self.population[i])
        if (self.best is None) or (self.route_length.min() < self.best):
            self.best = self.route_length.min()
        if (self.worst is None) or (self.route_length.max() > self.worst):
            self.worst = self.route_length.max()

def test():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")

    cl = CityList()
    cl.readtxt("data/tsp4test.txt")
    rs = RandomSearch(cl, POPULATION_SIZE)
    best = []
    worst = []
    for i in range(N_GENERATION):
        best.append(rs.best)
        worst.append(rs.worst)
        rs.random_route()
        rs.get_route_length()
        print(i, ":\ ", rs.route_length.min())
    best = np.array(best, dtype=np.float32)
    np.save("random_search_best.npy", best)
    plt.plot(np.arange(N_GENERATION)*100, best)
    plt.xlabel(r"$Evaluations \times 100$")
    plt.ylabel(r"$Shortest\ Path$")
    plt.title(r"Random Search", size="x-large")
    plt.show()

if __name__ == "__main__":
    test()
