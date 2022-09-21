from citylist import *
from parameters import *
from base_model import BaseModel
import numpy as np

class GA(BaseModel):
    def __init__(self, city_list, pop_size=POPULATION_SIZE,
                 cross_rate=CROSS_RATE, mutate_rate=MUTATE_RATE):
        super(GA, self).__init__(city_list)
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.fitness = -self.route_length + self.route_length.max()
        self.selected_pop = None
        self.best_length = 10000
        self.best_path = None
        self.find_best()

    def select_parent(self):
        self.fitness = -self.route_length + self.route_length.max()
        p = self.fitness / self.fitness.sum()
        selected_idx = np.random.choice(np.arange(self.pop_size),
                                        size=self.pop_size, replace=True, p=p)
        self.selected_pop = self.population[selected_idx]

    def crossover(self, p1, p2):
        cross_points = np.random.randint(0, self.n_city, 2)
        if cross_points[0] > cross_points[1]:
            dad_segment = np.array(p1[cross_points[1]:(cross_points[0]+1)])
            mon_segment = np.array([bit for bit in p2 if bit not in dad_segment])
        else:
            dad_segment = np.array(p2[cross_points[0]:(cross_points[1]+1)])
            mon_segment = np.array([bit for bit in p1 if bit not in dad_segment])
        offspring = np.concatenate((dad_segment, mon_segment))
        return offspring

    def mutate(self, gene):
        for pointA in range(self.n_city):
            if np.random.rand() < self.mutate_rate:
                pointB = np.random.randint(self.n_city)
                cityA = gene[pointA]
                gene[pointA] = gene[pointB]
                gene[pointB] = cityA
        return gene

    def select(self):
        pass

    def evolute(self):
        new_pop = np.zeros_like(self.population)
        self.get_route_length()
        self.select_parent()
        for i in range(self.pop_size):
            offspring = self.population[i]
            if np.random.rand() < self.cross_rate:
                offspring = self.crossover(offspring, self.selected_pop[i])
            offspring = self.mutate(offspring)
            new_pop[i] = offspring
        self.population = new_pop
        self.find_best()

    def find_best(self):

        if self.route_length.min() < self.best_length:
            self.bestPath = self.population[self.route_length.argmin()]
            self.best_length = self.route_length.min()


def test():
    import matplotlib.pyplot as plt
    cl = CityList()
    cl.readtxt("data/tsp.txt")
    ga = GA(cl, pop_size=100)
    plt.ion()
    plt.scatter(cl.citylist.T[0], cl.citylist.T[1], marker=".", color="gray")
    for i in range(N_GENERATION):
        ga.evolute()
        print(i, ": ", ga.best_length)
        new_pts = cl.citylist[ga.bestPath]
        plt.gca().lines.clear()
        pls = plt.plot(new_pts.T[0], new_pts.T[1], linewidth=0.2, color = 'k'); plt.pause(0.05)
    plt.ioff(); plt.show()

if __name__ == "__main__":
    test()
