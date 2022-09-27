from citylist import *
from parameters import *
from base_model import BaseModel
import numpy as np
import sys

class GA(BaseModel):
    def __init__(self, city_list, pop_size=POPULATION_SIZE,
                 cross_rate=CROSS_RATE, mutate_rate=MUTATE_RATE):
        super(GA, self).__init__(city_list)
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.fitness = self.fitness_exp()
        self.selected_pop = None
        self.best_length = 10000
        self.best_path = None
        self.get_route_length()
        self.find_best()

    def fitness_exp(self):
        return np.exp(self.pop_size ** 2 / 20 / self.route_length)
        #return np.exp(self.pop_size*2/self.route_length)

    def select_parent(self):
        self.fitness = self.fitness_exp()
        p = self.fitness / self.fitness.sum()
        selected_idx = np.random.choice(np.arange(self.pop_size),
                                        size=self.pop_size, replace=True, p=p)
        self.selected_pop = self.population[selected_idx].copy()

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

    def crossover_tsp1000(self, p1, p2):
        cross_points = np.random.randint(0, self.n_city, 2)
        dad_segment = np.array(p2[cross_points.min():cross_points.max()+1])
        mon_segment = np.array([bit for bit in p1 if bit not in dad_segment])
        offspring = np.concatenate((dad_segment, mon_segment))
        return offspring

    def mutate(self, gene):
        for pointA in range(self.n_city):
            if self.city.get_city_distance(pointA-1, pointA) > 5*np.pi/self.n_city:
                rate = self.mutate_rate * 5
            else:
                rate = self.mutate_rate
            if np.random.rand() < rate:
                pointB = np.random.randint(self.n_city)
                cityA = gene[pointA]
                gene[pointA] = gene[pointB]
                gene[pointB] = cityA
        return gene

    def evolute(self):
        new_pop = np.zeros_like(self.population)
        self.select_parent()
        for i in range(self.pop_size):
            offspring = self.population[i].copy()
            if np.random.rand() < self.cross_rate:
                j = np.random.randint(0, self.pop_size, size=1)
                offspring = self.crossover_tsp1000(offspring, self.selected_pop[j].reshape(self.n_city))
            offspring = self.mutate(offspring)
            new_pop[i] = offspring
        self.population = new_pop
        self.get_route_length()
        self.find_best()

    def find_best(self):
        if self.route_length.min() < self.best_length:
            self.bestPath = self.population[self.route_length.argmin()]
            self.best_length = self.route_length.min()


def test(path="data/tsp.txt"):
    import matplotlib.pyplot as plt
    cl = CityList()
    cl.readtxt(path)
    ga = GA(cl, pop_size=POPULATION_SIZE)
    last_best = ga.best_length
    plt.ion()
    plt.gcf().set_size_inches(8,6)
    plt.axis('equal')
    plt.scatter(cl.citylist.T[0], cl.citylist.T[1], marker=".", color="k")
    text_length = plt.text(0.9, 0.9, "length: {}".format(last_best.round(4)))
    title = plt.title("Shortest Path GA, n = {}".format(ga.n_city))
    for i in range(N_GENERATION):
        ga.evolute()
        print(i, ",", ga.best_length)
        if ga.best_length < last_best:
            last_best = ga.best_length
            text_length.set_text("length: {}\ngeneration: {}".format(last_best.round(4), i))
            new_pts = cl.citylist[ga.bestPath]
            new_pts = np.r_[new_pts, [new_pts[0]] ]
            plt.gca().lines.clear()
            plt.plot(new_pts.T[0], new_pts.T[1], linewidth=1, color = 'k'); plt.pause(0.05)
    plt.ioff(); plt.show()

if __name__ == "__main__":
    filename = sys.argv[1]
    test("data/tsp_{}.txt".format(filename))
    #test()
