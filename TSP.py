import numpy as np
import matplotlib.pyplot as plt

POPULATION_SIZE = 70
CROSS_RATE = 0.7
MUTATE_RATE = 0.00
N_GENERATION = 500

class City:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]

    @staticmethod
    def get_distance(city1, city2):
        return np.sqrt((city1.x - city2.x)**2 + (city1.y - city2.y)**2)

class CityList:
    def __init__(self, city_list):
        self.city_list = city_list
        self.n_city = len(city_list)

    def get_city_point(self, city):
        return np.array([self.city_list[city].x, self.city_list[city].y],
                         dtype=np.float32)

    def get_city_distance(self, city1, city2):
        return City.get_distance(self.city_list[city1], self.city_list[city2])

    def get_route_length(self, route):
        if self.n_city <= 1:
            raise Exception("Number of city less than 1")
        l2 = 0        # total length
        for i in range(1, self.n_city):
            l2 += self.get_city_distance(route[i], route[i-1])
        return np.sqrt(l2)

'''
def random_route(self):
        self.route = np.random.sample(self.city_list, len(self.city_list))
        return self.route
'''

class GA:
    def __init__(self, city_list, pop_size=POPULATION_SIZE,
                 cross_rate=CROSS_RATE, mutate_rate=MUTATE_RATE):
        self.n_city = city_list.n_city
        self.pop_size = pop_size
        self.city_list = city_list
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.fitness_list = np.zeros(self.pop_size)
        self.population = self.first_generation()
        self.selected_pop = None

    def first_generation(self):
        p = []
        for i in range(self.pop_size):
            new_indvidual = np.arange(self.n_city)
            np.random.shuffle(new_indvidual)
            p.append(new_indvidual)
        return np.array(p, dtype=np.int16)


    def get_fitness(self):
        for i in range(self.pop_size):
            self.fitness_list[i] = 1 / self.city_list.get_route_length(self.population[i])
        self.fitness_list = self.fitness_list - self.fitness_list.min() + 0.1
        return self.fitness_list

    def select(self):
        p = self.fitness_list / self.fitness_list.sum()
        selected_idx = np.random.choice(np.arange(self.pop_size),
                                        size=self.pop_size, replace=True, p=p)
        self.selected_pop = self.population[selected_idx]
        return self.selected_pop

    def crossover(self, dad):
        if np.random.rand() < self.cross_rate:
            mon = self.population[np.random.randint(0, self.pop_size)]        # select the mon id from population
            cross_segment = np.random.randint(0, self.n_city+1, 2)              # select segment to swap
            mon_segment = np.array(mon[cross_segment.min():cross_segment.max()])          # segment from mon
            dad_segment = np.array([gene for gene in dad if gene not in mon_segment])
            child = np.concatenate((dad_segment, mon_segment))
            return child
        return dad

    def mutate(self, route):
        for pointA in range(self.n_city):
            if np.random.rand() < self.mutate_rate:
                pointB = np.random.randint(self.n_city)
                cityA = route[pointA]
                route[pointA] = route[pointB]
                route[pointB] = cityA
        return route

    def select_next_gen(self, new_pop):
        next_gen_fitness = np.zeros(new_pop.shape[0])
        for i in range(new_pop.shape[0]):
            self.city_list.get_route_length(new_pop[i])
        next_gen_idx = next_gen_fitness.argsort()[:self.pop_size]
        return new_pop[next_gen_idx]

    def evolute(self):
        self.get_fitness()
        self.select()
        new_pop = self.population.copy()
        for dad in self.selected_pop:
            child = self.crossover(dad)
            child = self.mutate(child)
            new_pop = np.r_[new_pop, [child]]
        self.population = self.select_next_gen(new_pop.astype(np.int16))

    def get_best(self):
        self.get_fitness()
        i = self.fitness_list.argmax()
        return self.population[i]

    def get_worst(self):
        #self.get_fitness()
        i = self.fitness_list.argmin()
        return self.population[i]

def test():
    #load cities
    city_list = []
    pts = []
    with open("tsp4test.txt") as ifile:
        lines = ifile.readlines()
        for line in lines:
            new_pt = np.array(line.strip().split(","), dtype=np.float32)
            pts.append(new_pt)
            city_list.append(City(new_pt))
    pts = np.array(pts, dtype=np.float32)
    city_list = CityList(city_list)

    ga = GA(city_list)
    plt.ion()
    plt.scatter(pts.T[0], pts.T[1], marker=".", color="gray")
    for g in range(N_GENERATION):
        best = ga.get_best()
        worst = ga.get_worst()
        print("generation: ", g,
              "minimum length: ", city_list.get_route_length(best),
              "maximum length: ", city_list.get_route_length(worst))
        new_pts = []
        for city_idx in best:
            new_pts.append(city_list.get_city_point(city_idx))
        ga.evolute()
        new_pts = np.array(new_pts)
        plt.gca().lines.clear()
        pls = plt.plot(new_pts.T[0], new_pts.T[1], linewidth=0.2, color = 'k'); plt.pause(0.05)
    plt.ioff(); plt.show()


if __name__ == "__main__":
    test()
