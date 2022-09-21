from citylist import CityList
from parameters import *
import numpy as np

class BaseModel:
    def __init__(self, citylist, pop_size=POPULATION_SIZE):
        self.city = citylist
        self.pop_size = pop_size
        self.n_city = self.city.n_city
        self.population = None
        self.route_length = np.zeros(self.pop_size)
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

    def get_route_length(self):
        for i in range(self.pop_size):
            self.route_length[i] = self.city.get_route_length(self.population[i])
