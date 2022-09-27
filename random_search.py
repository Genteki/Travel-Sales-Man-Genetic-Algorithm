from citylist import CityList
from base_model import BaseModel
from parameters import *
import numpy as np

class RandomSearch(BaseModel):
    def __init__(self, citylist, pop_size=POPULATION_SIZE):
        super(RandomSearch, self).__init__(citylist)
        self.first_generation()
        self.best_length = self.route_length.min()

    def first_generation(self):
        self.random_route()
        self.get_route_length()

    def evo(self):
        self.random_route()
        self.get_route_length()
        if self.route_length.min() < self.best_length:
            self.best_length = self.route_length.min()

def test_2(path):
    cl = CityList()
    cl.readtxt("data/"+path)
    rs = RandomSearch(cl, pop_size=POPULATION_SIZE)
    file = open("output/rs_"+path, "w")
    for i in range(N_GENERATION):
        rs.evo()
        file.write(str(i)+","+str(rs.best_length)+"\n")
        print(str(i)+","+str(rs.best_length)+"\n")
    file.close()

if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    test_2(filename)
