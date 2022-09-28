from base_model import BaseModel
from citylist import CityList
from parameters import *
import numpy as np

class HillClimber(BaseModel):
    def __init__(self, city_list, pop_size=1):
        super(HillClimber,self).__init__(city_list)
        self.get_route_length()
        self.best_length = self.route_length[0]
        self.best_path = self.population[0]

    def find_next(self, last_path):
        neighbors = []
        neighbors_length = []
        for i in range(self.city.n_city-1):
            for j in range(i+1, self.city.n_city):
                new_neighbor = last_path.copy()
                new_neighbor[i] = last_path[j]
                new_neighbor[j] = last_path[i]
                neighbors.append(new_neighbor)
                neighbors_length.append(self.city.get_route_length(new_neighbor))
        neighbors_length = np.array(neighbors_length, dtype=np.float32)
        best_index = neighbors_length.argmin()
        best_neighbor = neighbors[best_index]
        return np.array(best_neighbor, dtype=np.int16)

    def climb(self):
        self.population[0] = self.find_next(self.population[0])
        self.get_route_length()
        if self.route_length[0] < self.best_length:
            self.best_length = self.route_length[0]
            self.best_path = self.population[0]

def test(path="data/tsp_circle.txt"):
    import matplotlib.pyplot as plt
    cl = CityList()
    cl.readtxt(path)
    hc = HillClimber(cl)
    last_best = hc.best_length
    plt.ion()
    plt.gcf().set_size_inches(8,6)
    plt.axis('equal')
    plt.scatter(cl.citylist.T[0], cl.citylist.T[1], marker=".", color="k")
    text_length = plt.text(0.9, 0.9, "length: {}".format(last_best.round(4)))
    title = plt.title("Shortest Path Hill Climber, n = {}".format(hc.n_city))
    for i in range(N_GENERATION):
        hc.climb()
        if hc.best_length < last_best:
            last_best = hc.best_length
            new_pts = cl.citylist[hc.best_path]
            text_length.set_text("length: {}\ngeneration: {}".format(last_best.round(4), i))
            new_pts = np.r_[new_pts, [new_pts[0]] ]
            plt.gca().lines.clear()
            plt.plot(new_pts.T[0], new_pts.T[1], linewidth=1, color = 'k'); plt.pause(0.05)
    plt.ioff(); plt.show()

def test_2(path="circle"):
    cl = CityList()
    cl.readtxt("data/tsp_"+path+".txt")
    file = open("output/short_hc_"+path, "w")
    for p in range(10):
        hc = HillClimber(cl, pop_size=POPULATION_SIZE)
        for i in range(1500):
            file.write(str(hc.best_length)+", ")
            print(i, ": ", hc.best_length)
            hc.climb()
        file.write("\n")

if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    test_2(filename)
    #test("data/tsp_{}.txt".format(filename))
