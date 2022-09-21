import numpy as np

class CityList:
    def __init__(self, arr=None):
        if arr is None: return
        self.citylist = np.array(arr, dtype=np.float32)
        self.n_city = self.citylist.shape[0]

    def get_city_distance(self, city1, city2):
        n = ((self.citylist[city1,0] - self.citylist[city2,0]) ** 2 +
              (self.citylist[city1,1] - self.citylist[city2,1]) ** 2)
        return np.sqrt(n)

    def get_route_length(self, route):
        l = 0
        for i in range(0, len(route)):
             l += self.get_city_distance(route[i-1], route[i])
        return l

    def readtxt(self, path="data/tsp.txt"):
        with open(path) as ifile:
            lines = ifile.readlines()
            cl = [line.strip().split(",") for line in lines]
        self.citylist = np.array(cl, dtype=np.float32)
        self.n_city = self.citylist.shape[0]

def test():
    citylist = CityList()
    citylist.readtxt("data/tsp4test.txt")
    print("list: \n", citylist.citylist)
    route = np.arange(citylist.n_city)
    print("n_city: ", citylist.n_city)
    print("route length: ", citylist.get_route_length(route))

if __name__ == "__main__":
    test()
