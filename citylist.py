import numpy as np

class CityList:
    def __init__(self, arr=None):
        if arr is None: return
        self.citylist = np.array(arr, dtype=np.float32)
        self.n_city = self.citylist.shape[0]
        self._distance = self.cal_city_distance_table()

    def get_city_distance(self, city1, city2):
        return self._distance[city1, city2]

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
        self._distance = self.cal_city_distance_table()

    def cal_city_distance_table(self):
        t = np.zeros((self.n_city, self.n_city))
        for i in range(self.n_city):
            for j in range(self.n_city):
                t[i,j] = np.sqrt((self.citylist[i,0] - self.citylist[j,0]) ** 2 +
                                 (self.citylist[i,1] - self.citylist[j,1]) ** 2)
        return t

def test():
    citylist = CityList()
    citylist.readtxt("data/tsp4test.txt")
    print("list: \n", citylist.citylist)
    route = np.arange(citylist.n_city)
    print("n_city: ", citylist.n_city)
    print("route length: ", citylist.get_route_length(route))

if __name__ == "__main__":
    test()  
