class NPoint:
    def __init__(self, *args):
        self.coords = args
        self.length = len(self.coords)

    def calc_distance(self, point):
        if not isinstance(point, NPoint):
            raise TypeError('Point class required')
        
        if self.length != point.length:
            raise ValueError('Length must be the same')

        s = 0
        for i in range(self.length):
            s += (self.coords[i] - point.coords[i]) ** 2

        return round(s ** .5, 3)
    

if __name__ == '__main__':
    p1 = NPoint(1, 3)
    p2 = NPoint(5, 6)


    assert p1.calc_distance(p2) == 5.0
