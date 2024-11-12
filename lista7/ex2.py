import random


class vert:
    def __init__(self, name, valor=None):
        self.valor = valor
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class grafo:
    def __init__(self, list_vert):
        self.lv = list_vert
        self.edge = {vert: [] for vert in self.lv}

    def __str__(self):
        stri = ""
        for vert in self.edge.keys():
            stri += vert.name + ":"
            stri += " " + "".join(str([conex for conex in self.edge[vert]]))
            stri += "\n"
        return stri

    def adjascent(self, x, y):
        if y in self.edge[x]:
            return True
        if x in self.edge[y]:
            return True
        return False

    def neighbors(self, x):
        lista_v = []
        for i in self.edge[x]:
            lista_v.append(i)
        return lista_v

    def add_vertex(self, x):
        if x in self.lv:
            return False
        else:
            self.edge[x] = []

            return True

    def remove_vertex(self, x):
        if x in self.edge:
            del self.edge[x]

            for vert in self.edge.keys():
                if x in self.edge[vert]:
                    self.edge[vert].remove(x)
            return True
        return False

    def add_edge(self, x, y):
        if y in self.edge[x]:
            return False
        else:
            self.edge[x].append(y)
            return True

    def remove_edge(self, x, y):
        if y in self.edge[x]:
            self.edge[x].remove(y)
            return True
        else:
            return False

    @staticmethod
    def get_vertex_value(x):
        return x.valor

    @staticmethod
    def set_vertex_value(x, new_valor):
        x.valor = new_valor


def dist(point1, point2):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    return ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) ** 1 / 2


def minnimun_tree(points):
    Graf = grafo([])
    for i in range(len(points)):
        Graf.add_vertex(vert(f"{i+1}", points[i]))
    print(Graf)
    for verta in Graf.edge.keys():
        print(verta.valor)

    for i in Graf.edge.keys():
        min_vertex = float("inf")
        vertix = None
        for j in Graf.edge.keys():
            if dist(i.valor, j.valor) < min_vertex:
                min_vertex = dist(i.valor, j.valor)
                vertix = j

        Graf.edge[i] = vertix

def menor_ligacao(points):
    pass
def divide_conquis(points):
    points.sort()
    if len(points) <= 3:
        menor_ligacao(points)
    else:
        divide_conquis(points[0:len(points)//2])
        divide_conquis(points[len(points)//2:])
        


points = [[0, 0], [0, 1], [1, 0], [3, 0], [4, 0], [9, 3]]
minnimun_tree(points)
