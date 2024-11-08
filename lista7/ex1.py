# Grafo por lista de adjascência
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


def find_judge(n, t):
    Graf = grafo([])
    lista_v = []
    for i in range(n):
        vertex = vert(f"{i+1}")
        Graf.add_vertex(vertex)
        lista_v.append(vertex)
    for lig in t:
        Graf.add_edge(lista_v[lig[0] - 1], lista_v[lig[1] - 1])
    print(Graf)

    possible_judge = []
    for i in lista_v:
        if not Graf.neighbors(i):
            possible_judge.append(i)

    for i in possible_judge:
        judge = True
        for j in lista_v:
            if j != i:
                if i not in Graf.neighbors(j):
                    judge = False
        if judge:
            return i

    return -1


lig = [[1, 3], [2, 1], [2, 3]]
print("O juiz é a pessoas: ", find_judge(3, lig))
