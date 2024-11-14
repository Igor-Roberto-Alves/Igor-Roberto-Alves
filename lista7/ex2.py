import copy


def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)


class vert:
    def __init__(self, name, valor=None):
        self.valor = valor
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class edge:
    def __init__(self, p1, p2, valor):
        self.p1 = p1
        self.p2 = p2
        self.valor = valor

    def __repr__(self):
        return f"-> {self.p2}: {self.valor}"


# Escolhi aqui definir um grafo por matriz, achei mais simples para tratar arestas com peso
class grafo_matriz:
    def __init__(self, matriz):
        self.dict = {}
        self.list_vert = []
        for i in range(len(matriz)):
            vertex = vert(f"{i}")
            self.list_vert.append(vertex)
            self.dict[vertex] = []

        # A matriz é sempre quadrada
        for i in range(len(matriz)):
            for j in range(len(matriz)):
                if i != j:
                    if matriz[i][j] != 0:
                        self.dict[self.list_vert[i]].append(edge(i, j, matriz[i][j]))

    def __str__(self):
        stri = ""
        for vert in self.dict.keys():
            stri += vert.name + ":"
            stri += " " + "".join(str([conex for conex in self.dict[vert]]))
            stri += "\n"
        return stri


def merge_atualxpriority(atual, priority):
    if priority:
        lista = copy.deepcopy(priority)
        for cell in lista:
            j = cell[1]
            for cell2 in atual:
                if cell2[1] == j and cell2[2] < cell[2]:
                    cell[0], cell[2] = cell2[0], cell2[2]

        return lista
    else:
        return atual


def minnimun_spanning_tree(points):
    matriz = []
    for point in points:
        row = []
        for point2 in points:
            if point2 == point:
                row.append(0)
            else:
                row.append(dist(point, point2))
        matriz.append(row)
        print(row)

    num_vert = len(matriz)
    minimal_matriz = [[0 for _ in range(num_vert)] for _ in range(num_vert)]
    priorityqueue = []
    visiteds = set()
    visiteds.add(0)

    for j in range(1, num_vert):
        priorityqueue.append(
            [0, j, matriz[0][j]]
        )  # Todas as ligacoes com o vértice 0 (o vértice inicial)

    while len(visiteds) < num_vert:  # Enquanto nem todos foram interligados
        priorityqueue.sort(key=lambda x: x[2])  # Ordenando pelo tamanho da aresta
        p1, p2, size = priorityqueue.pop(0)
        if p2 not in visiteds:
            minimal_matriz[p1][p2] = size
            minimal_matriz[p2][p1] = size
            visiteds.add(p2)
            for j in range(num_vert):
                if j not in visiteds:
                    priorityqueue.append([p2, j, matriz[p2][j]])

    return grafo_matriz(minimal_matriz)


point = ((1, 0), (0, 1), (10, 1), (3, 4))
print(minnimun_spanning_tree(point))

point = ((1, 0), (0, 1), (10, 1), (3, 4))

print(minnimun_spanning_tree(point))
