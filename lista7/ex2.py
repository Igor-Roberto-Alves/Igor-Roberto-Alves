def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 1 / 2


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


def minnimun_spanning_tree(points):
    matriz = []
    for point in points:
        row = []
        for point2 in points:
            if point2 == point:
                row.append(0)
            else:
                row.append(dist(point, point2))

        # Definindo a matriz com o peso referente a distância euclidiana
        matriz.append(row)

    num_vert = len(matriz)
    minimal_matriz = [[0 for _ in range(num_vert)] for _ in range(num_vert)]
    priorityqueue = []
    visiteds = set()
    # Começaremos com o vértice 0, assim adicionado ele nos visitados
    visiteds.add(0)
    while len(visiteds) < num_vert:  # Enquanto não estão todos ligados
        for i in range(num_vert):
            for j in range(num_vert):
                if i != j and j not in visiteds:  # Se j ainda não foi incluído no grafo
                    priorityqueue.append(
                        (i, j, matriz[i][j])
                    )  # Adicionamos ele na fila de prioridade
            if priorityqueue:
                priorityqueue.sort(
                    key=lambda x: x[2]
                )  # Ordenando a fila de prioridade pela distância de i <-> j
                position1, position2, value = priorityqueue.pop(0)
                minimal_matriz[position1][position2] = value
                visiteds.add(
                    position2
                )  # Como foi adicionado uma aresta para j, precisamos contar j como visitado

    return grafo_matriz(minimal_matriz)


point = ((1, 0), (0, 1))

print(minnimun_spanning_tree(point))
