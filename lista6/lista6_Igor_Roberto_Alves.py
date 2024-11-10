# Lista 6 -> Igor Roberto Alves -> Prog 2

# QUESTÂO 1 ==================================================================================================================

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


def bfs(grafo, vert_ini):
    percorridos = []
    fila = [vert_ini]  # Começa o bfs com o vértice inicial passado
    while fila:
        vert_atual = fila.pop(0)
        if vert_atual not in percorridos:
            percorridos.append(vert_atual)
            for i in grafo.neighbors(vert_atual):
                if i not in percorridos:
                    fila.append(i)
    return percorridos


if __name__ == "__main__":
    A = vert("A", 1)
    B = vert("B", 1)
    C = vert("C", 0)
    D = vert("D", 0)
    E = vert("E", 1)
    F = vert("F", 1)
    grafo1 = grafo([A, B, C, D, E, F])
    grafo1.add_edge(A, B)
    grafo1.add_edge(A, D)
    grafo1.add_edge(B, C)
    grafo1.add_edge(C, A)
    grafo1.add_edge(C, E)
    grafo1.add_edge(D, F)
    print(grafo1)
    print(
        "Busca Em Largura partindo do vértice A:\n ", bfs(grafo1, A)
    )  # Esperado [A, B, D, C, F, E]

# QUESTÃO 2 ==================================================================================================================

# Utilizando a mesma implementação de grafo da questão 1


def busca_propriedade(grafo: grafo, value) -> vert:
    # Retornaremos todos os vértices com o valor desejado
    lista_vert = grafo.edge.keys()
    visitados = []
    tops = []
    for vert in lista_vert:
        if vert not in visitados:
            fila = [vert]
            while fila:
                vert_atual = fila.pop(0)
                if vert_atual.valor == value:
                    tops.append(vert_atual)
                if vert_atual not in visitados:
                    visitados.append(vert_atual)
                    for i in grafo.neighbors(vert_atual):
                        if i not in visitados:
                            fila.append(i)
    return tops


A = vert("A", 1)
B = vert("B", 1)
C = vert("C", 0)
D = vert("D", 0)
E = vert("E", 1)
F = vert("F", 1)
grafo1 = grafo([A, B, C, D, E, F])
grafo1.add_edge(A, B)
grafo1.add_edge(B, C)
grafo1.add_edge(C, A)
grafo1.add_edge(C, E)

print(
    "\nVértices que tem valor 1:\n", busca_propriedade(grafo1, 1)
)  # Esperado [A, B, E, F]

# QUESTÃO 3 ==================================================================================================================

import random


# Funções para gerar o mapa
def generateMap(m: int, n: int, ground_water_ration=0.2, water="0", ground="1"):
    r = int(m * n * ground_water_ration + 0.5)
    newMap = [[water] * m for _ in range(n)]
    coord = [(i, j) for i in range(n) for j in range(m)]
    random.shuffle(coord)
    for i, j in coord[:r]:
        newMap[i][j] = ground
    return newMap


def save_map(map_s, path="new_map.txt"):
    with open(path, "wt") as f:
        for row in map_s:
            f.write("".join(map(str, row)))
            f.write("\n")


def print_map(map_p):
    for row in map_p:
        print("".join(map(str, row)))


# Função que pega a matriz do txt e passa para matriz tipo lista de listas
def txt_to_matrix(path):
    with open(path) as f:
        linhas = f.read().splitlines()
    matrix = []
    for linha in linhas:
        a = []
        for char in linha:
            a.append(char)
        matrix.append(a)
    return matrix


def number_of_islands(matrix):
    def neighbors(index):
        # Contaremos também as diagonais nas direções
        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
        neighbors = []
        for dx, dy in directions:
            x, y = index[0] + dx, index[1] + dy
            if 0 <= x < n and 0 <= y < m:
                neighbors.append((x, y))
        return neighbors

    n = len(matrix)
    m = len(matrix[0])
    visitados = set()
    count = 0

    for i in range(n):
        for j in range(m):
            if matrix[i][j] == "1" and (i, j) not in visitados:
                count += 1
                fila = [(i, j)]
                while fila:
                    atual = fila.pop(0)
                    visitados.add(atual)
                    for vizinho in neighbors(atual):
                        if (
                            matrix[vizinho[0]][vizinho[1]] == "1"
                            and vizinho not in visitados
                        ):
                            fila.append(vizinho)

    return count


if __name__ == "__main__":
    random.seed(10012)
    m1 = generateMap(50, 10)
    save_map(m1, "test_map.txt")
    matriz = txt_to_matrix("test_map.txt")
    matriz2 = [
        ["1", "0", "1", "0", "0"],
        ["1", "0", "0", "1", "1"],
        ["0", "0", "0", "0", "0"],
        ["1", "1", "0", "1", "1"],
        ["1", "1", "0", "1", "1"],
    ]
    print("\nNúmero de ilhas da matriz gerada:\n", number_of_islands(matriz))
    print("\nNúmero de ilhas da matriz representada:\n", number_of_islands(matriz2))

# QUESTÃO 4 ==================================================================================================================


# Função que encontra as Ilhas máximas e mínimas,
# Se existir mais de uma Ilha máxima ou mínima retorna uma lista com todas elas
def island_min_max(matrix):
    def neighbors(index):
        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
        neighbors = []
        for dx, dy in directions:
            x, y = index[0] + dx, index[1] + dy
            if 0 <= x < n and 0 <= y < m:
                neighbors.append((x, y))
        return neighbors

    n = len(matrix)
    m = len(matrix[0])
    visitados = set()
    count = 0
    ilhas = []
    for i in range(n):
        for j in range(m):
            if matrix[i][j] == "1" and (i, j) not in visitados:
                count += 1
                fila = [(i, j)]
                ilha = set()
                ilha.add((i, j))
                while fila:
                    atual = fila.pop(0)
                    if atual not in visitados:
                        visitados.add(atual)
                        for vizinho in neighbors(atual):
                            if (
                                matrix[vizinho[0]][vizinho[1]] == "1"
                                and vizinho not in visitados
                            ):
                                ilha.add((vizinho[0], vizinho[1]))
                                fila.append(vizinho)

                lista_ilha = [v for v in ilha]
                ilhas.append(lista_ilha)

    ilha_min = [ilhas[0]]
    ilha_max = [ilhas[0]]
    for ilha in ilhas[1:]:
        if len(ilha) > len(ilha_max[0]):
            ilha_max = [ilha]
        elif len(ilha) == len(ilha_max[0]):
            ilha_max.append(ilha)
        elif len(ilha) < len(ilha_min[0]):
            ilha_min = [ilha]
        elif len(ilha) == len(ilha_min[0]):
            ilha_min.append(ilha)

    return ilha_min, ilha_max


def centroide(ilha):
    # As coordenadas x e y do centróide serão dadas pelo (xtotal, ytotal)/(tamanho da ilha)
    xtotal = 0
    ytotal = 0
    n = len(ilha)
    for cell in ilha:
        xtotal += cell[0]
        ytotal += cell[1]

    return (xtotal / n, ytotal / n)


def centr_min_max(matriz):
    ilhas_min, ilhas_max = island_min_max(matriz)
    centroides_min = []
    centroides_max = []
    for min in ilhas_min:
        centroides_min.append(centroide(min))
    for max in ilhas_max:
        centroides_max.append(centroide(max))

    return centroides_min, centroides_max


if __name__ == "__main__":
    # Testando com a matriz
    matriz = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "1"],
        ["1", "1", "0", "1", "1"],
        ["0", "0", "0", "1", "0"],
        ["1", "0", "1", "1", "0"],
    ]
    min, max = island_min_max(matriz)
    centroides_min, centroides_max = centr_min_max(matriz)
    print("Ilhas mínimas:")
    for m in min:
        print(m, end="")
        print(" -> Centróide =", centroide(m))
    # Teste para a função que busca os centróides direto
    print("Centróides mínimos: -> ", centroides_min)

    print("\nIlhas máximas:")
    for m in max:
        print(m, end="")
        print(" -> Centróide =", centroide(m))

    print("Centróides máximos: -> ", centroides_max)


# Teste para a matriz grande
"""    
    matriz = txt_to_matrix("test_map.txt")
    min, max = island_min_max(matriz)
    print("\nIlhas mínimas:")
    for m in min:
        print(m, end="")
        print(" -> Centróide =", centroide(m))

    print("\nIlhas máximas:")
    for m in max:
        print(m, end="")
        print(" -> Centróide =", centroide(m))
"""


# QUESTÃO 5 ==================================================================================================================


def find_lake(matrix):
    def neighbors(index):
        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ]  # Apenas 4 direções: cima, baixo, esquerda e direita
        # Isto porque já consideramos que as ilhas são ligadas pelas diagonais,
        # Deste jeito as águas não se ligam pela diagonal
        neighbors = []
        for dx, dy in directions:
            x, y = index[0] + dx, index[1] + dy
            if 0 <= x < n and 0 <= y < m:
                neighbors.append((x, y))
        return neighbors

    n = len(matrix)
    m = len(matrix[0])
    visitados = set()
    water = []

    for i in range(n):
        for j in range(m):
            if matrix[i][j] == "0" and (i, j) not in visitados:
                fila = [(i, j)]
                ilha = set()  # Serão as ilhas de água
                ilha.add((i, j))
                visitados.add((i, j))

                while fila:
                    atual = fila.pop(0)
                    for vizinho in neighbors(atual):
                        if (
                            matrix[vizinho[0]][vizinho[1]] == "0"
                            and vizinho not in visitados
                        ):
                            visitados.add(vizinho)
                            ilha.add(vizinho)
                            fila.append(vizinho)

                water.append(list(ilha))  # ilha é um conjunto
    lakes = []
    for conjunto in water:  # Para cada conjunto de água veremos se ele é um lago
        is_border_touching = False  # Será um lago se não tiver "0"s que tocam as bordas
        for zero in conjunto:
            # Verifica se o elemento zero está na borda
            if zero[0] == 0 or zero[0] == n - 1 or zero[1] == 0 or zero[1] == m - 1:
                is_border_touching = True
                break  # Interrompe o loop interno se encontrar um zero na borda

        # Apenas adiciona o conjunto se ele não tocar as bordas
        if not is_border_touching:
            lakes.append(conjunto)

    return lakes
    # Water carregará apenas os lagos que a matriz tiver, então se estiver vazia não há lagos
    # Se a lista estiver vazia a matriz não contém nenhum lago


if __name__ == "__main__":
    # Matriz que contém lagos
    matrix = [
        ["1", "1", "0", "1", "1"],
        ["1", "1", "1", "0", "1"],
        ["1", "1", "1", "1", "1"],
        ["0", "0", "1", "0", "1"],
        ["1", "1", "1", "1", "1"],
        ["1", "0", "0", "0", "0"],
    ]

    # Matriz que não contém lagos
    matrix2 = [
        ["1", "1", "0", "1", "1"],
        ["1", "1", "1", "1", "1"],
        ["1", "1", "1", "1", "1"],
        ["0", "0", "1", "1", "1"],
        ["1", "1", "1", "1", "1"],
        ["1", "0", "0", "0", "0"],
    ]

    matriz_grande = txt_to_matrix("test_map.txt")

    # Chamando a função para encontrar os lagos
    lakes = find_lake(matrix)
    lakes2 = find_lake(matrix2)
    lakes_grande = find_lake(matriz_grande)
    if lakes:
        print("\nA matriz 1 contém lagos, são eles:\n", lakes)
    else:
        print("\nA matriz 1 não contem lagos :(")

    if lakes2:
        print("\nA matriz 2 contém lagos, são eles:\n", lakes)
    else:
        print("\nA matriz 2 não contem lagos :(")

    if lakes_grande:
        print("\nA matriz grande contém lagos, são eles:\n", lakes_grande)
    else:
        print("\nA matriz grande não contem lagos :(")
