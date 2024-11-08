def number_of_islands(matrix):
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

    for i in range(n):
        for j in range(m):
            if matrix[i][j] == "1" and (i, j) not in visitados:
                count += 1
                fila = [(i, j)]
                while fila:
                    atual = fila.pop(0)
                    if atual not in visitados:
                        visitados.add(atual)
                        for vizinho in neighbors(atual):
                            if (
                                matrix[vizinho[0]][vizinho[1]] == "1"
                                and vizinho not in visitados
                            ):
                                fila.append(vizinho)

    return count


import random


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
    print("Número de ilhas da matriz gerada:\n", number_of_islands(matriz))
    print("Número de ilhas da matriz representada:\n", number_of_islands(matriz2))
