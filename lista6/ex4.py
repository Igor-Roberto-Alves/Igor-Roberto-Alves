from ler_txt import txt_to_matrix


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
    xtotal = 0
    ytotal = 0
    n = len(ilha)
    for cell in ilha:
        xtotal += cell[0]
        ytotal += cell[1]

    return (xtotal / n, ytotal / n)


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
    print("Ilhas mínimas:")
    for m in min:
        print(m, end="")
        print(" -> Centróide =", centroide(m))

    print("\nIlhas máximas:")
    for m in max:
        print(m, end="")
        print(" -> Centróide =", centroide(m))

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
