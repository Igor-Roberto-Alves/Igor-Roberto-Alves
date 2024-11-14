# Questão 1 ========================================================================================


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
    # Inicializando um grafo com n vértices e fazendo as devidas ligações
    Graf = grafo([])
    lista_v = []
    for i in range(n):
        vertex = vert(f"{i+1}")
        Graf.add_vertex(vertex)
        lista_v.append(vertex)
    for lig in t:
        Graf.add_edge(lista_v[lig[0] - 1], lista_v[lig[1] - 1])
    print(Graf)

    # Definiremos os possíveis juízes vendo quais pessoas não confiam em ninguém
    possible_judge = []
    for i in lista_v:
        if not Graf.neighbors(i):
            possible_judge.append(i)

    # Definido os possíveis juízes, podemos agora observar em qual deles a população inteira confia
    for i in possible_judge:
        judge = True
        for j in lista_v:
            if j != i:
                if i not in Graf.neighbors(j):
                    judge = False
        if judge:  # Caso existir esse juíz ele será retornado
            return i
    # Caso não, retorna -1
    return -1


lig = [[1, 3], [2, 1], [2, 3]]
print("O juiz é a pessoa: ", find_judge(3, lig))

# Questão 2 =======================================================================================

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


# Utilizei o algoritmo de Prim!
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

# Questão 3 ================================================================================================
import copy

x_fx = ([15, 200], [9, 400], [5, 600], [3, 800], [-2, 1000], [-5, 1200], [-15, 1400])


# Utilizaremos a interpolação por polinômio de lagrange
def lagrange(
    x_fx, alvo, invert=False
):  # Carregaremos invert, que caso for True => que queremos a relação altura x grau
    sum = 0

    x_fx = copy.deepcopy(x_fx)
    if invert:
        for x in x_fx:
            x[0], x[1] = x[1], x[0]

    for x in x_fx:
        a = 1
        for x2 in x_fx:
            if x != x2:
                a *= (alvo - x2[0]) / (x[0] - x2[0])

        sum += a * x[1]

    return sum


print("A altura em que o avião alcança 0 graus Celsius é: ", lagrange(x_fx, 0))
print(
    "\nA temperatura que provavelmente o avião estava a 700 metros é: ",
    lagrange(x_fx, 700, True),
)

# Questão 4 ===============================================================================================
# python 3

from typing import Any
import numpy as np
import matplotlib.pyplot as plt


class Domain:
    min = None
    max = None

    def __contains__(self, x):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        return self.__repr__()

    def copy(self):
        raise NotImplementedError


class Interval(Domain):
    def __init__(self, p1, p2):
        self.inff, self.supp = min(p1, p2), max(p1, p2)

    @property
    def min(self):
        return self.inff

    @property
    def max(self):
        return self.supp

    @property
    def size(self):
        return self.max - self.min

    @property
    def haf(self):
        return (self.max + self.min) / 2.0

    def __contains__(self, x):
        return np.all(np.logical_and(self.inff <= x, x <= self.supp))

    def __str__(self):
        return f"[{self.inff:2.4f}, {self.supp:2.4f}]"

    def __repr__(self):
        return f"[{self.inf!r:2.4f}, {self.supp!r:2.4f}]"

    def copy(self):
        return Interval(self.inff, self.supp)


class RealFunction:
    f = None
    prime = None
    domain = None

    def eval_safe(self, x):
        if self.domain is None or x in self.domain:
            return self.f(x)
        else:
            raise Exception("The number is out of the domain")

    def prime_safe(self, x):
        if self.domain is None or x in self.domain:
            return self.prime(x)
        else:
            raise Exception("The number is out of the domain")

    def __call__(self, x) -> float:
        return self.eval_safe(x)

    def plot(self):
        fig, ax = plt.subplots()
        X = np.linspace(self.domain.min, self.domain.max, 100)
        Y = self(X)
        ax.plot(X, Y)
        return fig, ax


def bissect(
    f: RealFunction,
    search_space: Interval,
    erroTol: float = 1e-4,
    maxItr: int = 1e4,
    eps: float = 1e-6,
) -> Interval:
    count = 0
    ss = search_space.copy()
    err = ss.size / 2.0
    fa, fb = f(ss.min), f(ss.max)
    if fa * fb > -eps:
        if abs(fa) < eps:
            return Interval(ss.min, ss.min)
        elif abs(fb) < eps:
            return Interval(ss.max, ss.max)
        else:
            raise Exception(
                "The interval extremes share the same signal;\n employ the grid search method to locate a valid interval."
            )
    while count <= maxItr and err > erroTol:
        count += 1
        a, b, m = ss.min, ss.max, ss.haf
        fa, fb, fm = f(a), f(b), f(m)
        if abs(fm) < eps:
            return Interval(m, m)
        elif fa * fm < -eps:
            ss = Interval(a, m)
        elif fb * fm < -eps:
            ss = Interval(m, b)
    return ss


def grid_search(f: RealFunction, domain: Interval = None, grid_freq=8) -> Interval:
    if domain is not None:
        D = domain.copy()
    else:
        D = f.domain.copy()
    L1 = np.linspace(D.min, D.max, grid_freq)
    FL1 = f(L1)
    TI = FL1[:-1] * FL1[1:]
    VI = TI <= 0
    if not np.any(VI):
        return None
    idx = np.argmax(VI)
    return Interval(L1[idx], L1[idx + 1])


def newton_root(
    fx, p=None, erro=1 * 10 ** (-4)
):  # A tolerância para nossa raiz será de 1*10**(-4)
    if p == None:
        p = bissect(fx, grid_search(fx, fx.domain, 100)).haf
    new_p = p - fx.eval_safe(p) / fx.prime_safe(p)  # Expansão do polinômio de taylor
    if abs(fx(new_p)) <= erro:
        return new_p
    else:
        newton_root(
            fx, new_p
        )  # Recursivamente chama newton_root com o ponto definido ao cruzar o eixo x


if __name__ == "__main__":

    class funcTest(RealFunction):
        f = lambda self, x: np.power(x, 4) - 100
        prime = lambda self, x: 4 * x**3
        domain = Interval(-100, 100)

    ft = funcTest()
    # print(ft.eval_safe(0))
    print(newton_root(ft))

# Questão 5 ==================================================================================================

import numpy as np
import scipy.interpolate as scp
import time


class interpolater:

    def evaluate(self, X):
        raise NotImplementedError

    def __call__(self, X):
        return self.evaluate(X)


# Passamos 2 arrays, sendo o primeiro representando x e o segundo representando y
class Lagrange(interpolater):
    def __init__(self, array_x, array_y):
        self.polinom = scp.lagrange(array_x, array_y)

    # Evaluate retorna o polinômio aplicado ao x passasdo
    def evaluate(self, x):
        return self.polinom(x)


class VandermondeMatrix(interpolater):
    def __init__(self, x, y):
        if len(x) != len(y):
            raise RuntimeError(
                f"Dimensions must be equal len(x) = {len(x)} != len(y) = {len(y)}"
            )
        self.data = [x, y]
        self._degree = len(x) - 1
        self._buildMatrix()
        self._poly = np.linalg.solve(self.matrix, self.data[1])

    def _buildMatrix(self):
        self.matrix = np.ones([self._degree + 1, self._degree + 1])
        for i, x in enumerate(self.data[0]):
            self.matrix[i, 1:] = np.multiply.accumulate(np.repeat(x, self._degree))

    def evaluate(self, X):
        r = 0.0
        for c in self._poly[::-1]:
            r = c + r * X
        return r


def random_sample(intv, N):
    r = np.random.uniform(intv[0], intv[1], N - 2)

    return np.array([intv[0]] + list(r) + [intv[1]])


def error_pol(f, P, intv, n=1000):
    x = random_sample(intv, n)
    vectError = np.abs(f(x) - P(x))
    return np.sum(vectError) / n, np.max(vectError)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    DataX = [10.7, 11.075, 11.45, 11.825, 12.2, 12.5]
    DataY = [-0.25991903, 0.04625002, 0.16592075, 0.13048074, 0.13902777, 0.2]

    t1 = time.time()
    Pl = Lagrange(DataX, DataY)
    print("Polinômio de Lagrange:\n ", Pl.polinom)
    t2 = time.time()
    print("tempo para formar o polinômio de lagrange: ", t2 - t1)
    t1 = time.time()
    Pvm = VandermondeMatrix(DataX, DataY)
    t2 = time.time()
    print("tempo para formar o polinômio de Vandermonte: ", t2 - t1)
