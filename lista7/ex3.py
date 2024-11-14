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
