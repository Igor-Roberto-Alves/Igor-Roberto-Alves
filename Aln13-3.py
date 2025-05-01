import pandas as pd
import altair as alt
import numpy as np

def poly(x):
    return (x - 2) ** 9

def poly_brute(x):
    return (x**9 - 18 * x**8 + 144 * x**7 - 672 * x**6 + 2016 * x**5 - 4032 * x**4 + 5376 * x**3 - 4608 * x**2 + 2304 * x- 512)

x = np.linspace(1.920, 2.081,300)
y_brute = poly_brute(x)
y = poly(x)

poly_df = pd.DataFrame({"x": x, "y": y_brute})
poly_df2 = pd.DataFrame({"x":x, "y": y})
poly_df3 = pd.DataFrame({"x":x, "error": np.sqrt((y_brute - y)**2)})

df_legenda = pd.DataFrame({
    'label': ['Erro', 'Polinômio comprimido', 'Polinômio explícito'],
    'cor': ['red', 'blue', 'green'],
})

legenda = alt.Chart(df_legenda).mark_square(size = 200).encode(
    y=alt.Y('label:N', axis=alt.Axis(title=None)),
    color=alt.Color('cor:N',scale=None),
    ).properties(width=100, height=100)

a = alt.Chart(poly_df).mark_line(color = "green").encode(alt.X("x",scale= alt.Scale(domain = [1.920,2.081])), y="y").properties(width = 800)
c = alt.Chart(poly_df2).mark_line(color = "blue").encode(alt.X("x",scale= alt.Scale(domain = [1.920,2.081])), y="y").properties(width = 800)
b = alt.Chart(poly_df3).mark_rule(color = "red").encode(alt.X("x",scale= alt.Scale(domain = [1.920,2.081])), y="error").properties(width = 800)
((a + c + b )|legenda).properties(title = "Diferença da avaliação no polinômio explícito e comprimido")
