import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px

st.set_page_config(page_title="Previsão de Heating Load", layout="wide")

st.title("Previsão de Heating Load")

Dados = pd.read_excel("ENB2012_data.xlsx")

st.subheader("Base de Dados")
st.dataframe(Dados, use_container_width=True)

X = Dados.drop(columns=["Y1", "Y2"])
y = Dados["Y1"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

predicoes = modelo.predict(X_test)

mae = mean_absolute_error(y_test, predicoes)
r2 = r2_score(y_test, predicoes)

st.subheader("Avaliação do Modelo")

col1, col2 = st.columns(2)

with col1:
    st.metric("MAE", f"{mae:.2f}")

with col2:
    st.metric("R²", f"{r2:.3f}")

coeficientes = pd.DataFrame({
    "Variável": X.columns,
    "Coeficiente": modelo.coef_
})

fig = px.bar(
    coeficientes,
    x="Variável",
    y="Coeficiente",
    color="Coeficiente",
    title="Influência das Variáveis"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Informe os valores das variáveis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    x1 = st.number_input("X1", value=float(Dados["X1"].mean()))

with col2:
    x2 = st.number_input("X2", value=float(Dados["X2"].mean()))

with col3:
    x3 = st.number_input("X3", value=float(Dados["X3"].mean()))

with col4:
    x4 = st.number_input("X4", value=float(Dados["X4"].mean()))

col5, col6, col7, col8 = st.columns(4)

with col5:
    x5 = st.number_input("X5", value=float(Dados["X5"].mean()))

with col6:
    x6 = st.number_input("X6", value=float(Dados["X6"].mean()))

with col7:
    x7 = st.number_input("X7", value=float(Dados["X7"].mean()))

with col8:
    x8 = st.number_input("X8", value=float(Dados["X8"].mean()))

if st.button("Prever Heating Load"):

    entrada = pd.DataFrame({
        "X1": [x1],
        "X2": [x2],
        "X3": [x3],
        "X4": [x4],
        "X5": [x5],
        "X6": [x6],
        "X7": [x7],
        "X8": [x8]
    })

    previsao = modelo.predict(entrada)[0]

    st.success(f"Heating Load previsto: {previsao:.2f}")

