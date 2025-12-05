import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

le = LabelEncoder()
scaler = StandardScaler()

df = pd.read_csv("docs/projeto/wine-final.csv", sep=",", encoding="UTF8")

df["Wine_Type"] = le.fit_transform(df["Wine_Type"])

X = df.drop(columns=["Alcohol", "cluster"], axis=1)
y = df["Alcohol"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rlm = LinearRegression()
rlm.fit(X_train_scaled, y_train)

coeficientes = rlm.coef_
variaveis = X.columns

print("\n<b>Coeficientes da regressão:</b>\n")
print(f"Intercepto = {round(rlm.intercept_, 4)}\n")
for i in range(len(coeficientes)):
    print(f"{variaveis[i]} = {round(coeficientes[i], 4)}\n")

y_pred = rlm.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)

print(f"\n<b>R² do modelo no conjunto de teste: {round(r2, 4)}</b>")