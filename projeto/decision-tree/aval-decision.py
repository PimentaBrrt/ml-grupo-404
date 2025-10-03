import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import tree

scaler = StandardScaler()

df = pd.read_csv("docs/projeto/wine-final.csv", sep=",", encoding="UTF8")

X = df.drop(columns=["Wine_Type", "cluster"], axis=1)
X = scaler.fit_transform(X)
y = df["Wine_Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

classifier = tree.DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do Modelo: {accuracy:.4f}")

feature_names = df.drop(columns=["Wine_Type", "cluster"], axis=1).columns

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importância": classifier.feature_importances_
})
print("<br>Importância das Features:")
print(feature_importance.sort_values(by="Importância", ascending=False).to_html() + "<br>")