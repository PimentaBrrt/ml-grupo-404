import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

le = LabelEncoder()
scaler = StandardScaler()

df = pd.read_csv("docs/projeto/wine-final.csv", sep=",", encoding="UTF8")

X = df.drop(columns=["Wine_Type", "cluster"], axis=1)
y = le.fit_transform(df["Wine_Type"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel="rbf", C=1, gamma="scale")

svm_model.fit(X_train_scaled, y_train)
y_pred = svm_model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print("Acurácia do SVM:", acc)

print("\nRelatório de Classificação:\n")
print(classification_report(y_test, y_pred))