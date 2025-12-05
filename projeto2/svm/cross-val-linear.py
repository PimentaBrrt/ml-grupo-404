import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score

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

svm_model = SVC(kernel="linear", C=1, gamma="scale")

svm_model.fit(X_train_scaled, y_train)
y_pred = svm_model.predict(X_test_scaled)

train_accuracy = svm_model.score(X_train_scaled, y_train)
test_accuracy = svm_model.score(X_test_scaled, y_test)
print(f"\n<b>Acurácias dos conjuntos - SVM linear</b>\n")
print(f"Acurácia no Treino: {train_accuracy:.4f} \n")
print(f"Acurácia no Teste: {test_accuracy:.4f}")

cv_scores = cross_val_score(svm_model, X, y, cv=5)
print(f"\n<b>Validação Cruzada (5-fold) -</b>\n")
print(f"Scores: {cv_scores}\n")
print(f"Média: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")