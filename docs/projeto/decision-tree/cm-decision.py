import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot(cmap="Blues")
plt.title("Matriz de Confusão - Wine")
# plt.savefig("docs/projeto/images/cm-d-tree.svg", format="svg", transparent=True)
plt.close()

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))