import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree

scaler = StandardScaler()

df = pd.read_csv("docs/projeto/wine-final.csv", sep=",", encoding="UTF8")

X = df.drop(columns=["Wine_Type", "cluster"], axis=1)
X = scaler.fit_transform(X)
y = df["Wine_Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

classifier = tree.DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

plt.figure(figsize=(12,9))
tree.plot_tree(
    classifier,
    feature_names=pd.DataFrame(X).columns,
    class_names=classifier.classes_,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=10
)
plt.title("Árvore de Decisão - Faixas de Álcool (Vinhos)")

# plt.savefig("docs/projeto/images/d-tree.svg", format="svg", transparent=True)
plt.close()