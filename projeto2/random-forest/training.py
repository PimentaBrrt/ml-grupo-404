import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

rf = RandomForestClassifier(n_estimators=100,
                            max_depth=5,
                            max_features='sqrt', 
                            random_state=42)

rf.fit(X_train_scaled, y_train)
predictions = rf.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, predictions)}")

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importância": rf.feature_importances_
})

print("<br>Importância das Features:")
print(feature_importance.sort_values(by="Importância", ascending=False).to_html() + "<br>")