import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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

rf = RandomForestClassifier(n_estimators=100,
                            max_depth=5,
                            max_features='sqrt', 
                            random_state=42)

rf.fit(X_train_scaled, y_train)
predictions = rf.predict(X_test_scaled)

cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"\n<b>Validação Cruzada (5-fold) -</b>\n")
print(f"Scores: {cv_scores}\n")
print(f"Média: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")