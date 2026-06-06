import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from src.utils.utils import save_object

# --- Dummy training data ---
data = pd.DataFrame({
    "study_hours": [2, 4, 6, 8, 10],
    "sleep_hours": [8, 7, 6, 5, 5],
    "attendance": [60, 70, 80, 90, 95],
    "score": [50, 60, 70, 80, 90]
})

X = data.drop("score", axis=1)
y = data["score"]

# --- Full pipeline: scaler + model ---
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

pipeline.fit(X, y)

# --- Save single pipeline ---
save_object("artifacts/pipeline.pkl", pipeline)

print("Training complete. Saved pipeline.pkl!")
