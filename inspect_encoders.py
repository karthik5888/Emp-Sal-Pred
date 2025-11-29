import joblib

encoders = joblib.load("encoders.pkl")
for col, le in encoders.items():
    print(f"{col}: {list(le.classes_)}")
