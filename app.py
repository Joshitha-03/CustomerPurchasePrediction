import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import gradio as gr

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("shoppers.csv")

# Preprocessing
df = df.dropna()
le = LabelEncoder()
df['Month'] = le.fit_transform(df['Month'])
df['VisitorType'] = le.fit_transform(df['VisitorType'])
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train & Compare Models
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

best_model = None
best_acc = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_model = model

print(f"\n✅ Best Model: {best_model.__class__.__name__} with Accuracy: {best_acc:.4f}")

# Save model & scaler
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# -----------------------------
# Gradio UI
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_purchase(Administrative, Administrative_Duration, Informational, Informational_Duration, 
                     ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, 
                     SpecialDay, Month, OperatingSystems, Browser, Region, TrafficType, 
                     VisitorType, Weekend):
    
    data = [[Administrative, Administrative_Duration, Informational, Informational_Duration,
             ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues,
             SpecialDay, Month, OperatingSystems, Browser, Region, TrafficType,
             VisitorType, Weekend]]
    
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    return "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase"


inputs = [
    gr.Number(label="Administrative"),
    gr.Number(label="Administrative_Duration"),
    gr.Number(label="Informational"),
    gr.Number(label="Informational_Duration"),
    gr.Number(label="ProductRelated"),
    gr.Number(label="ProductRelated_Duration"),
    gr.Number(label="BounceRates"),
    gr.Number(label="ExitRates"),
    gr.Number(label="PageValues"),
    gr.Number(label="SpecialDay"),
    gr.Number(label="Month (0-11)"),
    gr.Number(label="OperatingSystems"),
    gr.Number(label="Browser"),
    gr.Number(label="Region"),
    gr.Number(label="TrafficType"),
    gr.Number(label="VisitorType (0=New,1=Returning)"),
    gr.Number(label="Weekend (0=No,1=Yes)")
]


outputs = gr.Textbox(label="Prediction")

app = gr.Interface(fn=predict_purchase, inputs=inputs, outputs=outputs, title="🛒 Customer Purchase Prediction", description="Predict if a customer will purchase after browsing.")

app.launch()
