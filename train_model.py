import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

FEATURE_DIR = "features"

X = []
y = []

print("🔍 Loading feature files...\n")

for filename in os.listdir(FEATURE_DIR):
    if filename.endswith(".pt"):
        state_name = filename.replace(".pt", "")

        # ❌ Skip empty files
        data = torch.load(os.path.join(FEATURE_DIR, filename))
        if len(data) == 0:
            print(f"⚠️ Skipping {state_name} (empty)")
            continue

        print(f"📦 Loaded {state_name}: {len(data)} samples")

        for feat in data:
            X.append(feat)
            y.append(state_name)

X = np.array(X)
y = np.array(y)

print(f"\n✅ Total Samples: {len(X)}")
print(f"🏷️ Classes: {set(y)}\n")

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
print("🎯 Training model...")
model = SVC(kernel='rbf', C=10, gamma='scale')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Save model + scaler
torch.save(model, "accent_model_svm.pkl")
torch.save(scaler, "feature_scaler.pkl")

print("\n🎉 Model saved as accent_model_svm.pkl")
print("🔧 Scaler saved as feature_scaler.pkl")
