import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv("countries_dataset_no_label_full_world.csv")

# Generate high_income binary classification
score = (
    (df["gdp_per_capita_usd"] / 1000) +
    df["gdp_growth_percent"] +
    (df["life_expectancy"] / 10) +
    (df["internet_users_percent"] / 10) -
    df["unemployment_percent"] -
    df["inflation_percent"]
)

df["high_income"] = (score >= score.median()).astype(int)


# Split features and label
X = df.drop(["country","high_income"], axis=1)
y = df["high_income"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Build Deep Learning Model
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=8))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

# Evaluate
print(df["high_income"].value_counts())
predictions = (model.predict(X_test) > 0.5).astype(int)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
