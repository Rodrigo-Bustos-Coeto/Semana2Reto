import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. Load data
# header=0 tells pandas the first row contains titles
columnas = ['Carga', 'Masa', 'TiempoVuelo', 'VelMax', 'PosFinalX', 'Clase']
df = pd.read_csv('datosCelulas.csv', header=0, names=columnas)

# 2. Features and Target
X = df[['Carga', 'Masa', 'TiempoVuelo', 'VelMax', 'PosFinalX']]
y = df['Clase']

# 3. Split data (70% to train, 30% to test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Initialize and Train AdaBoost
# Since it's a 0/1 outcome, 50 estimators is a good starting point
model = AdaBoostClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Results
print("--- Result Analysis ---")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))



df['Clase'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribución de Clases (0 vs 1)')
plt.xlabel('Clase')
plt.ylabel('Cantidad de Celulas')
plt.show()