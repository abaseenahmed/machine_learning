
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Actual values
y_actual = [1, 1, 1, 0, 0, 0, 1, 0]

# Predicted values
y_predicted = [1, 1, 0, 0, 1, 0, 1, 0]

# Create confusion matrix
cm = confusion_matrix(y_actual, y_predicted)

# Display the matrix
print("Confusion Matrix:")
print(cm)

# Visualize the matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Confusion Matrix")
plt.show()
