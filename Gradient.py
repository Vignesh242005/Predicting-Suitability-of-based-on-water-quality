from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve


# Load data from CSV file
df = pd.read_csv("/content/sample_data/realfishdataset.csv")

# Encode categorical target variable into integers
label_encoder = LabelEncoder()
df['fish_encoded'] = label_encoder.fit_transform(df['fish'])

# Splitting the data into features and target
X = df[['ph', 'temperature', 'turbidity']]
y = df['fish_encoded']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handling class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Train Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(random_state=42)
gb_classifier.fit(X_train_balanced, y_train_balanced)

# Making predictions using the Gradient Boosting Classifier
y_pred_gb = gb_classifier.predict(X_test_scaled)

# Calculating accuracy
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print("Accuracy (Gradient Boosting Classifier):", accuracy_gb)

# Function to predict fish based on input values using the Gradient Boosting Classifier
def predict_fish_gb(ph, temperature, turbidity):
    # Scale input features
    input_features = scaler.transform([[ph, temperature, turbidity]])
    # Predict the fish using the Gradient Boosting Classifier
    prediction = gb_classifier.predict(input_features)
    # Inverse transform the prediction to get original labels
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

# predict
ph_input = 6
temperature_input = 26
turbidity_input = 27

predicted_fish_gb = predict_fish_gb(ph_input, temperature_input, turbidity_input)
print("Predicted fish (Gradient Boosting Classifier):", predicted_fish_gb)


# Generate classification report
class_report = classification_report(y_test, y_pred_gb)
print("Classification Report:\n", class_report)
# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_gb)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Get feature importance
feature_importance = gb_classifier.feature_importances_

# Plot feature importance
plt.figure(figsize=(8, 6))
plt.bar(X.columns, feature_importance)
plt.title('Feature Importance Plot')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()

train_sizes, train_scores, test_scores = learning_curve(gb_classifier, X_train_scaled, y_train, cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend(loc="best")
plt.show()
