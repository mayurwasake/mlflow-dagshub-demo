import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns   

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter for the Random Forest Model 
max_depth = 15


# apply mlflow to track the model training
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris-dt")

with mlflow.start_run():
    # Train the Decision Tree Classifier
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Log parameters, metrics, and model to MLflow
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Accuracy: {accuracy}")

    # Logging the model to MLflow
    mlflow.sklearn.log_model(model, "decision_tree_model")

    #Adding the tags
    mlflow.set_tag("model_type", "Decision Tree")
    mlflow.set_tag("author", "Mayur")

    # Plot and log the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save the plot to a file and log it as an artifact
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
