from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def run_knn_classifier():
    """
    This function demonstrates the use of K-Nearest Neighbors (KNN) classifier
    to predict the species of iris flowers based on their measurements.

    It loads the iris dataset, splits it into training and testing sets,
    creates a KNN classifier, trains the classifier using the training set,
    and makes predictions on the test set.

    Returns:
    - predictions (array): The predicted species of the iris flowers in the test set.
    """

    # Load the dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the classifier
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = knn.predict(X_test)
    
    # Map the numeric predictions to the corresponding species
    species_dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    species_predictions = [species_dict[pred] for pred in predictions]

    return species_predictions
    
# Run the KNN classifier and print the predictions
print(run_knn_classifier())
