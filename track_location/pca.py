import numpy as np

def pca(X, num_components):
    # Normalize the data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Compute the covariance matrix
    cov_matrix = np.cov(X, rowvar=False)
    
    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort the eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the top 'num_components' eigenvectors
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]
    
    # Project the data onto the selected eigenvectors
    projected_data = np.dot(X, selected_eigenvectors)
    
    return projected_data


def dist(centroid, point):
    return np.sum(np.square(np.subtract(centroid, point)))

def k_means(dataset, centroids):
    centroids = [centroid for centroid in centroids]
    old_centroids = None

    while old_centroids is None or not all(np.allclose(old_centroids[i], centroids[i]) for i in range(len(centroids))):
        old_centroids = centroids[:]
        classes = [[] for _ in range(len(centroids))]
        for point in dataset:
            closest_centroid = np.argmin([dist(centroid, point) for centroid in centroids])
            classes[closest_centroid].append(point)

        for i, class_ in enumerate(classes):
            if len(class_) != 0:
                centroids[i] = np.mean(class_, axis=0)
        

    return tuple(centroid for centroid in centroids)


