import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people


def plot_vector_as_image(image, h, w):
    """
    utility function to plot a vector as image.
    Args:
    image - vector of pixels
    h, w - dimensions of original pi
    """
    plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
    plt.title('title', size=12)
    plt.show()


def get_pictures_by_name(name='Ariel Sharon'):
    """
    Given a name returns all the pictures of the person with this specific name.
    YOU CAN CHANGE THIS FUNCTION!
    THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
    """
    selected_images = []
    n_samples, h, w = lfw_people.images.shape
    target_label = list(lfw_people.target_names).index(name)
    for image, target in zip(lfw_people.images, lfw_people.target):
        if target == target_label:
            image_vector = image.reshape((h * w, 1))
            selected_images.append(image_vector)
    return selected_images, h, w


def load_data():
    # Don't change the resize factor!!!
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw_people


######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""


def PCA(X, k):
    """
    Compute PCA on the given matrix.

    Args:
        X - Matrix of dimensions (n,d). Where n is the number of sample points and d is the dimension of each sample.
        For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimension of the matrix
         would be (10,100).
        k - number of eigenvectors to return

    Returns:
      U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest
       k eigenvectors of the covariance matrix.
      S - k the largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
    """
    n, d = X.shape

    mean_vector = X.mean(axis=0)

    covariance_matrix = np.dot((X - mean_vector).T, (X - mean_vector)) / (n - 1)

    U, S, Vt = np.linalg.svd(covariance_matrix)

    U = U[:, :k]
    S = S[:k]

    return U, S


# data: n=1288, d=1850
# images: n=1288, d=50x37
# target: n=1288, label=1
# target_name: label name=7
lfw_people = load_data()

u, s = PCA(X=lfw_people['data'], k=5)

get_pictures_by_name()

print("done")
