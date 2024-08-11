import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import random


def plot_vector_as_image(image, h, w, ax, title=""):
    """
    Utility function to plot a vector as an image on given axes.
    Args:
        image: vector of pixels
        h, w: dimensions of original picture
        ax: matplotlib Axes object to plot on
        title: title of the subplot
    """
    ax.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
    ax.set_title(title, size=12)
    ax.axis('off')  # Turn off axis numbers and ticks


def get_pictures_by_name(name='Ariel Sharon'):
    """
    Given a name, returns all the pictures of the person with this specific name.
    """
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw_people.images.shape
    target_label = list(lfw_people.target_names).index(name)
    selected_images = [image.ravel() for image, target in zip(lfw_people.images, lfw_people.target) if
                       target == target_label]
    return np.array(selected_images), h, w


def reconstruct_images(U, S, mean_vector, X, k):
    """
    Project images to k dimensions and reconstruct them back to original dimensions.
    """
    # Reduce dimension
    Z = np.dot(X - mean_vector, U[:, :k])

    # Reconstruct images
    X_reconstructed = np.dot(Z, U[:, :k].T) + mean_vector

    return X_reconstructed


def plot_comparison(original, reconstructed, h, w, title=""):
    """
    Plot the original and reconstructed images side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(original.reshape((h, w)), cmap=plt.cm.gray)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(reconstructed.reshape((h, w)), cmap=plt.cm.gray)
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def PCA(X, k):
    """
    Compute PCA on the given matrix.
    """
    n, d = X.shape
    mean_vector = X.mean(axis=0)
    covariance_matrix = np.dot((X - mean_vector).T, (X - mean_vector)) / (n - 1)
    U, S, Vt = np.linalg.svd(covariance_matrix)
    return U[:, :k], S[:k]
