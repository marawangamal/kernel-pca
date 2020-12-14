import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles as sk_make_circles


if __name__ == '__main__':

    # Generate toy data (concentric circles)
    X, y = sk_make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
    fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True)


    # Regular PCA
    [U, S, Vt] = np.linalg.svd(X.T)
    U = U.T
    axes[0].scatter(U[0, y==0], np.zeros((500,1)), s=40, color='r')
    axes[0].scatter(U[1, y==1], np.zeros((500,1)), s=5, color = 'b')
    axes[0].set_title('PCA')
    axes[0].set_xlabel('PC1')


    # Kernel PCA

    # 1. initialize KPCA object. This learns the principal subspace using data X
    kPCA = KernelPCA(X)

    # 2. get transformed data's projection onto first two principal components
    X_new = kPCA.A[:, :2].T
    axes[1].scatter(X_new[0, y==0], np.zeros((500, 1)), s=40, color='r')
    axes[1].scatter(X_new[0, y==1], np.zeros((500, 1)), s=5, color='b')
    axes[1].set_title('KPCA')
    axes[1].set_xlabel('PC1')

    plt.show()
