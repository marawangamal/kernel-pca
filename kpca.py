class KernelPCA(object):
    """A class used to compute Kernel PCA. Initialize with data matrix, then use reduce inference method for dim. reduction and reconstruct method for reconstruction.
    Kernel function can be changed in the kernel method. (default is gaussian)
    Attributes:
      A: (matrix) [NxN] columns contain eigenvectors of Ka_k = N*lambda*a_k
      K: (matrix) [NxN] gram matrix comupted using kernel method
      K_tilde: (matrix) [NxN] centered gram matrix
    Methods:
      inferece: performs dimensionality reduction
      reconstruct: returns a reconstruction in original space
    """

    def __init__(self, X, kernel_c=20, compute_cond =False):
        """ Learning step. Finds a_k in eigen problem: K_tilde a_k = lamda a_k for k = 1, 2,...,N
        Args:
            X: (matrix) data. [d, N]
            kernel_c: (scalar) hyperparameter for gaussian kernel
        """
        super(KernelPCA, self).__init__()
        self.X = X
        self.d, self.N = X.shape
        self.c=kernel_c

        # Build K gram matrix using kernel function (default gaussian)
        self.K = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i, self.N):
                self.K[i, j] = self.kernel(X[:,i], X[:, j])
                self.K[j, i] = self.K[i, j]


        # Centered Kernel
        N=self.N
        K_tilde = self.K  -  (1/N)*np.ones((N, N)) @ self.K  -  self.K @ ((1/N)*np.ones((N, N)))
        + (1/N)*np.ones((N, N)) @ self.K @ ((1/N)*np.ones((N, N)))


        if(compute_cond):
          self.cond_K = np.linalg.cond(self.K)
          self.cond_K_tilde = np.linalg.cond(K_tilde)

        # Eignectors of centerd matrix
        evals, evecs = np.linalg.eig(K_tilde)
        indices = np.argsort(np.abs(evals))[::-1]
        self.A = evecs[:, indices] # [N, N] # ascending order

    def kernel(self, x, y):
        """ Gaussian kernel. kappa(x, y)
        """
        return np.exp(-np.square(np.linalg.norm(x-y))/self.c)

    def kernel_tilde(self, x, y):
        """ (centered) kernel function. kappa_tilde(x, y)
        Args:
            x, y: (vector) [d, 1] pre-image vectors.
        Returns:
            k_tilde_xy: (scalar) centered kernel value. phi_tilde(x).T phi_tilde(y)
        """
        # kx and ky
        kx = [self.kernel(x, self.X[:,n]) for n in range(self.N)]
        ky = [self.kernel(self.X[:,n], y) for n in range(self.N)]
        kx = np.reshape(np.array(kx), (-1, 1))
        ky = np.reshape(np.array(ky), (-1, 1))

        N = self.N
        k_tilde_xy = self.kernel(x, y)  -  (1/N)*np.ones((N,1)).T@kx  -  (1/N)*np.ones((N,1)).T@ky
        + (1/N*N)*np.ones((N,1)).T@ self.K @np.ones((N,1))

        return k_tilde_xy

    def inferece(self, x):
        """ Obtains coefficients of new basis that phi_tilde(x) is projected onto.
        Args:
            x: (vector) [d, 1] pre-image.
        Returns:
            betas: (vector) [N, 1] coefficients of principal components phi(x) is projected onto
        """

        K_tilde_x = np.reshape(np.array([self.kernel_tilde(x, self.X[:, n]) for n in range(self.N)]), (-1, 1))
        self.beta = self.A.T @ K_tilde_x # [N,N]  [N, 1]

        return self.beta

    def reconstruct(self, z0, K_max=2, iters=10, print_iters=False):
        """ Uses fixed point iteration to solve:  z_hat = min_z || phi_projected(x) - phi(z)||
        Args:
          z0: (vector) [dx1] initial point, where d is dimension in original space.
          K_max: (scaler) number of PCs to use.
          iter: (scaler) maximum number of iterations to perform FPI.
        Returns:
          z: (vector) [dx1] optimal reconstruction found after t steps.
        """
        _, self.inferece(z0)
        z = z0
        for t in range(iters):
            num=0
            den=0
            for n in range(self.N):
                gamma_n = self.A[n,:K_max] @ self.beta[:K_max]
                num += self.kernel(z, self.X[:, n]) * gamma_n * self.X[:, n]
                den += self.kernel(z, self.X[:, n]) * gamma_n
            z_new = num/den
            diff = np.linalg.norm(z_new - z)
            if(print_iters): print(diff)
            z = z_new
            if(diff < np.power(10.0, -3.0)):
                break

        return z
