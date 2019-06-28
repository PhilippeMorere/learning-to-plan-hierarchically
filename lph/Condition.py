import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.mixture import GaussianMixture
from lph.utils import SparseState


class AbstractCondition:
    def update(self, x, t):
        """
        Updates skill success conditions.
        :param x: input
        :param t: target (bool)
        :return: None
        """
        raise NotImplementedError

    def fails_in(self, x):
        """
        Returns whether the skill has previously failed in this state.
        :param x: dense state
        :return: bool
        """
        raise NotImplementedError

    def sample(self):
        """
        Returns a possible condition for skill to succeed.
        :return: dimensions, values as (tuple)
        """
        raise NotImplementedError


class LearnedCondition(AbstractCondition):
    def __init__(self, n_gmm_components=1, omp_tol=2.):
        self.gmm = GaussianMixture()  # Dummy init
        self.n_gmm_components = n_gmm_components
        self.omp_tol = omp_tol
        self.fitted = False
        self.all_x = []
        self.all_t = []
        self.used_dims = np.array([])
        self.gmm_samples_dim = np.array([])
        self.gmm_samples_values = np.array([])

    def update(self, x, t):
        """
        Updates skill success conditions.
        :param x: input
        :param t: target (bool)
        :return: None
        """
        # Don't add duplicates
        for i in range(len(self.all_x)):
            if self.all_t[i] == t and np.all(self.all_x[i] == x):
                return

        # Update skill dataset
        self.all_x.append(x)
        self.all_t.append(t)

        # Can't learn from too few elements
        n_x = len(self.all_x)
        n_unique_t = np.unique(self.all_t).shape[0]
        if n_x <= 2 or n_unique_t < 2:
            return

        # apply OMP
        all_x = np.array(self.all_x)
        all_t = np.array(self.all_t).reshape(-1, 1)
        omp = OrthogonalMatchingPursuit(tol=self.omp_tol)
        omp.fit(all_x, all_t)

        # Trim data
        self.used_dims, = omp.coef_.nonzero()
        x_trim = all_x[:, self.used_dims]
        x_true = x_trim[np.where(all_t)[0], :]

        if x_true.shape[0] >= 2 and x_true.shape[1] > 0:
            # train GMM
            self.gmm = GaussianMixture(
                n_components=min(len(self.used_dims), self.n_gmm_components))
            # Add a bit of noise to avoid duplicate points
            x_noisy = x_true + 0.01 * np.random.normal(size=x_true.shape)
            self.gmm.fit(x_noisy)
            self.fitted = True
            self.gmm_samples_dim, self.gmm_samples_values = \
                self.__generate_gmm_samples(20)

    def sample(self):
        """
        Returns a possible condition for skill to succeed.
        :return: dimensions, values as (tuple)
        """
        # Logistic regression weights reflect which dimensions are important
        if not self.fitted:
            return np.array([]), np.array([])

        rand_id = np.random.randint(0, len(self.gmm_samples_values))
        return self.gmm_samples_dim[rand_id], self.gmm_samples_values[rand_id]

    def __generate_gmm_samples(self, n):
        # Get sample from GMM
        x_samples = np.round(self.gmm.sample(n)[0])
        dims, values = [], []
        for i in range(n):
            non_zero = np.nonzero(x_samples[i])[0]
            dims.append(self.used_dims[non_zero])
            values.append(x_samples[i][non_zero])
        return dims, values

    def fails_in(self, x):
        """
        Returns whether the skill has previously failed in this state.
        :param x: dense state
        :return: bool
        """
        for i in range(len(self.all_x)):
            if not self.all_t[i]:
                if len(self.used_dims) == 0:
                    if np.all(self.all_x[i] == x):
                        return True
                elif np.all(self.all_x[i][self.used_dims] == x[self.used_dims]):
                    return True
        return False

    def is_ready(self):
        """
        Whether condition is ready to be used/reliable.
        :return: (bool)
        """
        return self.fitted or (
                len(self.all_x) > 0 and np.all(np.array(self.all_t)))

    def __str__(self):
        if self.is_ready():
            d, v = self.sample()
            return "LearnedCondition: d={}, v={}".format(d, v)
        else:
            return "LearnedCondition UNKNOWN\n\tx={}, y={}".format(
                self.all_x, self.all_t)


class FixedCondition(AbstractCondition):
    def update(self, x, t):
        """
        Updates skill success conditions.
        :param x: input
        :param t: target (bool)
        :return: None
        """
        pass

    def __init__(self, dims, values):
        self.dims = np.array(dims)
        self.values = np.array(values)

    @staticmethod
    def is_ready():
        """
        Whether condition is ready to be used/reliable.
        :return: (bool)
        """
        return True

    def sample(self):
        """
        Returns a possible condition for skill to succeed.
        :return: dimensions, values as (tuple)
        """
        return self.dims, self.values

    def fails_in(self, x):
        """
        Returns whether the skill has previously failed in this state.
        :param x: dense state
        :return: bool
        """
        dims, values = self.sample()
        ss = SparseState(dims, values, x.shape[0])
        return not ss.matches(x)

    def __str__(self):
        d, v = self.sample()
        return "FixedCondition: d={}, v={}".format(d, v)
