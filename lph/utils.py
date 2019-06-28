import numpy as np


class SparseState:
    @classmethod
    def from_dense_state(cls, state):
        """
        Creates sparse state from dense state
        :param state: dense state
        :return: (SparseState)
        """
        # dims are assumed to be sorted!
        dims = state.reshape(-1).nonzero()[0]
        values = state.reshape(-1)[dims]
        return cls(dims, values, len(state))

    def __init__(self, dims, values, ds):
        self.dims = np.array(dims)
        self.values = np.array(values)
        self.ds = ds

    def __len__(self):
        return self.ds

    def copy(self):
        return SparseState(self.dims.copy(), self.values.copy(), self.ds)

    def distance_from(self, other):
        """
        Computes distance from other state to this state. Distance (not
        symmetric) is defined as the number of elements of self that are
        different or absent from other.
        :param other: other (SparseState)
        :return: distance (float)
        """
        dist = 0
        for i, d in enumerate(self.dims):
            if d not in other.dims:
                dist += 1
            else:
                idx = np.where(other.dims == d)[0][0]
                dist += 1 * (other.values[idx] != self.values[i])
        return dist

    def remove(self, other):
        """
        In-place subtraction where dimension is removed if values match.
        :param other: (SparseState)
        :return: None
        """
        for id_other in range(len(other.dims)):
            if other.dims[id_other] in self.dims:
                id_dim = np.where(self.dims == other.dims[id_other])[0]
                if self.values[id_dim] == other.values[id_other]:
                    self.values = np.delete(self.values, id_dim)
                    self.dims = np.delete(self.dims, id_dim)

    def partial_difference(self, b):
        """
        Computes the difference between self and b as a sparse state,
        using only the dimensions of self.
        :param b: other sparse state (SparseState)
        :return: partial difference (SparseState)
        """
        dims = self.dims.copy()
        values = self.values.copy()
        for i, d in enumerate(b.dims):
            if d in dims:
                idx = dims.find(d)
                values[idx] -= b.values[i]

        new_ss = SparseState(dims, values, self.ds)
        return new_ss

    def matches(self, state):
        """
        Whether the given state matches this state: ie. all dimensions of
        this state have the same values as the given state.
        :param state: dense state or (SparseState)
        :return: bool
        """
        if len(self.dims) == 0:
            return True
        if isinstance(state, SparseState):
            for i in range(len(self.dims)):
                if self.dims[i] not in state.dims and self.values[i] != 0:
                    return False
                i_other = np.where(state.dims == self.dims[i])[0]
                if len(i_other) == 0:  # means self.values[i] == 0
                    pass
                if self.values[i] != state.values[i_other]:
                    return False
            return True
        else:
            return np.all(self.values == state[self.dims])

    def to_dense(self):
        """
        Returns dense representation of (SparseState).
        Note: This assumes default value is 0 (False).
        :return: numpy array
        """
        s = np.zeros((1, self.ds))
        s[0, self.dims] = self.values
        return s

    def __str__(self):
        return "SparseState[d={}, v={}]".format(self.dims, self.values)

    def union(self, other):
        """
        Returns the union of this sparse state and the other.
        :param other: (SparseState)
        :return: union as (SparseState)
        """
        # NOTE: in the case of intersection, keep values from self.
        new_dims = list(self.dims)
        new_values = list(self.values)
        for i, d in enumerate(other.dims):
            if d not in new_dims:
                new_dims.append(d)
                new_values.append(other.values[i])

        # Make sure dimensions are sorted
        sort_idx = np.argsort(new_dims)
        return SparseState(np.array(new_dims)[sort_idx],
                           np.array(new_values)[sort_idx], self.ds)


class Effect:
    @classmethod
    def from_dense_start_goal(cls, start, goal):
        """
        Creates effect from dense start and goal states.
        :param start: dense start state
        :param goal: dense goal state
        :return: (Effect)
        """
        ds = len(start)
        s_goal = SparseState.from_dense_state(goal)
        start_values = start[s_goal.dims]
        end_values = s_goal.values.copy()
        dims = s_goal.dims.copy()
        return cls(dims, start_values.reshape(-1), end_values, ds)

    @classmethod
    def from_sparse_start_goal(cls, start, goal):
        """
        Creates effect from sparse start and goal states.
        :param start: start state (SparseState)
        :param goal: goal state (SparseState)
        :return: (Effect)
        """
        start_values = np.zeros(goal.values.shape)
        for i in range(start_values.shape[0]):
            if goal.dims[i] in start.dims:
                idx = np.where(start.dims == goal.dims[i])[0]
                start_values[i] = start.values[idx]
        return cls(goal.dims, start_values, goal.values, goal.ds)

    def __init__(self, dims, start_values, end_values, ds):
        """
        Creates an effect.
        Note: dimensions (dims) are assumed to be sorted.
        :param dims: effect dimensions as numpy array (or list)
        :param start_values: start effect values as numpy array (or list)
        :param end_values: end effect values as numpy array (or list)
        :param ds: dense state size
        """
        # Delete dimensions/values if common to both start and goal
        np_start, np_end = np.array(start_values), np.array(end_values)
        idx = np.where(np_start != np_end)[0]

        self.dims = np.array(dims)[idx]
        self.start_state = SparseState(self.dims, np_start[idx], ds)
        self.end_state = SparseState(self.dims, np_end[idx], ds)

    def copy(self):
        return Effect(self.dims, self.start_state.values,
                      self.end_state.values, self.end_state.ds)

    def __key(self):
        return (tuple(self.dims),
                tuple(self.start_state.values),
                tuple(self.end_state.values))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(self, type(other)) and self.__key() == other.__key()

    def __add__(self, other):
        new_dims = list(self.dims)
        new_start = list(self.start_state.values)
        new_end = list(self.end_state.values)
        for i, d in enumerate(other.dims):
            if d not in new_dims:
                new_dims.append(d)
                new_start.append(other.start_state.values[i])
                new_end.append(other.end_state.values[i])

        # Make sure dimensions are sorted
        sort_idx = np.argsort(new_dims)
        return Effect(np.array(new_dims)[sort_idx],
                      np.array(new_start)[sort_idx],
                      np.array(new_end)[sort_idx], self.start_state.ds)

    def __contains__(self, other):
        """
        Whether other is included in this effect.
        :param other: (Effect)
        :return: (bool)
        """
        com_dim = np.intersect1d(self.dims, other.dims, True)
        if len(com_dim) != len(other.dims):
            return False
        idx_self = np.concatenate(
            [np.where(self.dims == d)[0] for d in com_dim])
        idx_other = np.concatenate(
            [np.where(other.dims == d)[0] for d in com_dim])
        c = (self.start_state.values[idx_self] == other.start_state.values[
            idx_other]) + \
            (self.start_state.values[idx_self] == other.start_state.values[
                idx_other])
        return np.all(c)

    def __str__(self):
        return "Effect: {} -> {}".format(self.start_state, self.end_state)

    def similarity_to(self, other):
        """
        Computes similarity between effect and other effect.
        :param other: other (Effect)
        :return: similarity (float)
        """
        com_dim = np.intersect1d(self.dims, other.dims, True)
        if len(com_dim) == 0:
            return 0.0
        idx_self = np.concatenate(
            [np.where(self.dims == d)[0] for d in com_dim])
        idx_other = np.concatenate(
            [np.where(other.dims == d)[0] for d in com_dim])
        sim = np.sum([self.start_state.values[idx_self[i]] ==
                      other.start_state.values[idx_other[i]] and
                      self.start_state.values[idx_self[i]] ==
                      other.start_state.values[idx_other[i]]
                      for i in range(len(idx_self))])
        return sim

    def unit_effects(self):
        """
        Generate a list of unit sub effects (unit meaning single dimension).
        :return: (list) of (Effect)
        """
        effects = []
        ds = self.end_state.ds
        for i, d in enumerate(self.dims):
            effects.append(Effect([d], self.start_state.values[[i]],
                                  self.end_state.values[[i]], ds))
        return effects
