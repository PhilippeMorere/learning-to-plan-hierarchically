import torch


class DynamicArray:
    """
    Class providing an array of dynamic and increasing size.
    """

    def __init__(self, d=1, init_size=100):
        """
        Initialises the array of dynamic size but fixed dimension.
        :return:
        """
        self.capacity = init_size
        self.dim = d
        self.size = 0
        self.data = None

    def extend(self, rows):
        """
        Appends multiple elements to the array.
        :return:
        """
        for row in rows:
            self.append(row)

    def append(self, x):
        """
        Appends one element to the array.
        :return:
        """
        if self.size == 0:
            self._clear(x.dtype, x.device)
        elif self.size == self.capacity:
            self.capacity *= 4
            newdata = torch.zeros((self.capacity, self.dim),
                                  dtype=self.data.dtype,
                                  device=self.data.device)
            newdata[:self.size, :] = self.data
            self.data = newdata

        self.data[self.size, :] = x
        self.size += 1

    def get(self):
        """
        Returns the data from the array.
        :return: Tensor with data
        """
        if self.size == 0:
            return None
        return self.data[:self.size, :]

    def _clear(self, dtype, device):
        self.data = torch.zeros((self.capacity, self.dim),
                                dtype=dtype, device=device)
        self.size = 0

    def clear(self):
        """
        Clears the data from the array.
        :return: None
        """
        if self.data is not None:
            self._clear(self.data.dtype, self.data.device)

    def __getitem__(self, key):
        return self.data[key, :]

    def __setitem__(self, key, value):
        self.data[key, :] = value

    def __len__(self):
        return self.size
