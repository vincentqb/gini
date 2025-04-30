import numpy as np


class GiniHistogramApproximator:
    def __init__(self, bin_edges):
        self.bin_edges = bin_edges
        self.bin_centers = 0.5 * (self.bin_edges[1:] + self.bin_edges[:-1])
        self.bin_counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
        self.total_sum = 0

    def update(self, values):
        values = np.asarray(values)
        if any(values < 0):
            print("Warning: Cannot process negative values. Skipping.")
            return
        self.bin_counts += np.histogram(values, bins=self.bin_edges)[0]
        self.total_sum += np.sum(values)

    def compute(self):
        total_count = self.bin_counts.sum()
        total_sum = self.total_sum
        if total_count == 0 or total_sum == 0:
            return 0.0

        # Compute cumulative rank positions
        end_ranks = np.cumsum(self.bin_counts)
        start_ranks = end_ranks - self.bin_counts + 1  # ranks start at 1
        avg_ranks = (start_ranks + end_ranks) / 2.0

        # Gini numerator: sum (2i - n - 1) * x_i for i = rank
        numerator = ((2 * avg_ranks - total_count - 1) * self.bin_centers * self.bin_counts).sum()
        denominator = total_count * total_sum
        return numerator / denominator


if __name__ == "__main__":
    approximator = GiniHistogramApproximator(bin_edges=np.linspace(0, 100, 101))  # 100 bins
    for _ in range(10):
        # Simulate multiple batches
        errors = np.abs(np.random.randn(1000) * 10)  # Simulated quantile errors
        approximator.update(errors)
    print(approximator.compute())

    array = np.zeros((1000))
    array[0] = 1.0
    approximator = GiniHistogramApproximator(bin_edges=np.linspace(0, 100, 101))  # 100 bins
    approximator.update(array)
    print(np.float64(0.998900109989001), "?")
    print(approximator.compute())

    array = np.random.uniform(-1, 0, 1000)
    array += 1e-7 - array.min()  # FIXME cannot be negative
    approximator = GiniHistogramApproximator(bin_edges=np.linspace(0, 1, 10))  # 100 bins
    approximator.update(array)
    print(np.float64(0.33020664112202275), "?")
    print(approximator.compute())

    array = np.ones((1000))
    approximator = GiniHistogramApproximator(bin_edges=np.linspace(0, 1000, 101))  # 100 bins
    approximator.update(array)
    print(np.float64(-6.938893903907231e-17), "?")
    print(approximator.compute())
