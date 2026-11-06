import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np


def lagrange(xs, ys, x):
    total = 0.0

    for i in range(len(xs)):
        term = ys[i]

        for j in range(len(xs)):
            if j != i:
                term *= (x - xs[j]) / (xs[i] - xs[j])

        total += term

    return total


def divided_differences(xs, ys):
    row = np.array(ys, dtype=float)
    coefficients = [row[0]]

    for order in range(1, len(xs)):
        row = (row[1:] - row[:-1]) / (xs[order:] - xs[:-order])
        coefficients.append(row[0])

    return np.array(coefficients)


def newton(xs, coefficients, x):
    values = np.asarray(x, dtype=float)
    result = np.full_like(values, coefficients[-1])

    for k in range(len(coefficients) - 2, -1, -1):
        result = coefficients[k] + (values - xs[k]) * result

    return result


def plot_function(index, function, name, a, b, node_counts):
    figure, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()

    print(f"II.{index}: {name} on [{a}, {b}]")
    print(
        f"{'N':>4} {'xi':>10} {'f(xi)':>14} {'Lagrange':>14} {'Newton':>14} "
        f"{'|L-f|':>12} {'|N-f|':>12} {'|L-N|':>12}"
    )

    for axis, count in zip(axes, node_counts):
        xs = np.linspace(a, b, count)
        ys = function(xs)
        coefficients = divided_differences(xs, ys)

        xi = a + 0.371234 * (b - a)
        exact = function(xi)
        lagrange_value = lagrange(xs, ys, xi)
        newton_value = float(newton(xs, coefficients, xi))

        print(
            f"{count:4d} {xi:10.6f} {exact:14.8f} {lagrange_value:14.8f} "
            f"{newton_value:14.8f} {abs(lagrange_value - exact):12.3e} "
            f"{abs(newton_value - exact):12.3e} "
            f"{abs(lagrange_value - newton_value):12.3e}"
        )

        dense = np.linspace(a, b, 400)
        values = function(dense)
        margin = 0.2 * (values.max() - values.min() + 1e-12)

        axis.plot(dense, values, label="f(x)")
        axis.plot(dense, newton(xs, coefficients, dense), label="P(x)")
        axis.scatter(xs, ys, color="black", s=12)
        axis.set_title(f"N = {count}")
        axis.set_xlim(a, b)
        axis.set_ylim(values.min() - margin, values.max() + margin)
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle(name)
    figure.tight_layout()


def main():
    functions = [
        (
            1,
            lambda x: x**3 - 6.5 * x**2 + 11.0 * x - 4.0,
            "x^3 - 6.5x^2 + 11x - 4",
            2.0,
            4.0,
        ),
        (
            2,
            lambda x: 3.0 * np.cos(np.pi * x / 8.0),
            "3 cos(pi x / 8)",
            0.5,
            3.0,
        ),
        (
            3,
            lambda x: np.exp(-x / 4.0) * np.sin(x / 3.0),
            "exp(-x/4) sin(x/3)",
            4.0,
            10.0,
        ),
        (
            4,
            lambda x: 8.0 * x * np.exp(-(x**2) / 12.0),
            "8x exp(-x^2/12)",
            0.0,
            12.0,
        ),
    ]

    node_counts = [3, 4, 8, 10, 16, 32, 64, 256]

    for index, function, name, a, b in functions:
        plot_function(index, function, name, a, b, node_counts)

    plt.show()


if __name__ == "__main__":
    main()
