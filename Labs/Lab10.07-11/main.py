import argparse
import os

import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np


def least_squares(x, y, columns):
    matrix = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    residual = y - matrix @ coefficients
    rmse = np.sqrt(np.mean(residual**2))
    return coefficients, rmse


def plot(x, y, curve_x, curve_y, title, equation):
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.scatter(x, y, color="tab:blue", label="data")
    axes.plot(curve_x, curve_y, color="tab:red", label="model")
    axes.set_title(f"{title}\n{equation}")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()


def show(number, name, equation, rmse, x, y, model):
    print(f"Task {number}: {name}")
    print(f"  {equation}")
    print(f"  RMSE = {rmse:.6f}")
    print()

    curve_x = np.linspace(x.min(), x.max(), 400)
    plot(x, y, curve_x, model(curve_x), f"Task {number}. {name}", equation)


def task_7():
    x = np.array([51, 67, 84, 81, 101, 109, 71, 97, 109, 51, 105, 89], dtype=float)
    y = np.array([25, 30, 43, 44, 57, 58, 43, 46, 62, 45, 55, 45], dtype=float)

    coefficients, rmse = least_squares(x, y, [np.ones_like(x), x])
    b0, b1 = coefficients

    show(
        7,
        "linear regression",
        f"y = {b0:.6f} {b1:+.6f} * x",
        rmse,
        x,
        y,
        lambda values: b0 + b1 * values,
    )


def task_8():
    x = np.array(
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
        dtype=float,
    )
    y = np.array(
        [0.4, 0.3, 1.0, 1.7, 2.1, 3.4, 4.1, 5.8, 7.7, 9.4, 11.4, 13.6, 15.6, 18.6, 21.2, 24.1],
        dtype=float,
    )

    coefficients, rmse = least_squares(x, y, [np.ones_like(x), x, x**2])
    b0, b1, b2 = coefficients

    show(
        8,
        "quadratic regression",
        f"y = {b0:.6f} {b1:+.6f} * x {b2:+.6f} * x^2",
        rmse,
        x,
        y,
        lambda values: b0 + b1 * values + b2 * values**2,
    )


def task_9():
    x = np.arange(1.0, 13.0)
    y = np.array(
        [2.11, 2.45, 2.61, 2.73, 2.75, 2.81, 2.87, 2.91, 2.96, 3.03, 3.05, 3.12],
        dtype=float,
    )

    coefficients, rmse = least_squares(x, y, [np.ones_like(x), np.log(x)])
    b0, b1 = coefficients

    show(
        9,
        "logarithmic regression",
        f"y = {b0:.6f} {b1:+.6f} * ln(x)",
        rmse,
        x,
        y,
        lambda values: b0 + b1 * np.log(values),
    )


def task_10():
    x = np.arange(1.0, 13.0)
    y = np.array(
        [0.10, 0.21, 0.43, 0.51, 0.62, 0.81, 1.01, 1.23, 1.47, 1.53, 1.75, 2.25],
        dtype=float,
    )

    coefficients, rmse = least_squares(x, y, [np.ones_like(x), np.exp(0.1 * x)])
    b0, b1 = coefficients

    show(
        10,
        "exponential regression",
        f"y = {b0:.6f} {b1:+.6f} * exp(0.1 * x)",
        rmse,
        x,
        y,
        lambda values: b0 + b1 * np.exp(0.1 * values),
    )


def task_11():
    x = np.array(
        [2.11, 2.45, 2.61, 2.73, 2.75, 2.81, 2.87, 2.91, 2.96, 3.03, 3.05, 3.12],
        dtype=float,
    )
    y = np.array(
        [0.10, 0.21, 0.43, 0.51, 0.62, 0.81, 1.01, 1.23, 1.47, 1.53, 1.75, 2.25],
        dtype=float,
    )

    coefficients, rmse = least_squares(x, y, [np.ones_like(x), x, np.sin(8.0 * x)])
    b0, b1, b2 = coefficients

    show(
        11,
        "regression with a sinusoidal term",
        f"y = {b0:.6f} {b1:+.6f} * x {b2:+.6f} * sin(8 * x)",
        rmse,
        x,
        y,
        lambda values: b0 + b1 * values + b2 * np.sin(8.0 * values),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=int,
        choices=[7, 8, 9, 10, 11],
        help="task number, all tasks by default",
    )
    arguments = parser.parse_args()

    tasks = {
        7: task_7,
        8: task_8,
        9: task_9,
        10: task_10,
        11: task_11,
    }

    if arguments.task is None:
        for task in tasks.values():
            task()
    else:
        tasks[arguments.task]()

    plt.show()


if __name__ == "__main__":
    main()
