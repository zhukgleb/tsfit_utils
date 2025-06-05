import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from tsfit_utils import get_model_data
from typing import List, Union
from tsfit_utils import clean_pd
import scienceplots
from scipy.stats import norm


def weighted_kde(x, weights, x_grid, bandwidth=0.1):
    kde_values = np.zeros_like(x_grid)
    for xi, wi in zip(x, weights):
        kde_values += wi * norm.pdf(x_grid, loc=xi, scale=bandwidth)
    return kde_values / kde_values.sum()


def teff_graph(path2result: str):
    data = np.genfromtxt(path2result)
    data = data[
        (data[:, -1] == 0) & (data[:, -2] == 0)
    ]  # Remove all warning and error lines
    teff = data[:, 1]
    teff_err = data[:, 2]
    ew = data[:, -3]
    with plt.style.context("science"):
        _, ax = plt.subplots(figsize=(4, 4))
        ax.errorbar(
            teff,
            ew,
            xerr=teff_err,
            fmt="o",
            color="black",
            alpha=0.8,
            elinewidth=1,
            capsize=0,
        )
        ax.set_xlabel(r"$T_{eff}$, K")
        ax.set_ylabel("EW")
        plt.show()
    print(data)


def plot_scatter_df_results(
    df_results: pd.DataFrame,
    x_axis_column: str,
    y_axis_column: str,
    xlim=None,
    ylim=None,
    color="black",
    invert_x_axis=False,
    invert_y_axis=False,
    **pltargs,
):
    if color in df_results.columns.values:
        pltargs["c"] = df_results[color]
        pltargs["cmap"] = "viridis"
        pltargs["vmin"] = df_results[color].min()
        pltargs["vmax"] = df_results[color].max()
        plot_colorbar = True
    else:
        pltargs["color"] = color
        plot_colorbar = False
    plt.scatter(df_results[x_axis_column], df_results[y_axis_column], **pltargs)
    plt.xlabel(x_axis_column)
    plt.ylabel(y_axis_column)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if invert_x_axis:
        plt.gca().invert_xaxis()
    if invert_y_axis:
        plt.gca().invert_yaxis()
    if plot_colorbar:
        # colorbar with label
        plt.colorbar(label=color)
    plt.show()
    plt.close()


def plot_metall(data: pd.DataFrame, ratio: str = "Fe_H"):
    metallicity = data[ratio].to_numpy(float)
    error = data["chi_squared"].to_numpy(float)

    xy_point_density = np.vstack([metallicity, error])
    z_point_density = gaussian_kde(xy_point_density)(xy_point_density)
    idx_sort = z_point_density.argsort()
    x_plot, y_plot, z_plot = (
        metallicity[idx_sort],
        error[idx_sort],
        z_point_density[idx_sort],
    )

    weights = 1 / error
    weighted_avg = np.sum(metallicity * weights) / np.sum(weights)

    median = np.median(metallicity)

    lower_bound = np.percentile(metallicity, 2.5)
    upper_bound = np.percentile(metallicity, 97.5)

    print(f"Взвешенное среднее: {weighted_avg:.3f}")
    print(f"Медиана: {median:.3f}")
    print(f"95% доверительный интервал: [{lower_bound:.3f}, {upper_bound:.3f}]")

    with plt.style.context("science"):
        _, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(r"Metallicity IRAS Z02229+6208")
        ax.set_ylabel(r"$\chi^{2}$")
        ax.set_xlabel(r"Metallicity, [Fe/H]")
        ax.set_ylim((0, 10))
        density = ax.scatter(x_plot, y_plot, c=z_plot)
        plt.colorbar(density)
        plt.show()


def plot_ion_balance(data: pd.DataFrame):
    metallicity = data["Fe_H"].to_numpy(float)
    ew = data["ew"].to_numpy(float)
    ew = ew
    lamb = data["wave_center"].to_numpy(float)
    rel_ew = ew / lamb

    # Regression
    x = rel_ew.reshape((-1, 1))
    y = metallicity
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    slope = model.coef_
    slope = slope[0]
    intercept = model.intercept_
    print("coefficient of determination:", r_sq)
    print("intercept:", model.intercept_)
    print("slope:", model.coef_)
    xfit = np.linspace(0, 100, 1000)
    yfit = model.predict(xfit[:, np.newaxis])
    with plt.style.context("science"):
        _, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(r"Ionization balance of IRAS Z02229+6208")
        ax.set_ylabel(r"Metallicity, [Fe/H]")
        ax.set_xlabel(r"$EW / \lambda$")
        ax.set_ylim((-0.5, 0))
        # ax.set_xscale("log")
        text = f"Slope: {slope:.4f}\nIntercept: {intercept:.2f}\nR squared: {r_sq:.2f}"
        plt.annotate(
            text,
            xy=(0.95, 0.95),  # upper left corner
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"
            ),
        )
        plt.scatter(
            rel_ew, metallicity, color="black", alpha=0.5, label="derived metallicity"
        )
        plt.plot(xfit, yfit, label="linear regression")
        plt.legend()
        plt.show()


def plot_metallVS(data_1: pd.DataFrame, data_2: pd.DataFrame, ratio: str = "Fe_H"):
    metallicity_1 = data_1[ratio].to_numpy(float)
    error_1 = data_1["chi_squared"].to_numpy(float)

    xy_point_density_1 = np.vstack([metallicity_1, error_1])
    z_point_density_1 = gaussian_kde(xy_point_density_1)(xy_point_density_1)
    idx_sort_1 = z_point_density_1.argsort()
    x_plot_1, y_plot_1, z_plot_1 = (
        metallicity_1[idx_sort_1],
        error_1[idx_sort_1],
        z_point_density_1[idx_sort_1],
    )

    metallicity_2 = data_2[ratio].to_numpy(float)
    print(np.std(metallicity_2))
    error_2 = data_2["chi_squared"].to_numpy(float)

    xy_point_density_2 = np.vstack([metallicity_2, error_2])
    z_point_density_2 = gaussian_kde(xy_point_density_2)(xy_point_density_2)

    idx_sort_2 = z_point_density_2.argsort()
    x_plot_2, y_plot_2, z_plot_2 = (
        metallicity_2[idx_sort_2],
        error_2[idx_sort_2],
        z_point_density_2[idx_sort_2],
    )

    with plt.style.context("science"):
        _, ax = plt.subplots(nrows=2, ncols=1, figsize=(6.5, 9))
        # ax[0].set_title(r"Metallicity IRAS Z02229+6208")
        ax[0].set_title(r"Metallicity IRAS 07430+1115, T=6000, log~g=1")

        ax[0].set_ylabel(r"$\chi^{2}$")
        ax[0].set_xlabel(r"Metallicity, [Fe/H]")
        ax[0].set_ylim((0, 100))
        ax[1].set_title(r"Metallicity IRAS 07430+1115, T=4900, log~g=0.5")
        ax[1].set_ylabel(r"$\chi^{2}$")
        ax[1].set_xlabel(r"Metallicity, [Fe/H]")
        ax[1].set_ylim((0, 100))
        density_1 = ax[0].scatter(x_plot_1, y_plot_1, c=z_plot_1)
        density_2 = ax[1].scatter(x_plot_2, y_plot_2, c=z_plot_2)

        plt.colorbar(density_2)
        plt.colorbar(density_1)
        # plt.savefig("ReddyVSZhuck.pdf", dpi=600)
        plt.show()


def hist_estimation(df, bins):
    metall = "Fe_H"
    # b = df.iloc[:, 1:].values
    counts, bins = np.histogram(pd.to_numeric(df[metall]), 30)
    print(counts)
    sigma = (max(bins) ** 0.5) / (
        (bins[-1] - bins[-2]) * len(pd.to_numeric(df[metall]))
    )
    print(sigma)
    plt.stairs(counts, bins)
    # plt.hist(df[metall], bins, histtype="bar", alpha=0.5)
    # plt.xlabel(metall)
    # plt.ylabel("Count")
    plt.show()


def plot_metall_error(data: pd.DataFrame):
    metallicity = data["Fe_H"].to_numpy(float)
    error = data["chi_squared"].to_numpy(float)
    kde = gaussian_kde(metallicity, bw_method="scott")  # bw_method можно настроить
    x_grid = np.linspace(metallicity.min() - 0.05, metallicity.max() + 0.05, 1000)
    pdf = kde(x_grid)  # Плотность вероятности

    mode = x_grid[np.argmax(pdf)]

    cdf = np.cumsum(pdf) / np.sum(pdf)  # Нормированная кумулятивная функция
    lower_bound = x_grid[np.searchsorted(cdf, 0.025)]
    upper_bound = x_grid[np.searchsorted(cdf, 0.975)]

    # Визуализация
    plt.figure(figsize=(8, 5))
    plt.plot(x_grid, pdf, label="KDE (оценка плотности)", color="blue")
    plt.axvline(mode, color="red", linestyle="--", label=f"Мода: {mode:.3f}")
    plt.axvline(
        lower_bound, color="green", linestyle="--", label=f"2.5%: {lower_bound:.3f}"
    )
    plt.axvline(
        upper_bound, color="green", linestyle="--", label=f"97.5%: {upper_bound:.3f}"
    )
    plt.title("Ядровая оценка плотности (KDE)")
    plt.xlabel("Металличность")
    plt.ylabel("Плотность вероятности")
    plt.legend()
    print(f"Мода металличности: {mode:.3f}")
    print(f"95% доверительный интервал: [{lower_bound:.3f}, {upper_bound:.3f}]")
    plt.show()


def plot_metall_KDE(data: pd.DataFrame, ratio: str = "Fe_H", bandwidth=0.05):
    metallicity = data[ratio].to_numpy(float)
    print(f"Mean: {np.mean(metallicity)}")
    chi_squared = data["chi_squared"].to_numpy(float)
    weights = 1 / chi_squared
    weights /= weights.sum()

    x_grid = np.linspace(
        metallicity.min() - bandwidth, metallicity.max() + bandwidth, 1000
    )
    pdf = weighted_kde(metallicity, weights, x_grid, bandwidth=bandwidth)

    mode = x_grid[np.argmax(pdf)]

    cdf = np.cumsum(pdf) / np.sum(pdf)
    lower_bound = x_grid[np.searchsorted(cdf, 0.025)]
    upper_bound = x_grid[np.searchsorted(cdf, 0.975)]
    with plt.style.context("ggplot"):
        plt.figure(figsize=(8, 6))
        plt.plot(x_grid, pdf, label="KDE", color="blue")
        plt.axvline(mode, color="red", linestyle="--", label=f"Mode: {mode:.3f}")
        plt.axvline(
            lower_bound, color="green", linestyle="--", label=f"2.5%: {lower_bound:.3f}"
        )
        plt.axvline(
            upper_bound,
            color="green",
            linestyle="--",
            label=f"97.5%: {upper_bound:.3f}",
        )
        plt.title("KDE with weight")
        plt.xlabel("Metallicity")
        plt.ylabel("PDF")
        plt.legend()
        plt.show()

        print(f"Мода металличности: {mode:.3f}")
        print(f"95% доверительный интервал: [{lower_bound:.3f}, {upper_bound:.3f}]")
        # 68 % interval
        lower_bound_68 = x_grid[np.searchsorted(cdf, 0.16)]  # 16-й процентиль
        upper_bound_68 = x_grid[np.searchsorted(cdf, 0.84)]  # 84-й процентиль

        error = (upper_bound_68 - lower_bound_68) / 2
        print(f"Средняя ошибка металличности (1σ): {error:.3f}")


def teff_analysis(pd_data: pd.DataFrame, save=False, object="star"):
    wavelenght = pd_data["wave_center"].values
    wavelenght = [float(wavelenght[x]) for x in range(len(wavelenght))]
    errors = pd_data["chi_squared"].values
    errors = [float(errors[x]) for x in range(len(errors))]

    if np.argwhere(pd_data.columns.values == "Teff") != -1:  # Check the pyright!
        column_data = pd_data["Teff"].values
        column_data = [float(column_data[x]) for x in range(len(column_data))]
        column_data_median = np.median(column_data)
        with plt.style.context("science"):
            fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            fig.suptitle(r"$T_{eff}$ estimation of " + object)
            ax[0].set_title(r"$T_{eff}$ variation")
            ax[0].scatter(wavelenght, column_data, color="black", alpha=0.9)
            ax[0].errorbar(wavelenght, column_data, errors, marker="o", ls=" ")
            ax[0].set_xlim((5000, 7000))
            # draw a medion line
            ax[0].plot(
                (wavelenght[0], wavelenght[-1]),
                (column_data_median, column_data_median),
                ls="dashed",
                color="crimson",
                lw=2,
                label=r"median $T_{eff}$ value",
            )
            ax[0].set_xlabel(r"Wavelegth, \AA")
            ax[1].set_ylabel(r"$T_{eff}$, K")
            print(f"Median teff: {column_data_median}")

            lower = np.percentile(column_data, 5, axis=0)
            upper = np.percentile(column_data, 95, axis=0)
            lower_p = np.percentile(column_data, 16)
            upper_p = np.percentile(column_data, 84)

            print(
                f"percentile is \n lower:{lower_p} \n upper: {upper_p} \n std: {np.std(column_data)}"
            )
            ax[0].fill_between(
                wavelenght, lower, upper, color="blue", alpha=0.2, label=r"$3\sigma$"
            )
            ax[0].fill_between(
                wavelenght,
                lower_p,
                upper_p,
                color="navy",
                alpha=0.7,
                label=r"$\sigma$",
            )
            ax[1].set_title(r"$T_{eff}$ histogram")
            ax[1].hist(
                column_data, bins=15, orientation="horizontal", color="black", alpha=0.7
            )
            ax[1].set_xlabel("N")
            ax[0].legend()
            plt.tight_layout()
            if save:
                plt.savefig("teff_estimation.pdf", dpi=300)
            else:
                plt.show()


def gaussian(Z, A, mu, sigma):
    return A * np.exp(-((Z - mu) ** 2) / (2 * sigma**2))


def gauss_metall(pd_data: pd.DataFrame, ratio: str = "Fe_H"):
    feh_values = pd_data[ratio].to_numpy(float)
    chi2_values = pd_data["chi_squared"].to_numpy(float)
    sorted_idx = np.argsort(feh_values)
    feh_values = feh_values[sorted_idx]
    chi2_values = chi2_values[sorted_idx]
    # Найдём минимум Chi^2
    min_index = np.argmin(chi2_values)
    feh_best = feh_values[min_index]
    chi2_min = chi2_values[min_index]

    # Оценим погрешность по критерию Delta Chi^2 = 1
    threshold = chi2_min + 1

    # Определим границы неопределенности
    feh_lower = np.max(
        feh_values[feh_values < feh_best][
            chi2_values[feh_values < feh_best] <= threshold
        ],
        initial=feh_best,
    )
    feh_upper = np.min(
        feh_values[feh_values > feh_best][
            chi2_values[feh_values > feh_best] <= threshold
        ],
        initial=feh_best,
    )

    # Вывод результатов
    print(f"Оптимальное значение [Fe/H]: {feh_best:.3f}")
    print(
        f"Неопределенность: -{feh_best - feh_lower:.3f} / +{feh_upper - feh_best:.3f}"
    )

    # Визуализация
    plt.plot(feh_values, chi2_values, "o-", label="Chi^2")
    plt.axhline(threshold, linestyle="--", color="gray", label=r"$\chi^2_{min} + 1$")
    plt.axvline(
        feh_best, linestyle="--", color="red", label=f"Best [Fe/H] = {feh_best:.3f}"
    )
    plt.fill_betweenx(
        [min(chi2_values), max(chi2_values)],
        feh_lower,
        feh_upper,
        color="gray",
        alpha=0.3,
    )
    plt.xlabel("[Fe/H]")
    plt.ylabel(r"$\chi^2$")
    plt.legend()
    plt.show()


def velocity_dispersion(pd_data: pd.DataFrame, ratio: str = "Fe_H"):
    velocity = pd_data["Doppler_Shift_add_to_RV"].values
    velocity = [float(velocity[x]) for x in range(len(velocity))]
    metallicty = pd_data[ratio].values
    metallicty = [float(metallicty[x]) for x in range(len(metallicty))]
    chi = pd_data["chi_squared"].to_numpy(float)

    with plt.style.context("science"):
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        ax.set_title(r"Velocity dispersion of FeI on IRAS 07430+1115")
        ax.set_ylabel(r"$\chi^{2}$")
        ax.set_xlabel(r"Metallicity, [Fe/H]")
        ax.set_ylim((0, 20))

        density = ax.scatter(metallicty, chi, c=velocity)
        plt.colorbar(density, label="Velocity dispersion")
        plt.show()


def energy_dispersion(
    pd_data: pd.DataFrame, ratio: str, path2vald: str, element_name: str
):
    metallicty = pd_data[ratio].values
    metallicty = np.array([float(metallicty[x]) for x in range(len(metallicty))])
    chi = pd_data["chi_squared"].to_numpy(float)
    centers = pd_data["wave_center"].astype(float)
    energy_arr = extract_enegry(path2vald, element_name, centers_in_linemask=centers)
    energy_arr = np.array(energy_arr)
    lc = energy_arr[:, 0]
    energy = energy_arr[:, 1]
    idx = energy_arr[:, 2].astype(int)
    metallicty = metallicty[idx]
    chi = chi[idx]

    with plt.style.context("science"):
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        ax.set_title(r"Energy dispersion of FeI on IRAS 07430+1115")
        ax.set_ylabel(r"$\chi^{2}$")
        ax.set_xlabel(r"Metallicity, [Fe/H]")
        ax.set_ylim((0, 20))

        density = ax.scatter(metallicty, chi, c=energy)
        plt.colorbar(density, label="Velocity dispersion")
        plt.show()


def wavelenght_group(pd_data: pd.DataFrame, ratio: str):
    metallicty = pd_data[ratio].values
    metallicty = np.array([float(metallicty[x]) for x in range(len(metallicty))])
    chi = pd_data["chi_squared"].to_numpy(float)
    centers = pd_data["wave_center"].astype(float)

    with plt.style.context("science"):
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        ax.set_title(r"Wavelenght scatter of C I on IRAS 07430+1115")
        ax.set_ylabel(r"$\chi^{2}$")
        ax.set_xlabel(r"Metallicity, [Fe/H]")
        ax.set_ylim((0, 20))

        density = ax.scatter(metallicty, chi, c=centers)
        plt.colorbar(density, label="Wavelenght")
        plt.show()


def wave2velocity(wavelength: float, rest_wavelength: float) -> float:
    z = (wavelength - rest_wavelength) / wavelength
    c = 2.998e8  # m/s
    return float(z * c)


def line_combiner(spectrum: np.ndarray, linelist: np.ndarray):
    lines_cut = []
    for line in linelist:
        line_data = np.where((spectrum[:, 0] >= line[1]) & (spectrum[:, 0] <= line[2]))
        lines_cut.append(spectrum[line_data])

    velocity_cut = []
    number_of_line = 0
    for line in lines_cut:
        line_velocity = []
        for element in line:
            line_velocity.append(wave2velocity(element[0], linelist[number_of_line][0]))
        number_of_line += 1
        print(number_of_line)
        velocity_cut.append(line_velocity)

    for i in range(len(velocity_cut)):
        plt.plot(velocity_cut[i], lines_cut[i][:, 1])
        # plt.plot(np.arange(0, len(lines_cut[i][:, 1])), lines_cut[i][:, 1])
    plt.show()


if __name__ == "__main__":
    from config_loader import tsfit_output
    from pathlib import Path

    print(f"TSFitPy output from config: {tsfit_output}")
    tsfit_output = Path(tsfit_output)

    out_1 = "Fe1_4900_0.0"
    out_1 = Path("05113_teff")
    out_2 = Path("2025-05-26-20-37-58_0.8075572190850129_LTE_Fe_1D")

    r = "Fe_H"

    pd_data_1 = get_model_data(tsfit_output / out_1)
    # pd_data_2 = get_model_data(tsfit_output / out_2)
    # line_combiner(spectrum, c_linemask)

    # pd_data_1 = clean_pd(pd_data_1, True, True)
    # pd_data_2 = clean_pd(pd_data_2, True, True)

    # velocity_dispersion(pd_data_1, r)
    # energy_dispersion(pd_data_1, ratio="C_Fe", path2vald="C1data", element_name="'C 1'")
    # wavelenght_group(pd_data_1, r)

    # plot_metall(pd_data_2, ratio=r)
    # plot_metallVS(pd_data_1, pd_data_2, r)
    # plot_metall_KDE(pd_data_1, r)
    # plot_metall_KDE(pd_data_2, r)

    teff_analysis(pd_data_1, object="IRAS Z02229+6208", save=False)
    # plot_ion_balance(pd_data_2)
    # hist_estimation(pd_data_2, 30)
