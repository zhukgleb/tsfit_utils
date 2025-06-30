import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from math import floor, ceil
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm 


def calculate_grid(num_of_models: int) -> list:
    if num_of_models**0.5 == floor(num_of_models**0.5):
        return [num_of_models**0.5, num_of_models**0.5]

    else:
        pass

    rows, cols = 0, 0
    return [rows, cols]


def teff_analysis(pd_data: pd.DataFrame | list, save=False, object="star"):
    if type(pd_data) is list and len(pd_data[0]) > 1:
        parsed_data = pd_data[0]
        parsed_names = pd_data[1]
        model_data = pd_data[2]
        s_num = len(parsed_names)
        assert s_num == len(parsed_data), "Data parsed not correct."

        # plot part
        with plt.style.context("science"):
            # x, y = calculate_grid(s_num)

            chi_values = np.concatenate([d["chi_squared"].values.astype(np.float64) for d in parsed_data])
            # norm = LogNorm(vmin=np.min(chi_values), vmax=np.max(chi_values))
            norm = Normalize(vmin=np.min(chi_values), vmax=np.max(chi_values))
            cmap = plt.get_cmap('hot')
                

            mean_teff = []
            mad_teff = []
            std_teff = []
            
            n_rows = 3  
            n_cols = 4  

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
            fig.tight_layout(pad=5.0) 

            if n_rows > 1 and n_cols > 1:
                axes = axes.ravel()
            else:
                axes = [axes] 

            for x in range(min(len(parsed_data), n_rows * n_cols)):
                wavelength = parsed_data[x]["wave_center"].values.astype(np.float64)
                teff = parsed_data[x]["Teff"].values.astype(np.float64)
                teff_error = parsed_data[x]["Teff_error"].values.astype(np.float64)
                chi_2 = parsed_data[x]["chi_squared"].values.astype(np.float64)
                
                sc = axes[x].scatter(wavelength, teff, c=chi_2, cmap=cmap, norm=norm, s=50)
                
                axes[x].errorbar(wavelength, teff, yerr=teff_error, fmt='none', 
                                ecolor='gray', alpha=0.5, capsize=3)
                
                axes[x].set_xlabel('Wavelength')
                axes[x].set_ylabel('Teff')
                axes[x].set_title(f'{model_data[x][0]}, {model_data[x][1]}, {model_data[x][2]}, {x+1}')
                axes[x].grid(True, alpha=0.3)

       
            fig.subplots_adjust(right=0.85)  

            cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  
            cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
            cbar.set_label('chi squared value', rotation=270, labelpad=15)

            plt.show() 
            # for graph in range(len(parsed_data)):
            #     wavelenght = parsed_data[graph]["wave_center"].values.astype(np.float64)
            #     teff = parsed_data[graph]["Teff"].values.astype(np.float64)
            #     teff_error = parsed_data[graph]["Teff_error"].values.astype(np.float64)
            #     chi_2 = parsed_data[graph]["chi_squared"].values.astype(np.float64)
            #     mean = np.mean(teff)
            #     mad = median_abs_deviation(teff)
            #     std = np.std(teff)
            #     print(f"Mean {mean} \n MAD {mad} \n STD {std}")
            #     mean_teff.append(mean)
            #     mad_teff.append(mad)
            #     std_teff.append(std)

            #     # ax[graph].scatter(wavelenght, chi_2)
            #     ax[graph].errorbar(
            #         wavelenght,
            #         teff,
            #         teff_error,
            #         marker="o",
            #         ls=" ",
            #         color="black",
            #         alpha=0.9,
            #     )
            #     ax[graph].set_xlim((5000, 7000))

    else:
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
                    wavelenght,
                    lower,
                    upper,
                    color="blue",
                    alpha=0.2,
                    label=r"$3\sigma$",
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
                    column_data,
                    bins=15,
                    orientation="horizontal",
                    color="black",
                    alpha=0.7,
                )
                ax[1].set_xlabel("N")
                ax[0].legend()
                plt.tight_layout()
                if save:
                    plt.savefig("teff_estimation.pdf", dpi=300)
                else:
                    plt.show()
