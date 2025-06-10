import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def draw_H_profile(pd_data: pd.DataFrame | list, save=False, object="star"):
    pass


def test_profile():
    path = "/home/epsilon/TSFitPy/output_files/2025-06-09-19-24-22_0.2907879679521571_LTE_H_1D/"
    normal = path + "result_spectrum_iras05113_hand.txt.spec"
    conv = path + "result_spectrum_iras05113_hand.txt_convolved.spec"

    data_normal = np.genfromtxt(normal)
    data_conv = np.genfromtxt(conv)

    plt.plot(data_normal[:, 0], data_normal[:, 1])
    plt.plot(data_conv[:, 0], data_conv[:, 1])
    plt.show()


if __name__ == "__main__":
    test_profile()