from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union


class Model:
    def __init__(self, path2model: str | Path):
        self.path2model = (
            Path(path2model) if isinstance(path2model, str) else path2model
        )
        self.path2output = self.path2model / "output"
        self.path2fit = self.path2model / "fitlist.txt"
        self._model_anal()

    def _read_and_parse_file(self, file_path: Path) -> pd.DataFrame:
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return pd.DataFrame()

        data = [line.split() for line in lines]

        header = data[0]
        if header[0].startswith("#"):  # for fitlist file case
            header[0] = header[0][1:]

        num_columns = len(header)
        data_rows = data[1:]

        filtered_data = [row for row in data_rows if len(row) == num_columns]

        return pd.DataFrame(filtered_data, columns=header)

    @property
    def model_data(self) -> pd.DataFrame | list[Union[np.ndarray, list]]:
        full_data = self._read_and_parse_file(self.path2output)

        if self.spectra_num > 1:
            print("output file is compicated")
            """
            May be a not so good idea use numpy arrays for this, but....
            it's my code
            """
            n_of_lines = int((len(full_data) / self.spectra_num))
            data_cube = []
            start_index = 0
            end_index = n_of_lines
            try:
                for s_name in range(self.spectra_num):
                    data_cube.append(full_data[start_index:end_index])
                    start_index = end_index
                    end_index = start_index + n_of_lines
            except TypeError:
                print("Output file is broken")
                breakpoint()

            full_data = full_data.to_numpy()
            return [data_cube, self.spectra_names]
        else:
            print("file is simple")
            return full_data

    @property
    def fitlist_data(self) -> pd.DataFrame:
        return self._read_and_parse_file(self.path2fit)

    def _model_anal(self) -> None:
        fitlist = self.fitlist_data
        self.spectra_names = fitlist.values[:, 0]
        self.spectra_num = int(len(self.spectra_names))


if __name__ == "__main__":
    data_path = "/home/epsilon/TSFitPy/output_files/05113_teff/"
    m = Model(data_path)
    d = m.model_data
    print(d[1])
