from pathlib import Path
import pandas as pd


class Model:
    def __init__(self, path2model: str | Path):
        self.path2model = (
            Path(path2model) if isinstance(path2model, str) else path2model
        )
        self.path2output = self.path2model / "output"
        self.path2fit = self.path2model / "fitlist.txt"

    def _read_and_parse_file(self, file_path: Path) -> pd.DataFrame:
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return pd.DataFrame()

        data = [line.split() for line in lines]

        header = data[0]
        if header[0].startswith("#"):
            header[0] = header[0][1:]

        num_columns = len(header)
        data_rows = data[1:]

        filtered_data = [row for row in data_rows if len(row) == num_columns]

        return pd.DataFrame(filtered_data, columns=header)

    def get_model_data(self) -> pd.DataFrame:
        return self._read_and_parse_file(self.path2output)

    def get_fitlist(self) -> pd.DataFrame:
        return self._read_and_parse_file(self.path2fit)


if __name__ == "__main__":
    data_path = "/home/epsilon/TSFitPy/output_files/05113_teff/"
    m = Model(data_path)
    # pd_data = m.get_model_data()
    fitlist = m.get_fitlist()
    print(fitlist)
