import configparser
import pathlib

config = configparser.ConfigParser()
config.read("settings.ini")
tsfit_path = config["TSFit"]["tsfit_path"]
tsfit_output = config["TSFit"]["tsfit_output"]


def tsfit_pathes():
    if "inside":
        # parent up, operator "/" for join path
        tsfit_output = pathlib.Path().resolve().parent / "output_files"
    print(tsfit_output)

    return {"ts_out": tsfit_output}


if __name__ == "__main__":
    tsfit_pathes()
