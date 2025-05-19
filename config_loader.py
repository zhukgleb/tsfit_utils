import configparser
import pathlib

config = configparser.ConfigParser()
config.read("settings.ini")
tsfit_path = config["TSFit"]["tsfit_path"]
tsfit_output = config["TSFit"]["tsfit_output"]
if "inside":
    tsfit_output = pathlib.Path().resolve().parent / "output_files"


def tsfit_pathes():
    return {"ts_out": tsfit_output}


if __name__ == "__main__":
    tsfit_pathes()
