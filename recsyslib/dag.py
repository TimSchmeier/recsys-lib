from ds_lib.dag import BaseDag
import recsyslib  # NOQA


class Dag(BaseDag):
    def __init__(self):
        super().__init__()

    def get_project(self):
        return "recsyslib"

    def get_version(self):
        return "v{}".format("recsyslib.__version__")
