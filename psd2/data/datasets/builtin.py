"""Registration of the standard person-search datasets.

Only the original CUHK-SYSU and PRW splits are registered here.  Derived
degradation protocols are intentionally excluded from the automatic catalog.
"""

import os

from psd2.evaluation import evaluator as _evaluator  # noqa: F401

from .. import mappers
from ..catalog import DatasetCatalog, MapperCatalog, MetadataCatalog
from .cuhk_sysu import load_cuhk_sysu
from .cuhk_sysu_tbps import load_cuhk_sysu_tbps
from .prw import load_prw
from .prw_tbps import load_prw_tbps


def _register(name, loader, mapper, evaluator_type=None):
    DatasetCatalog.register(name, loader)
    MapperCatalog.register(name, mapper)
    if evaluator_type is not None:
        MetadataCatalog.get(name).set(evaluator_type=evaluator_type)


def register_cuhk_sysu_all(dataset_dir):
    def register(name_suffix, load_subset, evaluator_type=None):
        def loader(dataset_dir=dataset_dir, subset=load_subset):
            return load_cuhk_sysu(dataset_dir, subset)

        _register(
            "CUHK-SYSU_" + name_suffix,
            loader,
            mappers.CuhksysuMapper,
            evaluator_type,
        )

    register("Train", "Train")
    register("Gallery", "Gallery", "det")
    register("GalleryGT", "Gallery", "det")
    register("TestG50", "TestG50", "query")
    register("TestG50GT", "TestG50", "query")
    register("TestG100", "TestG100", "query")
    register("TestG100GT", "TestG100", "query")
    register("TestG4000", "TestG4000", "query")
    register("TestG4000GT", "TestG4000", "query")


def register_cuhk_sysu_tbps_all(dataset_dir):
    def register(name_suffix, load_subset, evaluator_type=None):
        def loader(dataset_dir=dataset_dir, subset=load_subset):
            return load_cuhk_sysu_tbps(dataset_dir, subset)

        _register(
            "CUHK-SYSU_TBPS_" + name_suffix,
            loader,
            mappers.CuhksysuTBPSMapper,
            evaluator_type,
        )

    register("Train", "Train")
    register("Gallery", "Gallery", "det")
    register("GalleryGT", "Gallery", "det")
    register("TestG50", "TestG50", "query")
    register("TestG50GT", "TestG50", "query")
    register("TestG100", "TestG100", "query")
    register("TestG100GT", "TestG100", "query")
    register("TestG500", "TestG500", "query")
    register("TestG1000", "TestG1000", "query")
    register("TestG2000", "TestG2000", "query")
    register("TestG4000", "TestG4000", "query")
    register("TestG4000GT", "TestG4000", "query")


def register_prw_all(dataset_dir):
    def register(name_suffix, load_subset, evaluator_type=None):
        def loader(dataset_dir=dataset_dir, subset=load_subset):
            return load_prw(dataset_dir, subset)

        _register("PRW_" + name_suffix, loader, mappers.PrwMapper, evaluator_type)

    register("Train", "Train")
    register("Query", "Query", "query")
    register("QueryGT", "Query", "query")
    register("Gallery", "Gallery", "det")
    register("GalleryGT", "Gallery", "det")


def register_prw_tbps_all(dataset_dir):
    def register(name_suffix, load_subset, evaluator_type=None):
        def loader(dataset_dir=dataset_dir, subset=load_subset):
            return load_prw_tbps(dataset_dir, subset)

        _register(
            "PRW_TBPS_" + name_suffix,
            loader,
            mappers.PrwTBPSMapper,
            evaluator_type,
        )

    register("Train", "Train")
    register("TrainXi", "TrainXi")
    register("Query", "Query", "query")
    register("QueryGT", "Query", "query")
    register("Gallery", "Gallery", "det")
    register("GalleryGT", "Gallery", "det")


_DATA_ROOT = "Data"
register_cuhk_sysu_all(os.path.join(_DATA_ROOT, "cuhk_sysu"))
register_cuhk_sysu_tbps_all(os.path.join(_DATA_ROOT, "cuhk_sysu"))
register_prw_all(os.path.join(_DATA_ROOT, "PRW"))
register_prw_tbps_all(os.path.join(_DATA_ROOT, "PRW"))
