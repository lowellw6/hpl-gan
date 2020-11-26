import os.path as osp

PKG_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
DATASET_PATH = osp.join(PKG_ROOT, "datasets")
MODEL_PATH = osp.join(PKG_ROOT, "models")
RESULT_PATH = osp.join(PKG_ROOT, "results")