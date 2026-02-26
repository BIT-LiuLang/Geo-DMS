from yacs.config import CfgNode as CN

def add_dms_config(cfg):
    """Add DMS-specific configurations to the existing cfg node."""
    _C = cfg

    # =========================================================
    # 1. 定义 MODEL.DMS 部分
    # =========================================================
    if not hasattr(_C.MODEL, "DMS"):
        _C.MODEL.DMS = CN()
        
    _C.MODEL.DMS.ENABLE = False
    _C.MODEL.DMS.INPUT_DIM = 1024
    _C.MODEL.DMS.AGGREGATOR_TYPE = "channel" 
    _C.MODEL.DMS.FUSION_TYPE = "multipole"
    _C.MODEL.DMS.NUM_SCALES = 3
    _C.MODEL.DMS.USE_POSE = True
    _C.MODEL.BACKBONE.NUM_OUT_LAYERS = 4
    
    _C.MODEL.DMS.DEPTH = 2
    _C.MODEL.DMS.DROPOUT = 0.1

    # 任务定义
    _C.MODEL.DMS.TASKS = [
        ("emotion", 7),       
        ("drowsy", 2),        
        ("distraction", 10)    
    ]

    # 【修复点】新增 LOSS_WEIGHTS 节点定义
    # 这样 YAML 文件里就可以写 LOSS_WEIGHTS: ... 了
    _C.MODEL.DMS.LOSS_WEIGHTS = CN()
    _C.MODEL.DMS.LOSS_WEIGHTS.EMOTION = 1.0
    _C.MODEL.DMS.LOSS_WEIGHTS.DROWSY = 1.0
    _C.MODEL.DMS.LOSS_WEIGHTS.DISTRACTION = 1.0

    # =========================================================
    # 2. 定义 DATASETS 部分
    # =========================================================
    if not hasattr(_C, "DATASETS"):
        _C.DATASETS = CN()

    _C.DATASETS.TRAIN_DMS = ("affectnet",)
    _C.DATASETS.VAL_DMS = ("affectnet",)