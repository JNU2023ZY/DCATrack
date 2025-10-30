class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/kangze/MixFormerV2'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/kangze/MixFormerV2/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data/kangze/MixFormerV2/pretrained_networks'
        self.lasot_dir = '/data/Datasets/LaSOTBenchmark/'
        self.got10k_dir = '/data/Datasets/got10k/got_10k_data/train'
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_dir = '/data/TrackingNet/'
        self.trackingnet_lmdb_dir = ''
        self.coco_dir = '/data/Datasets/coco/'
        self.coco_lmdb_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenet_lmdb_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
