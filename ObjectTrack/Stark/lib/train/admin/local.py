class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/D/MLProject/ObjectTrack/Stark/tracking'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/D/MLProject/ObjectTrack/Stark/tracking/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/D/MLProject/ObjectTrack/Stark/tracking/pretrained_networks'
        self.lasot_dir = '/D/MLProject/ObjectTrack/Stark/tracking/data/lasot'
        self.got10k_dir = '/D/MLProject/ObjectTrack/Stark/tracking/data/got10k'
        self.lasot_lmdb_dir = '/D/MLProject/ObjectTrack/Stark/tracking/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/D/MLProject/ObjectTrack/Stark/tracking/data/got10k_lmdb'
        self.trackingnet_dir = '/D/MLProject/ObjectTrack/Stark/tracking/data/trackingnet'
        self.trackingnet_lmdb_dir = '/D/MLProject/ObjectTrack/Stark/tracking/data/trackingnet_lmdb'
        self.coco_dir = '/D/MLProject/ObjectTrack/Stark/tracking/data/coco'
        self.coco_lmdb_dir = '/D/MLProject/ObjectTrack/Stark/tracking/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/D/MLProject/ObjectTrack/Stark/tracking/data/vid'
        self.imagenet_lmdb_dir = '/D/MLProject/ObjectTrack/Stark/tracking/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
