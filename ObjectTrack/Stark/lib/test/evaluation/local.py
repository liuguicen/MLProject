from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/E/dataset/ObjectTracking/got10k_lmdb'
    settings.got10k_path = '/E/dataset/ObjectTracking/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.posetrack2018_path = "/E/dataset/PoseTrack/PoseTrack2018/images/val"
    settings.lasot_lmdb_path = '/E/dataset/ObjectTracking/lasot_lmdb'
    settings.lasot_path = '/E/dataset/ObjectTracking/lasot'
    settings.network_path = '/D/MLProject/ObjectTrack/Stark/tracking/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/E/dataset/ObjectTracking/nfs'
    settings.otb_path = '/E/dataset/ObjectTracking/OTB2015'
    settings.prj_dir = '/D/MLProject/ObjectTrack/Stark'
    settings.result_plot_path = '/D/MLProject/ObjectTrack/Stark/tracking/test/result_plots'
    settings.results_path = '/D/MLProject/ObjectTrack/Stark/tracking/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/D/MLProject/ObjectTrack/Stark/tracking'
    settings.segmentation_path = '/D/MLProject/ObjectTrack/Stark/tracking/test/segmentation_results'
    settings.tc128_path = '/E/dataset/ObjectTracking/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/E/dataset/ObjectTracking/trackingNet'
    settings.uav_path = '/E/dataset/ObjectTracking/UAV123'
    settings.vot_path = '/E/dataset/ObjectTracking/VOT2019'
    settings.youtubevos_dir = ''


    return settings

