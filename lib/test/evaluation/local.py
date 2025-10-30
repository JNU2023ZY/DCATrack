from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = '/data/Datasets/got10k/got_10k_data'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = '/data/Datasets/LaSOTBenchmark/'
    settings.network_path = '/data/kangze/MixFormerV2/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.prj_dir = '/data/kangze/MixFormerV2'
    settings.result_plot_path = '/data/kangze/MixFormerV2/test/result_plots'
    settings.results_path = '/data/kangze/MixFormerV2/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/kangze/MixFormerV2'
    settings.segmentation_path = ''
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''
    settings.zoo145_path = '/core/ZOO145'
    settings.animalsot_path = '/core/AnimalSOT'

    return settings

