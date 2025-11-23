from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = '/data_F/datasets/got10k/got_10k_data'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = '/data_F/datasets/LaSOTBenchmark'
    settings.lasot_extension_subset_path = '/data_F/datasets/LaSOT_ext/LaSOT_extension_subset'
    settings.network_path = '/data/kangze/MixFormerV2/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.prj_dir = '/data_F/zhouyong/MixFormerV2-main/'
    settings.result_plot_path = '/data_F/zhouyong/MixFormerV2-main/result_plots/s_6_12_l_online_6_mem_lasot_5.0_+-_score0.80update'
    settings.results_path = '/data_F/zhouyong/MixFormerV2-main/tracking_result/Test_FLOPs_L8'
    #settings.results_path = '/data_F/zhouyong/MixFormerV2-main/tracking_result/s_6_12_distill_t_base_got10k_4.6_exit_0.90_online_end_to_end_out_score_bceloss/'
    #settings.results_path = '/data_F/zhouyong/MixFormerV2-main/tracking_result/s_6_12_distill_t_small_got10k_4.6_exit_0.80_offline_end_to_end'
    #settings.results_path = '/data_F/zhouyong/MixFormerV2-main/tracking_result/Speed_test_mixvit2_2/'
    settings.save_dir = '/data_F/zhouyong/MixFormerV2-main'
    settings.segmentation_path = ''
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/data_F/datasets/TrackingNet/'
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''
    settings.zoo145_path = '/core/ZOO145'
    settings.animalsot_path = '/core/AnimalSOT'

    return settings
