from langmuirProbe import LangmuirProbe

if __name__ == "__main__":

    #  path to input data:
    file_name = "Te31x31plane_Tsweep200us_@RF=300V_60G_1mTorr_smallAntIn_5shotAvg_50ohm.hdf5"

    #  initialize the analysis code (file_path, resistance, attenuation/gain_factor, probe diameter, AMU,
    #  adjustable parameter to set cutoff for fit quality):
    probe_sweep = LangmuirProbe(te_plane_1_file, 50, 0.5, 0.5, 39.948, r_squared_cut=0.93)

    # Some plots we can make:

    probe_sweep.plot_vp_image('test.png')
    probe_sweep.plot_te_image('test.png')
    probe_sweep.plot_I_sat_image('test.png')
    probe_sweep.plot_n_e_image('test.png')
    probe_sweep.plot_line_out_n_e('test.png')
    f, a = probe_sweep.plot_line_out_t_e('test.png')


