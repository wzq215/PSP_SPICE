import spiceypy as spice

# Print out the toolkit version
spice.tkvrsn("TOOLKIT")
spice.kclear()
# Needed for leap seconds
spice.furnsh('kernels/naif0012.tls')  # !!! for OSX
# Needed for Earth
spice.furnsh('kernels/de430.bsp')
spice.furnsh('kernels/pck00010.tpc')

# %%
# Needed for PSP
spice.furnsh('kernels/spp_v300.tf')
spice.furnsh('kernels/spp_2018_224_2025_243_RO5_00_nocontact.alp.bc')
spice.furnsh('kernels/spp_001.tf')
spice.furnsh('kernels/spp_dyn_v201.tf')
spice.furnsh('kernels/spp_wispr_v002.ti')
spice.furnsh('kernels/spp_sweap_v100.ti')
spice.furnsh('kernels/spp_sclk_0866.tsc')

# !!! Load ephemeris data needed for specified time range
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20180812_20181008_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20181008_20190120_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20200301_20200505_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20200505_20200705_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20200705_20200802_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20200802_20201016_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20201016_20210101_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20210101_20210226_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20210226_20210325_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20210325_20210525_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20210524_20210723_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20210723_20210904_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20210904_20211104_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20211104_20211217_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20211217_20220329_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20220329_20220620_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20220620_20220725_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20220725_20220923_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20220923_20221030_v001.bsp')
spice.furnsh('All_reconstructed_ephemeris/spp_recon_20221030_20230124_v001.bsp')

AU = 1.49e8  # distance from sun to earth

# spice.furnsh('kernels/spp_nom_20180812_20250831_v039_RO6.bsp')
# Load Attitude
# spice.furnsh('kernels/attitude_short_term_predict/spp_2021_277_2021_298_00.asp.bc')
# spice.furnsh('kernels/attitude_short_term_predict/spp_2021_298_2021_319_00.asp.bc')
# spice.furnsh('kernels/attitude_short_term_predict/spp_2021_319_2021_340_00.asp.bc')
# spice.furnsh('kernels/attitude_short_term_predict/spp_2021_340_2021_361_00.asp.bc')
# spice.furnsh('kernels/attitude_short_term_predict/spp_2021_361_2022_010_00.asp.bc')

# spice.furnsh('kernels/attitude_yearly_history/spp_2021_doy310_att.bc')
# spice.furnsh('kernels/attitude_yearly_history/spp_2020_doy365_att.bc')
# spice.furnsh('kernels/attitude_yearly_history/spp_2020_att.bc')
# spice.furnsh('kernels/spp_2021_297_04.ah.bc')


# %%

spice.furnsh('kernels/solo/solo_ANC_soc-sci-fk_V08.tf')
spice.furnsh('kernels/solo/solo_ANC_soc-orbit-stp_20200210-20301120_247_V1_00229_V01.bsp')

# %%
# spice.furnsh('kernels/sta/ahead_science_09.sclk')
# spice.furnsh('kernels/sta/ahead_2016_152_01.epm.bsp')
# spice.furnsh('kernels/sta/stereo_rtn.tf')
# spice.furnsh('kernels/sta/ahead_2022_062_01.depm.xsp')
# spice.furnsh('kernels/sta/ahead_2022_076_01.depm.xsp')
# spice.furnsh('kernels/sta/ahead_2022_070_06.ah.xc')
# spice.furnsh('kernels/sta/ahead_2022_071_06.ah.xc')
# spice.furnsh('kernels/sta/ahead_2022_072_07.ah.xc')
# spice.furnsh('kernels/sta/ahead_2022_073_06.ah.xc')
# spice.furnsh('kernels/sta/ahead_2022_074_07.ah.xc')
# spice.furnsh('kernels/sta/ahead_2022_075_07.ah.xc')
# spice.furnsh('kernels/sta/ahead_2022_076_06.ah.xc')
# spice.furnsh('kernels/sta/ahead_2022_077_06.ah.xc')
# spice.furnsh('kernels/sta/ahead_2022_078_06.ah.xc')
