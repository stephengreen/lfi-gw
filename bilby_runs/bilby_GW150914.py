#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on GW150914

This example estimates all 15 parameters of the binary black hole system using
commonly used prior distributions. This will take several hours to run. The
data is obtained using gwpy, see [1] for information on how to access data on
the LIGO Data Grid instead.

[1] https://gwpy.github.io/docs/stable/timeseries/remote-access.html

Main modifications (for consistency with LFI analysis):
    * More precise trigger_time
    * duration -> 8 s
    * psd_duration -> 1024 s
    * Prior file GW150914.prior modified
    * reference_frequency -> 20 Hz
    * set minimum_frequency -> 20 Hz
    * set maximum_frequency -> 1024 Hz
    * set roll_off for data segment window to 0.4 s. Default is 0.2 s.
      This was causing an issue with a spike around 1000 Hz. Now it is
      consistent with the window function for the PSD estimation.
    * Changed to pycbc PSD estimation. For some reason, the gwpy methods
      were giving very slightly different results.
    * nlive -> 1500, nact -> 10
"""
# from __future__ import division, print_function
import bilby
from gwpy.timeseries import TimeSeries
import pycbc.psd
from scipy.signal import tukey
import numpy as np

logger = bilby.core.utils.logger
outdir = 'GW150914'
label = 'GW150914'

# Data set up
trigger_time = 1126259462.391

roll_off = 0.4  # Roll off duration of tukey window in seconds, default is 0.4s
duration = 8  # Analysis segment duration
post_trigger_duration = 2  # Time between trigger time and end of segment
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

psd_duration = 1024
psd_start_time = start_time - psd_duration
psd_end_time = start_time

# We now use gwpy to obtain analysis and psd data and create the ifo_list
ifo_list = bilby.gw.detector.InterferometerList([])
for det in ["H1", "L1"]:
    logger.info("Downloading analysis data for ifo {}".format(det))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    ifo.strain_data.roll_off = 0.4  # Set this explicitly. Default is 0.2.
    data = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
    ifo.strain_data.set_from_gwpy_timeseries(data)

    logger.info("Downloading psd data for ifo {}".format(det))
    psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time,
                                          cache=True)
    psd_alpha = 2 * roll_off / duration

    # Use pycbc psd routine
    sampling_rate = len(psd_data)/psd_duration
    psd_data_pycbc = psd_data.to_pycbc()
    w = tukey(int(duration * sampling_rate), psd_alpha)
    psd = pycbc.psd.estimate.welch(psd_data_pycbc,
                                   seg_len=int(duration * sampling_rate),
                                   seg_stride=int(duration * sampling_rate),
                                   window=w,
                                   avg_method='median')
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=np.array(psd.sample_frequencies),
        psd_array=np.array(psd))

    # gwpy routine was giving slightly different result from pycbc
    # psd = psd_data.psd(
    #     fftlength=duration,
    #     overlap=0,
    #     window=("tukey", psd_alpha),
    #     method="median"
    # )
    # ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
    #     frequency_array=psd.frequencies.value, psd_array=psd.value)

    ifo_list.append(ifo)

logger.info("Saving data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)

# Save strain data
ifo_list.save_data(outdir, label)

# We now define the prior.
# We have defined our prior distribution in a local file, GW150914.prior
# The prior is printed to the terminal at run-time.
# You can overwrite this using the syntax below in the file,
# or choose a fixed value by just providing a float value as the prior.

# Modified this file as well.
priors = bilby.gw.prior.BBHPriorDict(filename='GW150914.prior')
priors['geocent_time'].minimum = trigger_time - 0.1
priors['geocent_time'].maximum = trigger_time + 0.1

# In this step we define a `waveform_generator`. This is the object which
# creates the frequency-domain strain. In this instance, we are using the
# `lal_binary_black_hole model` source model. We also pass other parameters:
# the waveform approximant and reference frequency and a parameter conversion
# which allows us to sample in chirp mass and ratio rather than component mass
waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments={'waveform_approximant': 'IMRPhenomPv2',
                        'reference_frequency': 20,
                        'minimum_frequency': 20,
                        'maximum_frequency': 1024})

# In this step, we define the likelihood. Here we use the standard likelihood
# function, passing it the data and the waveform generator.
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list, waveform_generator, priors=priors, time_marginalization=True,
    phase_marginalization=True, distance_marginalization=True)

# Finally, we run the sampler. This function takes the likelihood and prior
# along with some options for how to do the sampling and how to save the data
result = bilby.run_sampler(
    likelihood, priors, sampler='dynesty', outdir=outdir, label=label,
    nlive=2000, nact=10, walks=100, n_check_point=10000, check_point_plot=True,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    plot=False)
# result.plot_corner()
