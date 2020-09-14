# This is based on the example script provided in the bilby documentation.

import numpy as np
import bilby

# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 8.
sampling_frequency = 4096.

# Specify the output directory and the name of the simulation.
outdir = '.'
label = 'injection'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(88170235)

# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
injection_parameters = dict(
    mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=1500., theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=100000000.02, ra=1.375, dec=-1.2108)

# Fixed arguments passed into the source model
waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=20., minimum_frequency=20.)

# Create the waveform_generator using a LAL BinaryBlackHole source function
# the generator will convert all the parameters
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments)

# Set up interferometers. These default to their design sensitivity
ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])

# Insert noise
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 6)

# Inject signal
ifos.inject_signal(waveform_generator=waveform_generator,
                   parameters=injection_parameters
                   )

# Save strain data
ifos.save_data(outdir, label)

# Start with standard BBH priors. Modify certain priors.
priors = bilby.gw.prior.BBHPriorDict()
priors['mass_1'] = bilby.prior.Uniform(
    minimum=10, maximum=80, name='mass_1', latex_label='$m_1$',
    unit='$M_{\\odot}$', boundary=None)
priors['mass_2'] = bilby.prior.Uniform(
    minimum=10, maximum=80, name='mass_2', latex_label='$m_2$',
    unit='$M_{\\odot}$', boundary=None)
priors['a_1'] = bilby.prior.Uniform(
    minimum=0, maximum=0.99, name='a_1', latex_label='$a_1$',
    unit=None, boundary='reflective')
priors['a_2'] = bilby.prior.Uniform(
    minimum=0, maximum=0.99, name='a_2', latex_label='$a_2$',
    unit=None, boundary='reflective')
priors['luminosity_distance'].maximum = 4000.0
priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 0.1,
    maximum=injection_parameters['geocent_time'] + 0.1,
    name='geocent_time', latex_label='$t_c$', unit='$s$')

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveoform generator, as well the priors.
# The explicit time, distance, and phase marginalizations are turned on to
# improve convergence, and the parameters are recovered by the conversion
# function.
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator, priors=priors,
    distance_marginalization=True, phase_marginalization=True,
    time_marginalization=True)

# Run sampler. In this case we're going to use the `cpnest` sampler Note that
# the maxmcmc parameter is increased so that between each iteration of the
# nested sampler approach, the walkers will move further using an mcmc
# approach, searching the full parameter space. The conversion function will
# determine the distance, phase and coalescence time posteriors in post
# processing.
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty',
    injection_parameters=injection_parameters, outdir=outdir,
    label=label, walks=5,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    plot=False)
