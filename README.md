Monte Carlo Uncertainty Analysis of Electrochemical Impedance Spectroscopy (EIS)

======================================================================

OVERVIEW

This project implements a complete Monte Carlo simulation pipeline for evaluating the performance of Electrochemical Impedance Spectroscopy (EIS) parameter estimation algorithms. The system generates synthetic EIS spectra with controlled noise, then analyzes how accurately fitting algorithms can recover the true underlying parameters under varying noise conditions.

**Primary Objective:** To develop a robust framework for Equivalent Circuit Model (ECM) parameter estimation that can reliably extract physically meaningful features from EIS data. These estimated parameters can then serve as engineered input features for machine learning models, enabling advanced predictive analytics in electrochemical systems such as battery health monitoring, corrosion analysis, and fuel cell diagnostics.

The implementation is intentionally split into two executable scripts:

spectra.py – synthetic spectra generation

fitting.py – nonlinear fitting and uncertainty analysis

This separation reflects good scientific practice and allows reuse of the
generated datasets.


Specifically, the code verifies that:

nonlinear least-squares fitting converges reliably,

estimated parameters are unbiased,

parameter variability reflects the imposed noise level,

permutation ambiguity between Randles elements is resolved correctly.

This project is a methodological reproduction and validation, not a
general-purpose EIS fitting toolbox.

MODEL DESCRIPTION

The impedance model corresponds to :

Z(jω) = R0 + jωL
+ R1 / (1 + R1 Q1 (jω)^α1)
+ R2 / (1 + R2 Q2 (jω)^α2)
+ 1 / (Qw (jω)^0.5)

The Warburg exponent is fixed to 0.5.
The true parameter vector is taken directly from the synthetic example used in
the paper.

MONTE CARLO SPECTRA GENERATION (spectra.py)

The spectra.py script performs the following steps:

Generates 20 logarithmically spaced frequencies from 0.01 Hz to 1000 Hz

Computes a noise-free reference impedance spectrum using the Model described above

Generates N = 500 noisy spectra, each representing a full EIS realization

Adds circularly symmetric complex Gaussian noise with standard deviation
σ = 1e-4 Ω (real and imaginary parts each receive σ / sqrt(2))

Saves the generated dataset to a compressed .npz file

Produces and saves inspection figures:

Nyquist plot (true spectrum and noisy cloud)

Bode magnitude plot (true curve with 5–95 % noise envelope)

Bode phase plot (true curve with 5–95 % noise envelope)

All outputs are written to the directory "results_spectra/".

PARAMETER FITTING AND UNCERTAINTY ANALYSIS (fitting.py)

Each noisy spectrum is fitted independently using constrained nonlinear
least squares.

GEOMETRIC INITIALIZATION

Initial parameter values are estimated directly from geometric features of the
Nyquist plot:

R0 and L are estimated from the highest-frequency point:

R0 ≈ Re(Z) at highest frequency

L ≈ Im(Z) / ω at highest frequency (if positive)

Two arc minima are identified on the Nyquist plot to separate the two
resistive–CPE branches.

The resistances R1 and R2 are estimated from horizontal distances between
characteristic points on the real axis.

For each arc, the CPE exponent α is estimated using:
α = (4 / π) · arctan( 2 · y_max / R )

where y_max is the maximum imaginary part of the corresponding arc.

The CPE coefficients Q are estimated using:
Q = 1 / [ R · ω_max^α ]

where ω_max is the angular frequency at which the arc reaches its maximum
imaginary value.

The Warburg coefficient Qw is estimated from the low-frequency straight-line
region of the Nyquist plot by fitting a linear relationship and using:
Qw = [ mean( ω^(2α) · ((x − x_D)² + y²) ) ]^(−1/2)

with α fixed to 0.5.

BOUND CONSTRAINTS

The optimization uses physically motivated bounds:

All parameters have lower bound 0.

Upper bounds:

20 × initial value for all R and Q parameters

α ≤ 2 for CPE exponents

L ≤ 2 µH

OPTIMIZATION STRATEGY

Trust-region reflective nonlinear least-squares solver

Multi-start strategy with 1, 5, and 10 randomized initial guesses

Best solution selected based on minimum root-mean-square error (RMSE)

RMSE is computed as:
RMSE = sqrt( mean( |Z_fit − Z_measured|² ) )

PERMUTATION RESOLUTION

To resolve ambiguity between the two CPE branches, parameters are reordered
after fitting using the fractional time constant:

τ = (R · Q)^(1 / α)

The branch with the smaller τ is labeled as the first branch.

RESULTS

For 500 Monte Carlo realizations, the recovered parameter estimates are:

Parameter estimates (mean ± standard error) vs true:

R0: 0.029997861 ± 1.64e-06 (true 0.03)
R1: 0.019997401 ± 8.98e-06 (true 0.02)
Q1: 1.0004098 ± 9.27e-04 (true 1)
a1: 0.89989979 ± 2.22e-04 (true 0.9)
R2: 0.025012444 ± 1.24e-05 (true 0.025)
Q2: 25.00248 ± 1.62e-02 (true 25)
a2: 0.79959015 ± 2.79e-04 (true 0.8)
Qw: 200.05955 ± 5.26e-02 (true 200)
L: 3.85e-09 ± 2.49e-10 (true 0)

These results demonstrate:

near-zero bias for all identifiable parameters,

uncertainty levels consistent with the imposed noise,

expected behavior for weakly identifiable parameters (e.g. inductance).

FILE STRUCTURE

spectra.py
Generates synthetic Monte Carlo EIS spectra and saves inspection plots

fitting.py
Fits the impedance model and performs uncertainty analysis

results_spectra/
Generated datasets and spectral plots

results_fitting/
Fitted parameters, RMSE statistics, and uncertainty plots

README.md
Project description and methodology

HOW TO RUN

Generate spectra:
python spectra.py

Fit spectra and analyze uncertainty:
python fitting.py

Both scripts are Spyder-ready and require no command-line arguments.

SCOPE AND LIMITATIONS

Synthetic data only (no experimental EIS data)

Fixed impedance model and frequency grid

Identifiability analysis beyond time-constant reordering is not included

Intended as a methodological and educational reference

END OF FILE