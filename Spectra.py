# -*- coding: utf-8 -*-
"""
spectra.py  (Spyder-ready)

Generates Monte Carlo EIS spectra :
- Creates a single true spectrum Z_true over the frequency grid
- Generates N noisy spectra (each realization is a full spectrum)
- Saves dataset to an .npz file for the fitting script
- Produces and SAVES figures (Nyquist + Bode magnitude/phase) so you can inspect and keep them

Run in Spyder (F5). Outputs go to ./results_spectra/
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# =========================
# USER SETTINGS
# =========================
SEED = 42

# Frequencies : 20 log-spaced points from 0.01 to 1000 Hz
N_FREQ = 20
F_MIN = 1e-2
F_MAX = 1e3

# Monte Carlo
N_SPECTRA = 500
SIGMA = 1e-4  # complex noise std (Ohm); Re and Im each get SIGMA/sqrt(2)

# True parameters 
# theta = [R0, R1, Q1, a1, R2, Q2, a2, Qw, L]
THETA_TRUE = np.array([0.03, 0.02, 1.0, 0.9, 0.025, 25.0, 0.8, 200.0, 0.0], dtype=float)

# Output folder
OUT_DIR = "results_spectra"
DATA_FILENAME = "spectra_mc.npz"

# Figures
SAVE_FIGS = True
SHOW_FIGS = True
PLOT_MAX_SPECTRA_IN_CLOUD = 200  # for readability


# =========================
# MODEL 2 
# =========================
def model2_impedance(freq_hz, theta):
    R0, R1, Q1, a1, R2, Q2, a2, Qw, L = theta
    w = 2.0 * np.pi * freq_hz
    s = 1j * w

    Z_R0 = R0
    Z_L  = s * L
    Z_Ra1 = R1 / (1.0 + R1 * Q1 * (s ** a1))
    Z_Ra2 = R2 / (1.0 + R2 * Q2 * (s ** a2))
    Z_W   = 1.0 / (Qw * (s ** 0.5))  # Warburg exponent fixed 0.5

    return Z_R0 + Z_L + Z_Ra1 + Z_Ra2 + Z_W


# =========================
# MAIN
# =========================
def main():
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Frequency grid
    freqs = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_FREQ)

    # True spectrum
    Z_true = np.array([model2_impedance(f, THETA_TRUE) for f in freqs], dtype=np.complex128)

    # Monte Carlo noise (circularly symmetric complex Gaussian)
    sigma_part = SIGMA / np.sqrt(2.0)
    noise_r = np.random.normal(0.0, sigma_part, size=(N_SPECTRA, N_FREQ))
    noise_i = np.random.normal(0.0, sigma_part, size=(N_SPECTRA, N_FREQ))

    Z_noisy = Z_true[None, :] + (noise_r + 1j * noise_i)  # shape (N_SPECTRA, N_FREQ)

    # Save dataset for fitting.py
    out_path = os.path.join(OUT_DIR, DATA_FILENAME)
    np.savez_compressed(
        out_path,
        freqs=freqs,
        Z_true=Z_true,
        Z_noisy=Z_noisy,
        theta_true=THETA_TRUE,
        sigma=SIGMA,
        seed=SEED,
    )

    print("Saved dataset:")
    print(" ", out_path)
    print("Shapes:")
    print("  freqs   :", freqs.shape)
    print("  Z_true  :", Z_true.shape)
    print("  Z_noisy :", Z_noisy.shape)
    print("Params (theta_true):", THETA_TRUE.tolist())
    print("Noise sigma (complex):", SIGMA, "Ohm")

    # =========================
    # FIGURE 1: Nyquist (true + noisy cloud)
    # =========================
    k = min(N_SPECTRA, PLOT_MAX_SPECTRA_IN_CLOUD)

    plt.figure(figsize=(7.5, 6))
    plt.plot(Z_true.real, -Z_true.imag, "k-", lw=2.5, label="True spectrum")

    # Scatter a subset of points from first k spectra (flattened)
    plt.scatter(
        Z_noisy[:k, :].real.ravel(),
        (-Z_noisy[:k, :].imag).ravel(),
        s=10,
        alpha=0.25,
        label=f"Noisy points (first {k} spectra)",
    )

    plt.xlabel("Re(Z) [Ω]")
    plt.ylabel("-Im(Z) [Ω]")
    plt.title("Monte Carlo EIS Spectra (Model 2) — Nyquist")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    if SAVE_FIGS:
        fig1 = os.path.join(OUT_DIR, "nyquist_mc.png")
        plt.savefig(fig1, dpi=300)
        print("Saved figure:", fig1)
    if SHOW_FIGS:
        plt.show()
    else:
        plt.close()

    # =========================
    # FIGURE 2: Bode magnitude (true + envelope of noisy)
    # =========================
    mag_true = np.abs(Z_true)
    mag_noisy = np.abs(Z_noisy)

    plt.figure(figsize=(7.5, 5.5))
    plt.semilogx(freqs, mag_true, "k-", lw=2.5, label="True |Z|")

    # show 5–95% band for noisy spectra
    p5 = np.percentile(mag_noisy, 5, axis=0)
    p95 = np.percentile(mag_noisy, 95, axis=0)
    plt.fill_between(freqs, p5, p95, alpha=0.25, label="Noisy |Z| band (5–95%)")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|Z| [Ω]")
    plt.title("Monte Carlo EIS Spectra (Model 2) — Bode Magnitude")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if SAVE_FIGS:
        fig2 = os.path.join(OUT_DIR, "bode_magnitude_mc.png")
        plt.savefig(fig2, dpi=300)
        print("Saved figure:", fig2)
    if SHOW_FIGS:
        plt.show()
    else:
        plt.close()

    # =========================
    # FIGURE 3: Bode phase (true + envelope of noisy)
    # =========================
    phase_true = np.angle(Z_true, deg=True)
    phase_noisy = np.angle(Z_noisy, deg=True)

    plt.figure(figsize=(7.5, 5.5))
    plt.semilogx(freqs, phase_true, "k-", lw=2.5, label="True phase(Z)")

    p5p = np.percentile(phase_noisy, 5, axis=0)
    p95p = np.percentile(phase_noisy, 95, axis=0)
    plt.fill_between(freqs, p5p, p95p, alpha=0.25, label="Noisy phase band (5–95%)")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase(Z) [deg]")
    plt.title("Monte Carlo EIS Spectra (Model 2) — Bode Phase")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if SAVE_FIGS:
        fig3 = os.path.join(OUT_DIR, "bode_phase_mc.png")
        plt.savefig(fig3, dpi=300)
        print("Saved figure:", fig3)
    if SHOW_FIGS:
        plt.show()
    else:
        plt.close()

    print("\nDone. Next: run fitting.py using the saved .npz dataset.")


if __name__ == "__main__":
    main()

