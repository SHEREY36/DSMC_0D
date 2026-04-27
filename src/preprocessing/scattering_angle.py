import os
import numpy as np
from scipy.optimize import curve_fit

from .data_loader import load_chi_data


def poly4(x, a, b, c, d, e):
    """4th-degree polynomial."""
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def fit_polynomial(chi_vals, bins=100):
    """Fit a 4th-degree polynomial to the histogram of chi values.

    Returns polynomial coefficients [a, b, c, d, e] or None on failure.
    """
    hist, bin_edges = np.histogram(chi_vals, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    try:
        popt, _ = curve_fit(poly4, bin_centers, hist, maxfev=10000)
        return popt
    except RuntimeError:
        return None


def phi_func(alpha, beta=1.0):
    """Inelasticity function: phi(alpha) = 1 - alpha^beta."""
    return 1.0 - alpha**beta


def build_design_matrix(x_data, AR_data, alpha_data, dp_data,
                        K=4, M=2, N=2, beta=1.0):
    """Build design matrix for delta_p model.

    delta_p(x, AR, alpha) = sum_{m,n,k} a_{m,n,k} * AR^m * phi(alpha)^n * x^k

    Returns (X, y) where X is the design matrix and y is the target vector.
    """
    R = len(x_data)
    num_cols = (K + 1) * (M + 1) * (N + 1)
    X = np.zeros((R, num_cols), dtype=float)
    y = np.zeros(R, dtype=float)

    phi_vals = [phi_func(a, beta=beta) for a in alpha_data]

    for r in range(R):
        x_r = x_data[r]
        AR_r = AR_data[r]
        ph_r = phi_vals[r]

        col_index = 0
        for m in range(M + 1):
            for n in range(N + 1):
                for k in range(K + 1):
                    X[r, col_index] = (AR_r**m) * (ph_r**n) * (x_r**k)
                    col_index += 1

        y[r] = dp_data[r]

    return X, y


def build_elastic_design_matrix(x_data, AR_data, p_data, K=4, M=2):
    """Build design matrix for the elastic scattering PDF.

    P_elastic(x, AR) = sum_{m,k} a_{m,k} * AR^m * x^k

    Returns (X, y).
    """
    R = len(x_data)
    num_cols = (K + 1) * (M + 1)
    X = np.zeros((R, num_cols), dtype=float)
    y = np.array(p_data, dtype=float)

    for r in range(R):
        col_index = 0
        for m in range(M + 1):
            for k in range(K + 1):
                X[r, col_index] = (AR_data[r]**m) * (x_data[r]**k)
                col_index += 1

    return X, y


def fit_scattering_models(root_dir, alpha_dirs, ar_dirs, K=4, M=2, N=2,
                          beta=0.5):
    """Fit elastic and inelastic scattering angle polynomial models.

    Reads chi.txt from CTC_data/Alpha/{alpha}/{AR}/ directories.

    Returns (a_elastic_reshaped, a_inelastic_reshaped, M, N, K, beta).
    """
    x_vals = np.linspace(0, 1, 100)

    # Step 1: Fit elastic (alpha=100) polynomials per AR
    elastic_coeffs = {}
    elastic_x_data = []
    elastic_AR_data = []
    elastic_p_data = []

    for ar in ar_dirs:
        path = os.path.join(root_dir, "100", str(ar), "chi.txt")
        chi_data = load_chi_data(path)
        if chi_data is None:
            continue

        coeffs = fit_polynomial(chi_data)
        if coeffs is None:
            continue

        elastic_coeffs[ar] = coeffs
        AR_val = ar / 10.0
        p_vals = poly4(x_vals, *coeffs)
        for i in range(len(x_vals)):
            elastic_x_data.append(x_vals[i])
            elastic_AR_data.append(AR_val)
            elastic_p_data.append(p_vals[i])

    # Fit global elastic model
    X_elastic, y_elastic = build_elastic_design_matrix(
        np.array(elastic_x_data), np.array(elastic_AR_data),
        np.array(elastic_p_data), K=K, M=M
    )
    a_vec_elastic, _, _, _ = np.linalg.lstsq(X_elastic, y_elastic, rcond=None)
    a_elastic_reshaped = a_vec_elastic.reshape((M + 1, K + 1), order='C')

    # Step 2: Fit inelastic corrections (delta_p)
    x_data = []
    AR_data = []
    alpha_data = []
    dp_data = []

    for alpha in alpha_dirs:
        if alpha == 100:
            continue
        for ar in ar_dirs:
            path = os.path.join(root_dir, str(alpha), str(ar), "chi.txt")
            chi_data = load_chi_data(path)
            if chi_data is None:
                continue

            inelastic_coeffs = fit_polynomial(chi_data)
            if inelastic_coeffs is None or ar not in elastic_coeffs:
                continue

            dp_coeffs = [
                inc - elc
                for inc, elc in zip(inelastic_coeffs, elastic_coeffs[ar])
            ]

            AR_val = ar / 10.0
            alpha_val = alpha / 100.0
            dp_vals = poly4(x_vals, *dp_coeffs)
            for i in range(len(x_vals)):
                x_data.append(x_vals[i])
                AR_data.append(AR_val)
                alpha_data.append(alpha_val)
                dp_data.append(dp_vals[i])

    # Fit global delta_p model
    X, y = build_design_matrix(
        np.array(x_data), np.array(AR_data),
        np.array(alpha_data), np.array(dp_data),
        K=K, M=M, N=N, beta=beta
    )
    a_vec, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a_inelastic_reshaped = a_vec.reshape((M + 1, N + 1, K + 1), order='C')

    return a_elastic_reshaped, a_inelastic_reshaped, M, N, K, beta


def P_elastic(x, AR, a_elastic_reshaped, M=2, K=4):
    """Evaluate the elastic scattering angle PDF."""
    x = np.atleast_1d(x)
    P = np.zeros_like(x, dtype=float)

    for m in range(M + 1):
        for k in range(K + 1):
            coeff = a_elastic_reshaped[m, k]
            P += coeff * (AR**m) * (x**k)

    return P if x.size > 1 else P[0]


def delta_p_model(x, AR, alpha, a_reshaped, M, N, K, beta=1.0):
    """Evaluate the inelastic correction to the scattering angle PDF."""
    x_arr = np.atleast_1d(x)
    phi_val = 1.0 - alpha**beta
    s = np.zeros_like(x_arr, dtype=float)

    for m in range(M + 1):
        for n in range(N + 1):
            for k_ in range(K + 1):
                coeff = a_reshaped[m, n, k_]
                s += coeff * (AR**m) * (phi_val**n) * (x_arr**k_)

    return s if x_arr.size > 1 else s[0]


def p_chi_AR_alpha(chi, AR, alpha, a_elastic_reshaped, a_inelastic_reshaped,
                   M, N, K, beta=0.5):
    """Full scattering angle PDF: P_elastic + delta_p."""
    P_el = P_elastic(chi, AR, a_elastic_reshaped, M=M, K=K)
    delP = delta_p_model(chi, AR, alpha, a_inelastic_reshaped, M, N, K,
                         beta=beta)
    return P_el + delP


def p_chi_mu_model(chi, mu, AR, alpha, a_elastic, a_inelastic,
                   K, M, N, J_el, J_ie, beta=0.5):
    """Evaluate P(chi | mu, AR, alpha) — mu-conditioned scattering PDF.

    mu = |eij · g_hat| is the cosine of the angle between the collision normal
    and the relative velocity direction, computed from NTC geometry.

    a_elastic   shape (M+1, J_el+1, K+1): coefficients of AR^m * mu^j * chi^k
    a_inelastic shape (M+1, N+1, J_ie+1, K+1): coefficients of AR^m * phi^n * mu^j * chi^k
    """
    chi = np.atleast_1d(np.asarray(chi, dtype=float))
    phi = 1.0 - alpha ** beta
    P = np.zeros_like(chi)
    for m in range(M + 1):
        AR_m = AR ** m
        for j in range(J_el + 1):
            mu_j = mu ** j
            for k in range(K + 1):
                P += a_elastic[m, j, k] * AR_m * mu_j * (chi ** k)
    for m in range(M + 1):
        AR_m = AR ** m
        for n in range(N + 1):
            phi_n = phi ** n
            for j in range(J_ie + 1):
                mu_j = mu ** j
                for k in range(K + 1):
                    P += a_inelastic[m, n, j, k] * AR_m * phi_n * mu_j * (chi ** k)
    P = np.maximum(P, 0.0)
    return P if P.size > 1 else float(P[0])
