import numpy as np
import pandas as pd
from scipy.stats import invwishart
from numba import njit
from itertools import combinations

########## CONFIG ##########
TICKERS       = ["AAPL", "AMZN", "GOOG", "GOOGL", "MSFT", "TSLA"]
BASE_SENTIMENTS = ["RoBERTa", "BoW", "TBlob", "VADER"] # new
#TV_FEATURES   = ["T_SENT", "HL_SENT"]  # testing both
#TV_FEATURES   = ["TRoBERTa", "HLRoBERTa"]  # initial
TV_FEATURES   = ["T_SENT"]
STAT_FEATURES = ["dlyret", "past_3ret", "past_7ret",
                 "volumeSPX", "dlyretSPX", "VIX"]              # Z : non-time-varying
TARGETS       = ["ret1", "ret7"]

N_ITER  = 2000 # Paper: 5000
BURN_IN = 500  # Paper: 1000
DELTA   = 0.9   # signal-to-noise ratio. Paper: 0.9


#### Loading data
dfAAPL = pd.read_csv("data/training/AAPL_open_open_AVG3.csv", index_col=0, parse_dates=True)
dfAMZN = pd.read_csv("data/training/AMZN_open_open_AVG3.csv", index_col=0, parse_dates=True)
dfGOOG = pd.read_csv("data/training/GOOG_open_open_AVG3.csv", index_col=0, parse_dates=True)
dfGOOGL = pd.read_csv("data/training/GOOGL_open_open_AVG3.csv", index_col=0, parse_dates=True)
dfMSFT = pd.read_csv("data/training/MSFT_open_open_AVG3.csv", index_col=0, parse_dates=True)
dfTSLA = pd.read_csv("data/training/TSLA_open_open_AVG3.csv", index_col=0, parse_dates=True)

#df = {"AAPL": dfAAPL, "AMZN": dfAMZN, "GOOG": dfGOOG, "GOOGL": dfGOOGL, "MSFT": dfMSFT, "TSLA": dfTSLA}
base_df = {"AAPL": dfAAPL, "AMZN": dfAMZN, "GOOG": dfGOOG, "GOOGL": dfGOOGL, "MSFT": dfMSFT, "TSLA": dfTSLA}


### new
SENTIMENT_MAP = {
      "RoBERTa": ("TRoBERTa", "HLRoBERTa"),
      "BoW":     ("TBoW",     "HLBoW"),
      "TBlob":   ("TTBlob",   "HLTBlob"),
      "VADER":   ("TVADER",   "HLVADER"),
  }

def all_non_empty_combos(items):
    return [list(c) for r in range(1, len(items) + 1) for c in combinations(items, r)]

def combo_label(combo):
    return "+".join(combo)

#def add_combo_features(df_in, tv_combo, hl_combo):
#    df_out = df_in.copy()
#    df_out["T_SENT"] = df_out[[SENTIMENT_MAP[s][0] for s in tv_combo]].mean(axis=1)
#    df_out["HL_SENT"] = df_out[[SENTIMENT_MAP[s][1] for s in hl_combo]].mean(axis=1)
#    return df_out

def add_combo_features(df_in, tv_combo):
    df_out = df_in.copy()
    df_out["T_SENT"] = df_out[[SENTIMENT_MAP[s][0] for s in tv_combo]].mean(axis=1)
    return df_out

def add_combo_features_hl(df_in, hl_combo):
    df_out = df_in.copy()
    df_out["HL_SENT"] = df_out[[SENTIMENT_MAP[s][1] for s in hl_combo]].mean(axis=1)
    return df_out

###


########### STEP 1 — BUILD ALIGNED PANEL ##########
def build_panel(df: dict, target: str, max_T: int = None):
    """
    Returns Y (T×M), X (T×M×K), Z (T×M×p) aligned on common trading dates.
    NaNs in sentiment are forward-filled then zero-filled.
    """
    M = len(TICKERS)
    K = len(TV_FEATURES)
    p = len(STAT_FEATURES)

    common_idx = df[TICKERS[0]].index
    for t in TICKERS[1:]:
        common_idx = common_idx.intersection(df[t].index)
    common_idx = common_idx.sort_values()
    if max_T is not None:
        common_idx = common_idx[:max_T]
    T = len(common_idx)

    Y = np.full((T, M), np.nan)
    X = np.full((T, M, K), np.nan)
    Z = np.full((T, M, p), np.nan)

    # Temporary
    def ewma_weights(window_vals, alpha):
        weights = (1 - alpha) ** np.arange(len(window_vals))[::-1]
        return np.sum(weights * window_vals) / np.sum(weights)

    for j, ticker in enumerate(TICKERS):
        dft = df[ticker].reindex(common_idx)
        #dft[TV_FEATURES]   = dft[TV_FEATURES].ffill().fillna(0) # ffill
        #dft[TV_FEATURES] = dft[TV_FEATURES].apply(lambda col: col.dropna().rolling(window=20, min_periods=1).mean().reindex(dft.index, method='ffill')).fillna(0) # Rolling average
        dft[TV_FEATURES] = dft[TV_FEATURES].apply(lambda col: col.dropna().rolling(window=5, min_periods=1).apply(lambda w: ewma_weights(w.values, 0.5)).reindex(dft.index, method='ffill')).fillna(0) # EWMA
        dft[STAT_FEATURES] = dft[STAT_FEATURES].ffill().fillna(0)
        Y[:, j]    = dft[target].values
        X[:, j, :] = dft[TV_FEATURES].values
        Z[:, j, :] = dft[STAT_FEATURES].values

    valid = ~np.isnan(Y).any(axis=1)
    return Y[valid], X[valid], Z[valid], common_idx[valid]


# =============================================================================
# NUMBA-ACCELERATED KERNELS
# These functions are JIT-compiled — called inside run_gibbs.
# Numba does not support scipy, so only numpy ops are used here.
# =============================================================================

@njit(cache=True)
def _forward_filter(r_k, Xk, Omega, W, sigma2_v, delta, T, M):
    """
    Forward Kalman filter for one feature k.
    r_k  : (T, M)  partial residual
    Xk   : (T, M)  feature k values across all stocks
    Returns m (T,M), C (T,M,M), m_pred (T,M), R_pred (T,M,M)
    """
    m      = np.zeros((T, M))
    C      = np.zeros((T, M, M))
    m_pred = np.zeros((T, M))
    R_pred = np.zeros((T, M, M))

    # Diffuse initialisation
    init_scale = sigma2_v / (1.0 - delta + 1e-10)
    for i in range(M):
        for j in range(M):
            R_pred[0, i, j] = init_scale * Omega[i, j]

    for t in range(T):
        # F_t = diag(Xk[t, :])
        # f_t = F_t @ m_pred[t]
        f_t = np.zeros(M)
        for i in range(M):
            f_t[i] = Xk[t, i] * m_pred[t, i]

        # Q_t = F_t @ R_pred[t] @ F_t.T + Omega
        Q_t = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                Q_t[i, j] = Xk[t, i] * R_pred[t, i, j] * Xk[t, j] + Omega[i, j]
        # Symmetrise + regularise
        for i in range(M):
            for j in range(M):
                Q_t[i, j] = 0.5 * (Q_t[i, j] + Q_t[j, i])
            Q_t[i, i] += 1e-8

        # Innovation
        e_t = np.zeros(M)
        for i in range(M):
            e_t[i] = r_k[t, i] - f_t[i]

        # Kalman gain: A_t = R_pred[t] @ F_t.T @ Q_inv
        # First: R_pred[t] @ F_t.T  (F_t is diagonal => F_t.T = F_t)
        RFt = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                RFt[i, j] = R_pred[t, i, j] * Xk[t, j]

        # Q_inv via Gaussian elimination (small M=6, fast)
        Q_inv = np.linalg.inv(Q_t)
        A_t = RFt @ Q_inv   # (M, M)

        # Update: m[t] = m_pred[t] + A_t @ e_t
        Ae = A_t @ e_t
        for i in range(M):
            m[t, i] = m_pred[t, i] + Ae[i]

        # C[t] = R_pred[t] - A_t @ F_t @ R_pred[t]
        # A_t @ F_t: A_t col j * Xk[t,j]
        AFR = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                for l in range(M):
                    AFR[i, j] += A_t[i, l] * Xk[t, l] * R_pred[t, l, j]
        for i in range(M):
            for j in range(M):
                C[t, i, j] = R_pred[t, i, j] - AFR[i, j]
        # Symmetrise + regularise
        for i in range(M):
            for j in range(M):
                C[t, i, j] = 0.5 * (C[t, i, j] + C[t, j, i])
            C[t, i, i] += 1e-8

        # Predict next state
        if t < T - 1:
            for i in range(M):
                m_pred[t+1, i] = m[t, i]
            for i in range(M):
                for j in range(M):
                    R_pred[t+1, i, j] = 0.5 * (C[t, i, j] + C[t, j, i]) + W[i, j]

    return m, C, m_pred, R_pred


@njit(cache=True)
def _backward_sample_means(m, C, m_pred, R_pred, T, M):
    """
    Stabilised backward pass — returns h (T,M) and B (T,M,M).
    Uses m[t+1] instead of sampled beta[t+1] (paper stabilisation).
    Caller draws samples from N(h[t], B[t]).
    """
    h = np.zeros((T, M))
    B = np.zeros((T, M, M))

    # Last step
    for i in range(M):
        h[T-1, i] = m[T-1, i]
    for i in range(M):
        for j in range(M):
            B[T-1, i, j] = C[T-1, i, j]

    B_next = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            B_next[i, j] = C[T-1, i, j]

    for t in range(T-2, -1, -1):
        R_next = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                R_next[i, j] = R_pred[t+1, i, j] + 1e-8 * (1.0 if i == j else 0.0)

        R_next_inv = np.linalg.inv(R_next)

        # CR = C[t] @ R_next_inv
        CR = C[t] @ R_next_inv   # (M, M)

        # h[t] = m[t] + CR @ (m[t+1] - m[t])
        diff = np.zeros(M)
        for i in range(M):
            diff[i] = m[t+1, i] - m[t, i]
        CR_diff = CR @ diff
        for i in range(M):
            h[t, i] = m[t, i] + CR_diff[i]

        # B[t] = C[t] + CR @ B_next @ CR.T - CR @ C[t]
        CR_Bnext = CR @ B_next          # (M, M)
        CR_Bnext_CRt = CR_Bnext @ CR.T  # (M, M)
        CR_Ct = CR @ C[t]               # (M, M)
        for i in range(M):
            for j in range(M):
                B[t, i, j] = C[t, i, j] + CR_Bnext_CRt[i, j] - CR_Ct[i, j]
        # Symmetrise + regularise
        for i in range(M):
            for j in range(M):
                B[t, i, j] = 0.5 * (B[t, i, j] + B[t, j, i])
            B[t, i, i] += 1e-8

        # Carry B[t] as B_next for next iteration
        for i in range(M):
            for j in range(M):
                B_next[i, j] = B[t, i, j]

    return h, B


########## STEP 2 — GIBBS SAMPLER ##########
def run_gibbs(Y, X, Z, n_iter=N_ITER, burn_in=BURN_IN, delta=DELTA, seed=42):
    """
    HD-SURDLM Gibbs sampler — paper-faithful implementation with Numba acceleration.

    Model
    -----
    y_{j,t} = alpha_j + sum_k X_{j,t,k} * beta_{j,t,k} + Z_{j,t} @ gamma_j + u_{j,t}
    beta_{j,t,k} = beta_{j,t-1,k} + v_{j,t,k}

    u_t ~ N_M(0, Omega)   (cross-stock correlated observation errors)
    v_{t,k} ~ N(0, Omega) (system errors share the same covariance structure)

    FFBS is run in full M-dimensional form for each feature k, so that
    cross-stock correlations in Omega propagate into beta sampling.
    The forward filter and backward pass are JIT-compiled via Numba.

    Parameters
    ----------
    Y : (T, M)
    X : (T, M, K)
    Z : (T, M, p)
    delta : signal-to-noise ratio (0.9 per paper)
    """
    rng   = np.random.default_rng(seed)
    T, M  = Y.shape
    K     = X.shape[2]
    p     = Z.shape[2]

    # ---- Initialisation ----
    alpha = np.zeros(M)                 # (M,)
    gamma = np.zeros((M, p))            # (M, p)
    beta  = np.zeros((T, M, K))        # (T, M, K)
    Omega = np.eye(M)                  # (M, M)  cross-stock covariance

    # System noise covariance: Sigma = A @ Omega @ A  with A = sqrt((1-delta)/delta) * I
    # => sigma2_v scalar; W_k = sigma2_v * Omega  for each feature k
    sigma2_v  = (1.0 - delta) / delta  # scalar factor

    # Storage (post burn-in)
    n_samples     = n_iter - burn_in
    alpha_samples = np.zeros((n_samples, M))
    gamma_samples = np.zeros((n_samples, M, p))
    beta_samples  = np.zeros((n_samples, T, M, K))
    Omega_samples = np.zeros((n_samples, M, M))

    def compute_residuals(alpha_, beta_, gamma_):
        """u_{t} = y_t - alpha - X_t * beta_t - Z_t @ gamma   shape (T, M)"""
        sent   = np.einsum('tmk,tmk->tm', X, beta_)      # (T, M)
        static = np.einsum('tmp,mp->tm', Z, gamma_)       # (T, M)
        return Y - alpha_[None, :] - sent - static

    # Warm up Numba JIT on first call (compiles once, cached after)
    print("  [Numba] Compiling JIT kernels on first iteration...")

    for it in range(n_iter):

        Omega_inv = np.linalg.inv(Omega)

        # 1. Sample alpha | rest  →  N_M(mu_alpha, Sigma_alpha)
        #    Full conditional uses Omega for cross-stock correlation
        r_alpha = (Y
                   - np.einsum('tmk,tmk->tm', X, beta)
                   - np.einsum('tmp,mp->tm', Z, gamma))   # (T, M)

        # Sigma_alpha_inv = T * Omega_inv  (flat prior => prior term ~ 0)
        Sigma_alpha = np.linalg.inv(T * Omega_inv)
        mu_alpha    = Sigma_alpha @ (Omega_inv @ r_alpha.sum(axis=0))
        alpha       = rng.multivariate_normal(mu_alpha, Sigma_alpha)

        # 2. Sample gamma | rest  →  N_{M×p}(mu_gamma, Sigma_gamma)
        #    Per-stock GLS using full Omega diagonal (off-diag ignored in
        #    gamma because Z is stock-specific; cross-terms enter via Omega)
        r_gamma = (Y
                   - alpha[None, :]
                   - np.einsum('tmk,tmk->tm', X, beta))   # (T, M)

        for j in range(M):
            Zj  = Z[:, j, :]        # (T, p)
            rj  = r_gamma[:, j]     # (T,)
            # Use full row of Omega_inv for cross-stock weighting
            # Simplified: use diagonal element (exact for independent stocks)
            sig_jj = Omega[j, j]
            Sig_gj_inv = Zj.T @ Zj / sig_jj   # flat prior
            Sig_gj     = np.linalg.inv(Sig_gj_inv + 1e-8 * np.eye(p))
            mu_gj      = Sig_gj @ (Zj.T @ rj) / sig_jj
            gamma[j]   = rng.multivariate_normal(mu_gj, Sig_gj)

        # 3. Sample beta via M-dimensional FFBS  (paper Section III.B)
        #    For each feature k, beta_{:,k} is an (M,)-vector evolving over t.
        #
        #    Observation eq:  r_{t,k} = diag(X_{t,:,k}) @ beta_{t,k} + u_t
        #    State eq:        beta_{t,k} = beta_{t-1,k} + v_{t,k}
        #
        #    u_t  ~ N_M(0, Omega)
        #    v_{t,k} ~ N_M(0, W_k)  with W_k = sigma2_v * Omega
        #
        #    Forward filter and backward pass are Numba-compiled.
        r_beta = (Y
                  - alpha[None, :]
                  - np.einsum('tmp,mp->tm', Z, gamma))    # (T, M)

        W = sigma2_v * Omega    # system noise covariance (M, M)

        for k in range(K):
            # Partial residual: remove other features' contributions
            r_k = r_beta.copy()                           # (T, M)
            for kk in range(K):
                if kk != k:
                    r_k -= X[:, :, kk] * beta[:, :, kk]

            Xk = X[:, :, k]   # (T, M)

            # --- Numba forward filter ---
            m, C, m_pred, R_pred = _forward_filter(
                r_k, Xk, Omega, W, sigma2_v, delta, T, M
            )

            # --- Numba backward pass (returns h, B for each t) ---
            h, B = _backward_sample_means(m, C, m_pred, R_pred, T, M)

            # --- Draw samples (numpy, not Numba — requires RNG) ---
            b = np.zeros((T, M))
            for t in range(T):
                b[t] = rng.multivariate_normal(h[t], B[t])

            beta[:, :, k] = b   # (T, M)

        # 4. Sample Omega | residuals  →  IW(nu, Psi)
        #    nu = T - M + 2   (paper eq. 16)
        #    Psi = sum_t u_t u_t^T   (paper eq. 17)
        #    Note: v_t NOT included (paper recommendation for stability)
        U   = compute_residuals(alpha, beta, gamma)       # (T, M)
        Psi = U.T @ U                                     # (M, M)
        nu  = max(T - M + 2, M + 2)                      # ensure nu > M-1
        Psi = (Psi + Psi.T) / 2 + 1e-6 * np.eye(M)
        Omega = invwishart.rvs(df=nu, scale=Psi, random_state=rng)

        # Store post burn-in
        if it >= burn_in:
            s = it - burn_in
            alpha_samples[s] = alpha
            gamma_samples[s] = gamma
            beta_samples[s]  = beta.copy()
            Omega_samples[s] = Omega

        if (it + 1) % 500 == 0:
            print(f"  Iteration {it+1}/{n_iter}")

    return {
        "alpha": alpha_samples,   # (n_samples, M)
        "gamma": gamma_samples,   # (n_samples, M, p)
        "beta":  beta_samples,    # (n_samples, T, M, K)
        "Omega": Omega_samples,   # (n_samples, M, M)
    }


########## STEP 3 — GELMAN-RUBIN DIAGNOSTIC ##########
def gelman_rubin(chains: list) -> float:
    """
    chains : list of 1-D arrays (one per chain), post burn-in samples.
    Returns R-hat. Convergence: R-hat < 1.1
    """
    n      = min(len(c) for c in chains)
    chains = [c[:n] for c in chains]
    m      = len(chains)
    theta_bar  = np.array([c.mean() for c in chains])
    grand_mean = theta_bar.mean()
    B     = n / (m - 1) * np.sum((theta_bar - grand_mean)**2)
    W     = np.mean([c.var(ddof=1) for c in chains])
    V_hat = (n - 1) / n * W + (m + 1) / (m * n) * B
    return np.sqrt(V_hat / (W + 1e-12))

def estimate_Sigma_post_gibbs(beta_samples_list):
    """
    Estime Σ après le Gibbs sampling (section 4.2, papier) :
    "we recommend estimating Σ after the Gibbs sampling to ensure stability".

    Utilise les incréments v_t = beta_t - beta_{t-1} (éq. A.22), calculés
    sur CHAQUE échantillon post burn-in de TOUTES les chaînes combinées.
    Version vectorisée (pas de boucle Python sur s/k).

    Parameters
    ----------
    beta_samples_list : list of arrays, chacun (n_samples, T, M, K)

    Returns
    -------
    Sigma_hat : (M, M)
    """
    M = beta_samples_list[0].shape[2]

    Psi_Sigma = np.zeros((M, M))
    n_total   = 0

    for beta_samples in beta_samples_list:
        n_samples, T, M_, K = beta_samples.shape

        # v: (n_samples, T-1, M, K) — incréments sur l'axe temps
        v = beta_samples[:, 1:, :, :] - beta_samples[:, :-1, :, :]

        # Réorganiser pour traiter (n_samples * (T-1) * K) comme des observations (M,)
        # v shape (n_samples, T-1, M, K) -> (n_samples, T-1, K, M) -> (n_samples*(T-1)*K, M)
        v = np.moveaxis(v, 2, -1)                      # (n_samples, T-1, K, M)
        V = v.reshape(-1, M)                            # (n_samples*(T-1)*K, M)

        Psi_Sigma += V.T @ V
        n_total   += V.shape[0]

    nu_Sigma  = n_total - M + 2          # par analogie avec nu = T - M + 2 (éq. 16)
    Psi_Sigma = (Psi_Sigma + Psi_Sigma.T) / 2 + 1e-6 * np.eye(M)

    Sigma_hat = Psi_Sigma / (nu_Sigma - M - 1)
    return Sigma_hat


########### STEP 4 — FORECAST  (paper multi-step forecasting) ##########
def forecast(chains: dict, X_new: np.ndarray, Z_new: np.ndarray, Sigma_hat: np.ndarray, N: int = 1):
    """
    Multi-step forecast à l'horizon N (papier, éq. 18-20).

    Sigma_hat : (M, M) — estimée séparément après le Gibbs sampling
                (cf. estimate_Sigma_post_gibbs), PAS dérivée de sigma2_v*Omega.

    Parameters
    ----------
    X_new : (M, K)   sentiment features à T+N
    Z_new : (M, p)   features financières à T+N
    N     : horizon de prévision (1 pour ret1, 7 pour ret7)

    Returns
    -------
    mu    : (M,)    prévision moyenne posterior par stock
    Sigma : (M, M)  matrice de covariance de prévision
    """
    M = len(TICKERS)
    K = len(TV_FEATURES)

    alpha_mean = chains["alpha"].mean(axis=0)       # (M,)
    gamma_mean = chains["gamma"].mean(axis=0)       # (M, p)
    m_T        = chains["beta"].mean(axis=0)[-1]    # (M, K) — dernier état d'entraînement
    Omega_mean = chains["Omega"].mean(axis=0)       # (M, M)

    # ---- Point forecast (éq. 19) ----
    sent   = np.einsum('mk,mk->m', X_new, m_T)          # (M,)
    static = np.einsum('mp,mp->m', Z_new, gamma_mean)    # (M,)
    mu     = alpha_mean + sent + static                  # (M,)

    # ---- Forecast covariance (éq. 20) ----
    # F_{T+N} : vraie matrice bloc-diagonale (M, M*K), un bloc diag(X_new[:,k]) par feature k
    F = np.zeros((M, M * K))
    for k in range(K):
        F[:, k*M:(k+1)*M] = np.diag(X_new[:, k])

    # C_N : covariance des états à l'horizon N — bloc-diagonale (M*K, M*K),
    # chaque bloc k = N * Sigma_hat (même Sigma_hat partagée à travers les features,
    # cohérent avec le sampler qui suppose une structure Omega/Sigma commune)
    C_N = np.zeros((M * K, M * K))
    for k in range(K):
        C_N[k*M:(k+1)*M, k*M:(k+1)*M] = N * Sigma_hat

    # T*Sigma : accumulation de l'incertitude sur tout l'historique d'entraînement
    T_train = chains["beta"].shape[1]
    Sigma_tv = np.zeros((M * K, M * K))
    for k in range(K):
        Sigma_tv[k*M:(k+1)*M, k*M:(k+1)*M] = T_train * Sigma_hat

    Sigma = F @ (C_N + Sigma_tv) @ F.T + Omega_mean    # (M, M)

    return mu, Sigma


########### MAIN ##########
if __name__ == "__main__":

    #COMMENT = input("Remark: ")

    RESULTS_CSV = "data/logAVG3bis.csv"

    #MODEL_SPECS = [
    #    {"comment": "T=VADER | HL=RoBERTa+BoW+TBlob+VADER", "source": "avg3", "tv": ["VADER"], "hl": ["RoBERTa", "BoW", "TBlob", "VADER"]},
    #    {"comment": "T=VADER | HL=VADER",                    "source": "avg3", "tv": ["VADER"], "hl": ["VADER"]},
    #    {"comment": "T=VADER | HL=BoW+TBlob+VADER",          "source": "avg3", "tv": ["VADER"], "hl": ["BoW", "TBlob", "VADER"]},
    #    {"comment": "overnight T=VADER",                     "source": "overnight", "tv": ["VADER"], "hl": None},
    #    {"comment": "overnight T=TBlob+VADER",               "source": "overnight", "tv": ["TBlob", "VADER"], "hl": None},
    #    {"comment": "HL=VADER",                              "source": "avg3", "tv": None, "hl": ["VADER"]},
    #    {"comment": "T=VADER",                               "source": "avg3", "tv": ["VADER"], "hl": None},
    #    {"comment": "overnight T=TBlob",                     "source": "overnight", "tv": ["TBlob"], "hl": None},
    #    {"comment": "T=BoW+VADER | HL=VADER",                "source": "avg3", "tv": ["BoW", "VADER"], "hl": ["VADER"]},
    #    {"comment": "T=VADER | HL=BoW+VADER",                "source": "avg3", "tv": ["VADER"], "hl": ["BoW", "VADER"]},
    #]
#
    #def make_runs(specs):
    #    runs = []
    #    for s in specs:
    #        features = []
    #        if s["tv"] is not None:
    #            features.append("T_SENT")
    #        if s["hl"] is not None:
    #            features.append("HL_SENT")
    #        runs.append({**s, "features": features})
    #    return runs
#
    #RUNS = make_runs(MODEL_SPECS)
    
    tv_combos = all_non_empty_combos(BASE_SENTIMENTS)

    for tv_combo in tv_combos:
        
        # THIS IS TO MAKE BOTH RUN
        #for hl_combo in hl_combos:
        #    COMMENT = f"T={combo_label(tv_combo)} | HL={combo_label(hl_combo)}"
        #    print(f"\nCONFIG: {COMMENT}")
#
        #    df = {k: v.copy() for k, v in base_df.items()}
        #    for ticker in TICKERS:
        #        df[ticker] = add_combo_features(df[ticker], tv_combo, hl_combo)



        # THIS IS TO MAKE ONLY ONE RUN
        COMMENT = f"open-open T={combo_label(tv_combo)}"
        print(f"\nCONFIG: {COMMENT}")
#
        df = {k: v.copy() for k, v in base_df.items()}
        for ticker in TICKERS:
            df[ticker] = add_combo_features(df[ticker], tv_combo)


    # THIS IS TO RUN PRESETS
    #for run in RUNS:
    #    COMMENT = run["comment"] + " | ewma5 0.3"
    #    TV_FEATURES = run["features"]
#
    #    print(f"\nCONFIG: {COMMENT}")
#
    #    df = load_panel(run["source"])
#
    #    for ticker in TICKERS:
    #        if run["tv"] is not None and run["hl"] is not None:
    #            df[ticker] = add_combo_features(df[ticker], run["tv"])
    #            df[ticker] = add_combo_features_hl(df[ticker], run["hl"])
    #        elif run["tv"] is not None:
    #            df[ticker] = add_combo_features(df[ticker], run["tv"])
    #        elif run["hl"] is not None:
    #            df[ticker] = add_combo_features_hl(df[ticker], run["hl"])

        ### THIS IS FOR ALL MODELS, JUST NEED TO CHANGE INDENT
        for target in ['ret1']:
            print(f"TARGET: {target}")
            Y, X, Z, dates = build_panel(df, target, max_T=629)
    
            # 1. Définition du split 80/20
            split_idx = int(len(Y) * 0.8)
            Y_train, X_train, Z_train = Y[:split_idx], X[:split_idx], Z[:split_idx]
            Y_test, X_test, Z_test, dates_test = Y[split_idx:], X[split_idx:], Z[split_idx:], dates[split_idx:]
    
            print(f"Y (train): {Y_train.shape}")
            print(f"Y (test): {Y_test.shape}")
    
            # 2. Exécution du Gibbs sampler uniquement sur le Train
            chains_list = []
            for chain_id, seed in enumerate([42, 123]):
                print(f"\nChain {chain_id + 1}/2 — seed={seed}")
                chain = run_gibbs(Y_train, X_train, Z_train, n_iter=N_ITER, burn_in=BURN_IN, delta=DELTA, seed=seed)
                chains_list.append(chain)
    
            # 3. Diagnostic Gelman-Rubin
            print("\nGelman-Rubin R-hat for alpha (Train):")
            for j, ticker in enumerate(TICKERS):
                rhat = gelman_rubin([c["alpha"][:, j] for c in chains_list])
                print(f"  {ticker}: R-hat = {rhat:.4f}")
    
            Sigma_hat = estimate_Sigma_post_gibbs([c["beta"] for c in chains_list])
    
            # 4. Combinaison des chaînes
            combined = {
                key: np.concatenate([c[key] for c in chains_list], axis=0)
                for key in chains_list[0]
            }
    
            # 5. Extraction des paramètres postérieurs moyens
            alpha_hat = combined["alpha"].mean(axis=0)        # (M,)
            gamma_hat = combined["gamma"].mean(axis=0)        # (M, p)
            m_T = combined["beta"].mean(axis=0)[-1]           # (M, K) - Dernier état d'entraînement
    
            # 6. Prédiction OOS (Application de l'équation 21 du papier)
            Y_pred = (alpha_hat[None, :]
                      + np.einsum('tmk,mk->tm', X_test, m_T)
                      + np.einsum('tmp,mp->tm', Z_test, gamma_hat))
    
            # OOS evaluation
            hit_rate = np.mean(np.sign(Y_pred) == np.sign(Y_test), axis=0)
    
    
            mse_model = np.sum((Y_test - Y_pred)**2, axis=0)
            mse_naive = np.sum(Y_test**2, axis=0)
            r2_oos = 1 - (mse_model / mse_naive)
    
            # ---- Sauvegarde des résultats en CSV ----
            row = {
                "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "comment": COMMENT,
                "avg": "log-weighted",
                "hit_rate_mean": hit_rate.mean(),
                "r2_oos_mean": r2_oos.mean(),
            }
            for j, ticker in enumerate(TICKERS):
                row[f"hit_rate_{ticker}"] = hit_rate[j]
    
            new_row_df = pd.DataFrame([row])
    
            try:
                existing = pd.read_csv(RESULTS_CSV)
                updated = pd.concat([existing, new_row_df], ignore_index=True)
            except FileNotFoundError:
                updated = new_row_df
    
            updated.to_csv(RESULTS_CSV, index=False)
            print(f"Results saved: {RESULTS_CSV}")
            import subprocess
            subprocess.run(['afplay', '/System/Library/Sounds/Bottle.aiff'])