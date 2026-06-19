import numpy as np
import pandas as pd
from scipy.stats import invwishart

########## CONFIG ##########
TICKERS       = ["AAPL", "AMZN", "GOOG", "GOOGL", "MSFT", "TSLA"]
TV_FEATURES   = ["TVADER", "HLTBlob"]                          # X : time-varying (sentiment)
STAT_FEATURES = ["dlyret", "past_3ret", "past_7ret",
                 "volumeSPX", "dlyretSPX", "VIX"]              # Z : non-time-varying
TARGETS       = ["ret1", "ret7"]

N_ITER  = 5000 # Paper: 5000
BURN_IN = 1000  # Paper: 1000
DELTA   = 0.9   # signal-to-noise ratio. Papper: 0.9




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

    for j, ticker in enumerate(TICKERS):
        dft = df[ticker].reindex(common_idx)
        dft[TV_FEATURES]   = dft[TV_FEATURES].ffill().fillna(0)
        dft[STAT_FEATURES] = dft[STAT_FEATURES].ffill().fillna(0)
        Y[:, j]    = dft[target].values
        X[:, j, :] = dft[TV_FEATURES].values
        Z[:, j, :] = dft[STAT_FEATURES].values

    valid = ~np.isnan(Y).any(axis=1)
    return Y[valid], X[valid], Z[valid], common_idx[valid]




########## STEP 2 — GIBBS SAMPLER ##########
def run_gibbs(Y, X, Z, n_iter=N_ITER, burn_in=BURN_IN, delta=DELTA, seed=42):
    """
    HD-SURDLM Gibbs sampler — paper-faithful implementation.

    Model
    -----
    y_{j,t} = alpha_j + sum_k X_{j,t,k} * beta_{j,t,k} + Z_{j,t} @ gamma_j + u_{j,t}
    beta_{j,t,k} = beta_{j,t-1,k} + v_{j,t,k}

    u_t ~ N_M(0, Omega)   (cross-stock correlated observation errors)
    v_{t,k} ~ N(0, Omega) (system errors share the same covariance structure)

    FFBS is run in full M-dimensional form for each feature k, so that
    cross-stock correlations in Omega propagate into beta sampling.

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
        #    Forward filter:  m_t (M,), C_t (M,M)
        #    Backward sample: stabilised FFBS using m_{t+1} not sampled beta
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
            # r_k[:, j] = X_{j,t,k} * beta_{j,t,k} + u_{j,t}

            # --- Forward filter ---
            # m[t]: (M,)   posterior mean of beta_{t,k}
            # C[t]: (M,M)  posterior covariance
            m      = np.zeros((T, M))
            C      = np.zeros((T, M, M))
            m_pred = np.zeros((T, M))
            R_pred = np.zeros((T, M, M))

            # Diffuse initialisation
            m_pred[0] = np.zeros(M)
            R_pred[0] = (sigma2_v / (1.0 - delta + 1e-10)) * Omega # Papier : diffuse initialization classique (grande variance)
            #  D'où vient 1.0 - delta ? Le papier ne spécifie pas cette constante. C'est une approximation heuristique

            for t in range(T):
                # F_t = diag(X_{t,:,k})  shape (M, M)
                F_t = np.diag(X[t, :, k])                # (M, M)

                # One-step forecast of observation
                # f_t = F_t @ m_pred[t]
                f_t = F_t @ m_pred[t]                    # (M,)
                # Q_t = F_t @ R_pred[t] @ F_t.T + Omega
                Q_t = F_t @ R_pred[t] @ F_t.T + Omega   # (M, M)
                Q_t = (Q_t + Q_t.T) / 2 + 1e-8 * np.eye(M)

                # Innovation
                e_t = r_k[t] - f_t                       # (M,)

                # Kalman gain
                Q_inv = np.linalg.inv(Q_t)
                A_t   = R_pred[t] @ F_t.T @ Q_inv        # (M, M)

                # Update
                m[t] = m_pred[t] + A_t @ e_t             # (M,)
                C[t] = R_pred[t] - A_t @ F_t @ R_pred[t] # (M, M)
                C[t] = (C[t] + C[t].T) / 2 + 1e-8 * np.eye(M)

                # Predict next state
                if t < T - 1:
                    m_pred[t+1] = m[t]                   # random walk
                    R_pred[t+1] = C[t] + W
                    R_pred[t+1] = (R_pred[t+1] + R_pred[t+1].T) / 2

            # --- Backward sampling (stabilised FFBS) ---
            # beta_T ~ N_M(m_T, C_T)
            b = np.zeros((T, M))
            C_T_reg = C[T-1] + 1e-8 * np.eye(M)
            b[T-1]  = rng.multivariate_normal(m[T-1], C_T_reg)

            for t in range(T-2, -1, -1):
                R_next     = R_pred[t+1] + 1e-8 * np.eye(M)
                R_next_inv = np.linalg.inv(R_next)

                # Stabilised: use m[t+1] instead of sampled b[t+1]
                # h_t = m_t + C_t @ R_{t+1}^{-1} @ (m_{t+1} - m_t)
                h_t = m[t] + C[t] @ R_next_inv @ (m[t+1] - m[t])

                # B_t = C_t + C_t @ R_{t+1}^{-1} @ B_{t+1} @ R_{t+1}^{-1} @ C_t
                #           - C_t @ R_{t+1}^{-1} @ C_t
                # At t=T-2, B_{t+1} = C_T (last posterior cov)
                if t == T - 2:
                    B_next = C[T-1]
                else:
                    B_next = B_t  # carried from previous backward step  # noqa

                CR  = C[t] @ R_next_inv
                B_t = C[t] + CR @ B_next @ CR.T - CR @ C[t]
                B_t = (B_t + B_t.T) / 2 + 1e-8 * np.eye(M)

                b[t] = rng.multivariate_normal(h_t, B_t)

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




########### STEP 4 — FORECAST  (paper multi-step forecasting) ##########
def forecast(chains: dict, X_new: np.ndarray, Z_new: np.ndarray, N: int = 1):
    """
    Multi-step forecast at horizon N (paper Section III.B).

    mu_{T+N}    = alpha + X_{T+N} @ m_T + Z_{T+N} @ gamma
    Sigma_{T+N} = F_{T+N} (C_N + T*Sigma) F_{T+N}.T + Omega

    Parameters
    ----------
    X_new : (M, K)   sentiment features at T+N
    Z_new : (M, p)   financial features at T+N
    N     : forecast horizon (1 for ret1, 7 for ret7)

    Returns
    -------
    mu    : (M,)   posterior mean forecast per stock
    Sigma : (M, M) forecast covariance matrix
    """
    alpha_mean = chains["alpha"].mean(axis=0)       # (M,)
    gamma_mean = chains["gamma"].mean(axis=0)       # (M, p)
    m_T        = chains["beta"].mean(axis=0)[-1]    # (M, K) — last time step
    Omega_mean = chains["Omega"].mean(axis=0)       # (M, M)

    # Point forecast
    sent   = np.einsum('mk,mk->m', X_new, m_T)          # (M,)
    static = np.einsum('mp,mp->m', Z_new, gamma_mean)    # (M,)
    mu     = alpha_mean + sent + static                  # (M,)

    # Forecast covariance: F_{T+N} is block-diag of X_new columns
    # Simplified: treat each feature k independently, sum contributions
    sigma2_v = (1.0 - DELTA) / DELTA
    C_N      = N * sigma2_v * Omega_mean               # grows with horizon N
    T_train   = chains["beta"].shape[1]     
    Sigma_tv  = T_train * sigma2_v * Omega_mean             # T*Sigma proxy

    # Est-ce une erreur par rapport au papier? Σ_{T+N} = F_{T+N} (C_N + T Σ) F_{T+N}^T + Ω où C_N est la covariance des états à l'horizon N.
    # Problème: on double N * sigma2_v * Omega_mean. Le terme T Σ du papier n'est pas clair (T = nombre d'observations ? Σ = ?). Dans votre code, vous utilisez deux fois le même terme. Ceci est une erreur probable par rapport au papier

    # F_{T+N}: (M, M*K) block diagonal — use sum over k for point forecast
    F = np.zeros((len(TICKERS), len(TICKERS)))
    for k in range(len(TV_FEATURES)):
        F += np.diag(X_new[:, k])

    Sigma = F @ (C_N + Sigma_tv) @ F.T + Omega_mean    # (M, M)

    return mu, Sigma




########### MAIN ##########
if __name__ == "__main__":

    for target in ['ret1']:
        print(f"TARGET: {target}")

        Y, X, Z, dates = build_panel(df, target, max_T=1258)
        
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
        mae_oos = np.abs(Y_test - Y_pred).mean(axis=0)
        
        print(f"\nOut-of-sample MAE per stock ({target}):")
        for j, ticker in enumerate(TICKERS):
            print(f"  {ticker}: {mae_oos[j]:.6f}")

        hit_rate = np.mean(np.sign(Y_pred) == np.sign(Y_test), axis=0)

        print("\nDirectional Accuracy (Hit Rate):")
        for j, ticker in enumerate(TICKERS):
            print(f"  {ticker}: {hit_rate[j]:.4f}")

        strat_returns = np.sign(Y_pred) * Y_test
        cum_pnl = strat_returns.sum(axis=0)

        print("\nCumulative OOS P&L:")
        for j, ticker in enumerate(TICKERS):
            print(f"  {ticker}: {cum_pnl[j]:.4f}")

        mse_model = np.sum((Y_test - Y_pred)**2, axis=0)
        mse_naive = np.sum(Y_test**2, axis=0)
        r2_oos = 1 - (mse_model / mse_naive)

        print("\nOut-of-Sample R-squared (R2 OOS):")
        for j, ticker in enumerate(TICKERS):
            print(f"  {ticker}: {r2_oos[j]:.6f}")

# . Absence d'estimation de Σ séparément — Mentionné dans le papier Paper (section III.B, dernier paragraphe) : "We recommend estimating Σ after the Gibbs sampling to ensure stability" Votre code n'estime jamais Σ (la matrice A Ω A ?). Vous utilisez directement Ω. Le papier semble distinguer : Ω = covariance des erreurs d'observatio Σ = A Ω A pour les erreurs système Dans votre modèle : W = sigma2_v * Ω → vous imposez A = sqrt(sigma2_v)·I. C'est une simplification non discutée dans le papier.
