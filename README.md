# Portfolio Replica Strategy

Replication of a black-box target portfolio using liquid futures, without knowing its holdings.

**Data**: Weekly Bloomberg prices, Oct 2007 – Apr 2021 (704 obs)  
**Target**: Monster Index — Sharpe-ratio-weighted combination of MXWO, MXWD, LEGATRUU, HFRXGL  
**Features**: 10 futures contracts — RX1, TY1, GC1, CO1, ES1, VG1, NQ1, TP1, DU1, TU2 *(LLL1 dropped: stale after ~100 zero-diff weeks)*

---

## `InvestmentReplica.ipynb` — Main Notebook

| Section | Content |
|---|---|
| Data Construction | Weekly returns via `pct_change()`; Monster Index built with fixed Sharpe-ratio weights; stale futures removed |
| EDA | QQ plots, Ljung-Box autocorrelation, ADF & Phillips-Perron stationarity tests |
| Markov Switching Elastic Net | Rolling EN with Baum-Welch HMM regime detection; regime-conditional hyperparameter grids (low-vol: flexible, high-vol: sparse); VIF feature selection; adaptive exponential decay sample weights; VaR(99%) ≤ 20% constraint |
| Kalman Filter | Dynamic state-space model with online Bayesian HMM; time-varying betas estimated via Kalman gain |
| Soft Bayesian Stacking | Posterior log-score weighting of all replica models; weights optimised on past predictive densities (causal, no look-ahead) |
| Final Analysis | Cornish-Fisher VaR, Filtered Historical Simulation VaR, Historical Stress VaR (GFC / COVID), cross-model comparison table |

---

## `failed_attempts.ipynb` — Exploratory Notebook

Documents approaches tested and discarded before settling on the final models.

| Approach | Notes |
|---|---|
| Elastic Net (OLS R1 → R4 LO) | Four configurations tested sequentially (weekly L/S → monthly rebalancing → EWM → long-only); no single config satisfied all thresholds simultaneously and none accounted for market regimes |
| Pooling (O'Doherty et al., 2017) | Futures grouped into four factor pools (US equity, intl equity, rates, commodities) combined via log-score optimal weights; tracking error and information ratio remained worse than plain EN on every metric |
| Kalman Filter — fixed Q | Standard Kalman filter with constant process noise matrix; dropped because market regimes were never taken into account |
| **LSTM** | TensorFlow/Keras sequential model; 8-week lookback, 32 hidden units, input + recurrent dropout 0.2, early stopping (patience 10), rolling 4-year training window; gating mechanism compared conceptually to Kalman gain — ultimately not selected because of poor performance
