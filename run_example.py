/**
 * Bayesian VUS Meta-Analysis Model  —  vus_meta.stan
 * =====================================================
 * Hierarchical binormal ROC surface model for pooling
 * diagnostic test evaluations across studies with
 * heterogeneous disease severity distributions.
 *
 * The model implements the same severity-weighted likelihood
 * and coverage-first principle as the Python pipeline:
 *   - Each observation weighted by w_j(s_i) = n_dis(s_i)/(n_dis(s_i)+n*)
 *   - Population-level VUS estimated from the posterior dome
 *   - Per-study VUS and PVUS reported in generated quantities
 *
 * HOW TO RUN
 * ----------
 * 1. Install CmdStan: https://mc-stan.org/users/interfaces/cmdstan
 * 2. Prepare data:    python src/run_simulations.py --output-stan-json
 * 3. Compile:         make /path/to/vus_meta
 * 4. Sample:          ./vus_meta sample data file=outputs/stan_data.json
 *
 * Or with CmdStanPy:
 *   import cmdstanpy
 *   model = cmdstanpy.CmdStanModel(stan_file='stan/vus_meta.stan')
 *   fit   = model.sample(data='outputs/stan_data.json')
 *
 * Citation: [paper citation]
 * License:  MIT
 */

data {
  int<lower=1> J;                        // number of studies
  int<lower=1> N;                        // total patients

  array[N] int<lower=0,upper=1> y;       // disease label
  vector[N] score;                       // continuous test score
  vector[N] sev_norm;                    // severity normalised [0,1]
  array[N] int<lower=1,upper=J> study_id;
  vector[N] weight;                      // reliability weight w_j(s_i)

  int<lower=3> n_sev_grid;              // grid points for VUS integration
}

parameters {
  real alpha0;
  real<lower=0> alpha1;                  // positive: worse disease => better disc.
  real<lower=0> beta0;
  real beta1;

  real<lower=0> tau_alpha0;
  real<lower=0> tau_alpha1;
  real<lower=0> tau_beta0;
  real<lower=0> tau_beta1;

  vector[J] z_alpha0;
  vector[J] z_alpha1;
  vector[J] z_beta0;
  vector[J] z_beta1;
}

transformed parameters {
  vector[J] alpha0_j = alpha0 + tau_alpha0 * z_alpha0;
  vector[J] alpha1_j = alpha1 + tau_alpha1 * z_alpha1;
  vector[J] beta0_j  = beta0  + tau_beta0  * z_beta0;
  vector[J] beta1_j  = beta1  + tau_beta1  * z_beta1;
}

model {
  // Priors
  alpha0 ~ normal(0.5, 1.5);
  alpha1 ~ normal(0.5, 0.5);
  beta0  ~ normal(1.0, 0.5);
  beta1  ~ normal(0.0, 0.3);
  tau_alpha0 ~ normal(0, 0.5);
  tau_alpha1 ~ normal(0, 0.3);
  tau_beta0  ~ normal(0, 0.3);
  tau_beta1  ~ normal(0, 0.2);
  z_alpha0 ~ std_normal();
  z_alpha1 ~ std_normal();
  z_beta0  ~ std_normal();
  z_beta1  ~ std_normal();

  // Severity-weighted likelihood
  for (i in 1:N) {
    int j    = study_id[i];
    real s   = sev_norm[i];
    real alpha_s = alpha0_j[j] + alpha1_j[j] * s;
    real beta_s  = fmax(beta0_j[j] + beta1_j[j] * s, 0.05);
    real mu_s    = alpha_s / beta_s;
    real sigma_s = 1.0   / beta_s;
    if (y[i] == 1)
      target += weight[i] * normal_lpdf(score[i] | mu_s, sigma_s);
    else
      target += weight[i] * normal_lpdf(score[i] | 0.0,  1.0);
  }
}

generated quantities {
  // Population-level AUC(s) and VUS
  vector[n_sev_grid] sev_grid_vals;
  vector[n_sev_grid] auc_per_s;
  real vus_summary;

  for (k in 1:n_sev_grid) {
    sev_grid_vals[k] = (k - 1.0) / (n_sev_grid - 1.0);
    real s       = sev_grid_vals[k];
    real alpha_s = alpha0 + alpha1 * s;
    real beta_s  = fmax(beta0 + beta1 * s, 0.01);
    real a_bin   = alpha_s / 1.65;
    auc_per_s[k] = Phi(a_bin / sqrt(1.0 + beta_s * beta_s));
  }
  vus_summary = mean(auc_per_s);

  // Per-study VUS
  vector[J] vus_per_study;
  for (j in 1:J) {
    vector[n_sev_grid] auc_j;
    for (k in 1:n_sev_grid) {
      real s       = sev_grid_vals[k];
      real alpha_s = alpha0_j[j] + alpha1_j[j] * s;
      real beta_s  = fmax(beta0_j[j] + beta1_j[j] * s, 0.01);
      real a_bin   = alpha_s / 1.65;
      auc_j[k]     = Phi(a_bin / sqrt(1.0 + beta_s * beta_s));
    }
    vus_per_study[j] = mean(auc_j);
  }

  // Log-likelihood for LOO-CV
  vector[N] log_lik;
  for (i in 1:N) {
    int j    = study_id[i];
    real s   = sev_norm[i];
    real alpha_s = alpha0_j[j] + alpha1_j[j] * s;
    real beta_s  = fmax(beta0_j[j] + beta1_j[j] * s, 0.05);
    if (y[i] == 1)
      log_lik[i] = normal_lpdf(score[i] | alpha_s/beta_s, 1.0/beta_s);
    else
      log_lik[i] = normal_lpdf(score[i] | 0.0, 1.0);
  }
}
