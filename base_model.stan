data {
    int<lower=1> B;
    array[B] int<lower=0> counts;
    vector[B] log_E_norm;
    vector[B] E_widths;
    real log_E_mean;
    matrix[B, B] M_eff; // detector response: M_eff[reco, true] = smearing * acceptance * exposure
    // prior hyperparameters (passed as data to match Python model config)
    real prior_log_phi_mean;
    real<lower=0> prior_log_phi_sd;
    real prior_log_eta_at_mean_mean; // = eta_log_mean - delta_prior_mean * log_E_mean
    real<lower=0> prior_log_eta_at_mean_sd;
    real prior_gamma_mean;
    real<lower=0> prior_gamma_sd;
    real prior_delta_mean;
    real<lower=0> prior_delta_sd;
}

parameters {
    real log_phi;
    real<lower=0.5, upper=6.0> gamma;
    real log_eta_at_mean;
    real<lower=2.0, upper=5.0> delta;
}

transformed parameters {
    real log_eta = log_eta_at_mean + delta * log_E_mean;
}

model {
    log_phi ~ normal(prior_log_phi_mean, prior_log_phi_sd);
    gamma ~ normal(prior_gamma_mean, prior_gamma_sd);
    log_eta_at_mean ~ normal(prior_log_eta_at_mean_mean, prior_log_eta_at_mean_sd);
    delta ~ normal(prior_delta_mean, prior_delta_sd);

    vector[B] mu_sig = M_eff * (exp(log_phi - gamma * log_E_norm) .* E_widths);
    vector[B] mu_bg = M_eff * (exp(log_eta_at_mean - delta * (log_E_norm - log_E_mean)) .* E_widths);
    counts ~ poisson(mu_sig + mu_bg);
}

generated quantities {
    array[B] int counts_rep;
    vector[B] mu_sig_rep = M_eff * (exp(log_phi - gamma * log_E_norm) .* E_widths);
    vector[B] mu_bg_rep  = M_eff * (exp(log_eta_at_mean - delta * (log_E_norm - log_E_mean)) .* E_widths);
    for (b in 1:B)
        counts_rep[b] = poisson_rng(mu_sig_rep[b] + mu_bg_rep[b]);
}
