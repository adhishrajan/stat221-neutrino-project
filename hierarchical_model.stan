data {
    int<lower=1> K; // number of seasons
    int<lower=1> B; // number of energy bins
    array[K, B] int<lower=0> counts; // observed counts [season, bin]
    vector[B] log_E_norm; // log(E_centers / E_REF)
    vector[B] E_widths; // bin widths
    real log_E_mean; // mean of log_E_norm
    matrix[B, B] M_eff; // detector response: M_eff[reco, true] = smearing * acceptance * exposure
    // prior hyperparameters (passed as data to match Python model config)
    real prior_mu_phi_mean;
    real<lower=0> prior_mu_phi_sd;
    real prior_mu_eta_at_mean_mean; // = eta_log_mean - delta_prior_mean * log_E_mean
    real<lower=0> prior_mu_eta_at_mean_sd;
    real prior_gamma_mean;
    real<lower=0> prior_gamma_sd;
    real prior_delta_mean;
    real<lower=0> prior_delta_sd;
}

parameters {
    // season-specific signal fluxes (non-centered)
    vector[K] z_k;
    real mu_phi;
    real<lower=0> sigma_phi;

    // season-specific background rates (non-centered)
    vector[K] z_eta_k;
    real mu_eta_at_mean;
    real<lower=0> sigma_eta;

    // shared parameters
    real<lower=0.5, upper=6.0> gamma;
    real<lower=2.0, upper=5.0> delta;
}

transformed parameters {
    vector[K] log_phi_k = mu_phi + sigma_phi * z_k;
    vector[K] log_eta_at_mean_k = mu_eta_at_mean + sigma_eta * z_eta_k;
}

model {
    // hyperpriors for phi
    mu_phi ~ normal(prior_mu_phi_mean, prior_mu_phi_sd);
    sigma_phi ~ normal(0, 0.5);
    z_k ~ std_normal();

    // hyperpriors for eta
    mu_eta_at_mean ~ normal(prior_mu_eta_at_mean_mean, prior_mu_eta_at_mean_sd);
    sigma_eta ~ normal(0, 0.5);
    z_eta_k ~ std_normal();

    // shared priors
    gamma ~ normal(prior_gamma_mean, prior_gamma_sd);
    delta ~ normal(prior_delta_mean, prior_delta_sd);

    // likelihood over all seasons
    for (k in 1:K) {
        vector[B] mu_sig = M_eff * (exp(log_phi_k[k] - gamma * log_E_norm) .* E_widths);
        vector[B] mu_bg = M_eff * (exp(log_eta_at_mean_k[k] - delta * (log_E_norm - log_E_mean)) .* E_widths);
        counts[k] ~ poisson(mu_sig + mu_bg);
    }
}

generated quantities {
    // posterior predictive for season 1 only as a check
    array[B] int counts_rep;
    {
        vector[B] mu_sig = M_eff * (exp(log_phi_k[1] - gamma * log_E_norm) .* E_widths);
        vector[B] mu_bg = M_eff * (exp(log_eta_at_mean_k[1] - delta * (log_E_norm - log_E_mean)) .* E_widths);
        for (b in 1:B)
            counts_rep[b] = poisson_rng(mu_sig[b] + mu_bg[b]);
    }
}
