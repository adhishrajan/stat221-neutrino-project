data {
    int<lower=1> K; // number of seasons
    int<lower=1> B; // number of energy bins
    array[K, B] int<lower=0> counts; // observed counts [season, bin]
    vector[B] log_E_norm; // log(E_centers / E_REF)
    vector[B] E_widths; // bin widths
    real log_E_mean; // mean of log_E_norm
}

parameters {
    // season specific fluxes (non centered)
    vector[K] z_k;
    real mu_phi;
    real<lower=0> sigma_phi;

    // shared parameters
    real<lower=0.5, upper=6.0> gamma;
    real log_eta_at_mean;
    real<lower=2.0, upper=5.0> delta;
}

transformed parameters {
    vector[K] log_phi_k = mu_phi + sigma_phi * z_k;
    real log_eta = log_eta_at_mean + delta * log_E_mean;
}

model {
    // hyperpriors
    mu_phic ~ normal(-11.51, 2.0);
    sigma_phi ~ normal(0, 0.5);

    // non centered latents
    z_k ~ std_normal();

    // shared priors
    gammac~ lognormal(log(2.5), 0.2);
    log_eta_at_mean ~ normal(-11.51 - 3.7 * log_E_mean, 1.0);
    deltac~ normal(3.7, 0.1);

    // likelihood over all seasons
    for (k in 1:K) {
        vector[B] mu_sig = exp(log_phi_k[k] - gamma * log_E_norm) .* E_widths;
        vector[B] mu_bgc= exp(log_eta_at_mean - delta * (log_E_norm - log_E_mean)) .* E_widths;
        counts[k] ~ poisson(mu_sig + mu_bg);
    }
}

generated quantities {
    // posterior predictive for season 1 only as a check
    array[B] int counts_rep;
    {
        vector[B] mu_sig = exp(log_phi_k[1] - gamma * log_E_norm) .* E_widths;
        vector[B] mu_bg = exp(log_eta_at_mean - delta * (log_E_norm - log_E_mean)) .* E_widths;
        for (b in 1:B)
            counts_rep[b] = poisson_rng(mu_sig[b] + mu_bg[b]);
    }
}