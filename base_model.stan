data {
    int<lower=1> B;
    array[B] int<lower=0> counts;
    vector[B] log_E_norm;
    vector[B] E_widths;
    real log_E_mean;
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
    log_phi ~ normal(-11.51, 1.0);
    gamma ~ lognormal(log(2.5), 0.2);
    log_eta_at_mean ~ normal(-11.51 - 3.7 * log_E_mean, 1.0);
    delta ~ normal(3.7, 0.1);

    vector[B] mu_sig = exp(log_phi - gamma * log_E_norm) .* E_widths;
    vector[B] mu_bg = exp(log_eta_at_mean - delta * (log_E_norm - log_E_mean)) .* E_widths;
    vector[B] mu = mu_sig + mu_bg;

    counts ~ poisson(mu);
}

generated quantities {
    array[B] int counts_rep;
    vector[B] mu_sig_rep = exp(log_phi - gamma * log_E_norm) .* E_widths;
    vector[B] mu_bg_rep  = exp(log_eta_at_mean - delta * (log_E_norm - log_E_mean)) .* E_widths;
    for (b in 1:B)
        counts_rep[b] = poisson_rng(mu_sig_rep[b] + mu_bg_rep[b]);
}