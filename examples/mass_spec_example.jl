# Example demonstrating mass-spec based Bayesian inference

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Mouls

println("="^70)
println("Mass-Spec Observation Example")
println("="^70)

# Load coupling table
coupling_table = create_coupling_table("contextual_couplings_Young1990")
println("\n1. Loaded coupling table")

# Initialize Bayesian prior
prior = initialize_prior_from_coupling_table(coupling_table, prior_strength=100.0)
println("\n2. Initialized Bayesian prior")
println(prior)

# Create synthetic sequence observation
println("\n3. first data from Pepsysco: SELLTPLGIDLDEW")
seq_obs = MassSpecObservation(
    target_sequence="SELLTPLGIDLDEW",
    mass_histogram=Dict{Float64,Int}(
        1285.09 => 48,
        1582.9 => 115,
        1601.2 => 208,
        1639.0 => 52,
        1657.3 => 77
    ),
    total_count=500
)
println(seq_obs)

println("\n4. Simulation of Pepsysco first data SELLTPLGIDLDEW")
mass_obs = create_synthetic_mass_observation(
    "SELLTPLGIDLDEW",
    from_bayesian_prior(prior),
    num_simulations=1000,
    truncation_rate=0.001
)
println(mass_obs)

println("\n" * "="^70)
println("5. Testing Histogram Comparison and Loss Calculation")
println("="^70)

println("\n--- Real Pepsysco Data: SELLTPLGIDLDEW ---")
coupling_test = from_bayesian_prior(prior)
predicted_mass_hist_real = sequence_to_mass_histogram(
    "SELLTPLGIDLDEW", coupling_test, 0.001, num_simulations=1000
)

println("\nObserved mass histogram (real Pepsysco data):")
for (mass, count) in sort(collect(seq_obs.mass_histogram), by=x->x[2], rev=true)
    prob = count / seq_obs.total_count
    println("  $(round(mass, digits=2)) Da: $count ($(round(prob*100, digits=1))%)")
end

println("\nPredicted mass histogram (showing top 5):")
for (mass, count) in sort(collect(predicted_mass_hist_real), by=x->x[2], rev=true)[1:min(5, length(predicted_mass_hist_real))]
    prob = count / sum(values(predicted_mass_hist_real))
    println("  $(round(mass, digits=2)) Da: $count ($(round(prob*100, digits=1))%)")
end

kernel_bw = 1.0  # Default kernel bandwidth in Da
real_log_likelihood = compare_histograms(
    predicted_mass_hist_real, seq_obs.mass_histogram, kernel_bw
)
println("\nReal data comparison log-likelihood: $(round(real_log_likelihood, digits=2))")
println("Kernel bandwidth used: $(kernel_bw) Da")

println("\n--- Synthetic Simulation: SELLTPLGIDLDEW ---")
predicted_mass_hist = sequence_to_mass_histogram(
    "SELLTPLGIDLDEW", coupling_test, 0.001, num_simulations=1000
)

println("\nObserved mass histogram (synthetic for testing):")
for (mass, count) in sort(collect(mass_obs.mass_histogram), by=x->x[2], rev=true)
    prob = count / mass_obs.total_count
    println("  $(round(mass, digits=2)) Da: $count ($(round(prob*100, digits=1))%)")
end

println("\nPredicted mass histogram (showing top 5):")
for (mass, count) in sort(collect(predicted_mass_hist), by=x->x[2], rev=true)[1:min(5, length(predicted_mass_hist))]
    prob = count / sum(values(predicted_mass_hist))
    println("  $(round(mass, digits=2)) Da: $count ($(round(prob*100, digits=1))%)")
end

synth_log_likelihood = compare_histograms(
    predicted_mass_hist, mass_obs.mass_histogram, kernel_bw
)
println("\nSynthetic data comparison log-likelihood: $(round(synth_log_likelihood, digits=2))")
println("Kernel bandwidth used: $(kernel_bw) Da")

println("\n--- Testing Kernel Bandwidth Sensitivity (Real Data) ---")
bandwidths = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
println("\nLog-likelihood vs. kernel bandwidth for real Pepsysco data:")
for bw in bandwidths
    ll = compare_histograms(predicted_mass_hist_real, seq_obs.mass_histogram, bw)
    println("  $(lpad(bw, 6)) Da: $(round(ll, digits=2))")
end

println("\n" * "="^70)
println("Example complete!")
println("="^70)



