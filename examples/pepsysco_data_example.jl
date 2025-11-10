using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Mouls
using Printf

println("="^70)
println("Pepsysco Mass Spec Data Example")
println("="^70)

println("\nLoading experimental mass-spec data from pepsysco.csv...")
observations = include(joinpath(@__DIR__, "..", "data", "pepsysco_observations.jl"))
println("✓ Loaded $(length(observations)) peptide observations")

println("\nShowing first 5 observations:")
for i in 1:min(5, length(observations))
    obs = observations[i]
    println("\n$i. Sequence: $(obs.target_sequence)")
    println("   Unique masses observed: $(length(obs.mass_histogram))")
    println("   Total counts: $(obs.total_count)")
    
    sorted_masses = sort(collect(obs.mass_histogram), by=x->x[2], rev=true)
    n=length(sorted_masses)
    println("   $n peaks:")
    for (mass, count) in sorted_masses
        prob = count / obs.total_count
        println("      $(round(mass, digits=2)) Da: $count ($(round(prob*100, digits=1))%)")
    end
end

println("\n--- Let's get ready to rumble! ---")

coupling_table = create_coupling_table("contextual_couplings_Young1990")
prior = initialize_prior_from_coupling_table(coupling_table, prior_strength=100.0)
prior_table = from_bayesian_prior(prior)

println("\n--- Testing Kernel Bandwidth Sensitivity (Pepsysco Data) ---")
bandwidths = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
println("\nLog-likelihood vs. kernel bandwidth for Pepsysco data:")
for bandwidth in bandwidths

    losses= [ 
        compare_histograms(sequence_to_mass_histogram(observations[i].target_sequence, from_bayesian_prior(prior), 0.001, num_simulations=1000), observations[i].mass_histogram, bandwidth) for i in 1:length(observations)
    ]
    println("  $(lpad(bandwidth, 6)) Da: $(round(sum(losses), digits=2))")
end

println("\n" * "="^70)
println("Use these observations with Bayesian inference:")


posterior = update_posterior(
    prior,
    observations[1:end],  # choose observations to include in optimisation: slows it down, obvs. 
    method=:map,
    num_likelihood_sims=200, # more or less linear wall-clock proport to this, once > 50
    max_iterations=50,
    kernel_bandwidth=20.0 # Da, bandwidth for kernel density comparison in mass spec histograms
)

println("\n5. Convert posterior back to coupling table for predictions")
updated_coupling_table = from_bayesian_prior(posterior)
println("\nUpdated Coupling Table:")
println(updated_coupling_table)

println("\nChange in coupling probabilities (Δ = updated - prior):")
deltas = abs.(updated_coupling_table.matrix .- prior_table.matrix)
println(CouplingTable(deltas, prior_table.amino_acids))

println("  Max increase: ", round(maximum(deltas), digits=4))
println("  Max decrease: ", round(minimum(deltas), digits=4))
println("  Number with |Δ| > 0.05: ", sum(abs.(deltas) .> 0.05))

function exploresynthesis(peptide::String, prior_table::CouplingTable, updated_coupling_table::CouplingTable)
    println("\nPredicting synthesis for: $peptide")
    println("Original coupling probabilities:")
    histogram = calculate_histogram(peptide, prior_table, num_simulations=10000)
    sorted_seqs = sort(collect(histogram), by=x->x[2], rev=true)
    total = sum(values(histogram))

    println("Top 10 predicted sequences:")
    for (i, (seq, count)) in enumerate(sorted_seqs[1:min(10, length(sorted_seqs))])
        prob = count / total
        mass = calculate_peptide_mass(seq)
        println(@sprintf("  %2d. %*s  Count: %5d  Prob: %.4f  Mass: %.2f Da", 
                        i, length(peptide), seq, count, prob, mass))
    end

    println("\nUpdated coupling probabilities:")

    histogram = calculate_histogram(peptide, updated_coupling_table, num_simulations=10000)
    sorted_seqs = sort(collect(histogram), by=x->x[2], rev=true)
    total = sum(values(histogram))

    println("Top 10 predicted sequences:")
    for (i, (seq, count)) in enumerate(sorted_seqs[1:min(10, length(sorted_seqs))])
        prob = count / total
        mass = calculate_peptide_mass(seq)
        println(@sprintf("  %2d. %*s  Count: %5d  Prob: %.4f  Mass: %.2f Da", 
                        i, length(peptide), seq, count, prob, mass))
    end
end

exploresynthesis("FISH", prior_table, updated_coupling_table)
exploresynthesis("SELLTPLGIDLDEW", prior_table, updated_coupling_table)

