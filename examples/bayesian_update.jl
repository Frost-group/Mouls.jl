# Example workflow while developing the Bayesian (Sort of) update code for coupling tables.
#   Don't ask me why the nautical theme; I was just feeling it.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Mouls
using Printf

println("="^70)
println("Bayesian Coupling Matrix Update - Example Workflow")
println("="^70)

println("\n1. Load existing coupling table")
coupling_table = create_coupling_table("contextual_couplings_Young1990")
println(coupling_table)

println("\n2. Initialize Bayesian prior from coupling table")
prior = initialize_prior_from_coupling_table(coupling_table, prior_strength=100.0)
println(prior)

println("\nPrior coupling probabilities (from raw experimental data):")
prior_table = from_bayesian_prior(prior)
println(prior_table)

println("\n3. Generate synthetic observations (simulating real synthesis data)")
println("Using prior-based coupling probabilities for simulation consistency...")
target_peptide = "MARLIN"
obs1 = create_synthetic_observation(
    "MARLIN",
    prior_table,
    num_simulations=500,
    truncation_rate=0.001
)
println("\nSynthetic Observation 1:")
println(obs1)

obs2 = create_synthetic_observation(
    "LIMPIT",
    prior_table,
    num_simulations=500,
    truncation_rate=0.001
)
println("\nSynthetic Observation 2:")
println(obs2)

# Synthetic observation: "AAAAAAAA" synthesized with 100% yield, 
# to drive [A,A] entry, while debugging the optimiser
# NB: Don't use Nealder-Mead for 400 parameters. It don't work. 
println("\nSynthetic Observation 3 (100% yield):")
obs3 = SequenceObservation(
    "AAAAAAAA",
    Dict("AAAAAAAA" => 500), # All attempts resulted in correct product
    500
)
println(obs3)

obs4 = SequenceObservation(
    "CRANKYMEGAFISHSTINGRAY",
    Dict("CRANKYMEGAFISHSTINGRAY" => 500),
    500
)
println(obs4)   


println("\n4. Update posterior using MAP estimation")
println("This will run optimization to find parameters that best explain the observations...")
posterior = update_posterior(
    prior,
    [obs1, obs2, obs3, obs4],
    method=:map,
    num_likelihood_sims=1000,
    max_iterations=50
)
println("\nUpdated Posterior:")
println(posterior)

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

println("\n6. Use updated coupling table for new predictions")

for peptide in ["CRANKYMEGAFISHSTINGRAY", "FISH"]

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

println("\n" * "="^70)
println("Example workflow complete!")
println("="^70)

