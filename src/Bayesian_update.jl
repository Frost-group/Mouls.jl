using Distributions
using Optim
using Statistics

# ============================================================================
# BAYESIAN PRIOR/POSTERIOR REPRESENTATION
# ============================================================================

@kwdef struct BayesianCouplingPrior
    coupling_α::Matrix{Float64}
    coupling_β::Matrix{Float64}
    truncation_α::Float64
    truncation_β::Float64
    amino_acids::Vector{Char}
    
    function BayesianCouplingPrior(coupling_α, coupling_β, truncation_α, truncation_β, amino_acids)
        @assert size(coupling_α) == size(coupling_β)
        @assert size(coupling_α, 1) == size(coupling_α, 2) == length(amino_acids)
        @assert truncation_α > 0 && truncation_β > 0
        new(coupling_α, coupling_β, truncation_α, truncation_β, amino_acids)
    end
end

Base.show(io::IO, prior::BayesianCouplingPrior) = begin
    n = length(prior.amino_acids)
    println(io, "BayesianCouplingPrior ($(n)×$(n) coupling matrix)")
    θ_trunc = prior.truncation_α / (prior.truncation_α + prior.truncation_β)
    println(io, "  Truncation probability (mean): $(round(θ_trunc, digits=5))")
    println(io, "  Total prior strength (coupling): $(round(sum(prior.coupling_α + prior.coupling_β) / (n*n), digits=1)) equiv. observations per cell")
end

# ============================================================================
# PRIOR INITIALIZATION
# ============================================================================

function initialize_prior_from_coupling_table(coupling_table::CouplingTable; 
                                               prior_strength::Float64=100.0,
                                               truncation_rate::Float64=0.001)
    n = length(coupling_table.amino_acids)
    coupling_α = zeros(n, n)
    coupling_β = zeros(n, n)
    
    for i in 1:n, j in 1:n
        aa_i = coupling_table.amino_acids[i]
        aa_j = coupling_table.amino_acids[j]
        
        raw_data = get(contextual_couplings_Young1990_raw, (aa_i, aa_j), nothing)
        
        if !isnothing(raw_data) && !ismissing(raw_data)
            incomplete, total = raw_data
            complete = total - incomplete
            total_smoothed = total + 2
            coupling_α[i, j] = (Float64(complete) + 1.0) * prior_strength / total_smoothed
            coupling_β[i, j] = (Float64(incomplete) + 1.0) * prior_strength / total_smoothed
        else
            p = coupling_table.matrix[i, j]
            coupling_α[i, j] = p * prior_strength
            coupling_β[i, j] = (1.0 - p) * prior_strength
        end
    end
    
    return BayesianCouplingPrior(
        coupling_α=coupling_α,
        coupling_β=coupling_β,
        truncation_α=truncation_rate * prior_strength,
        truncation_β=(1.0 - truncation_rate) * prior_strength,
        amino_acids=copy(coupling_table.amino_acids)
    )
end

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

get_coupling_means(prior::BayesianCouplingPrior) = 
    prior.coupling_α ./ (prior.coupling_α .+ prior.coupling_β)

get_truncation_mean(prior::BayesianCouplingPrior) = 
    prior.truncation_α / (prior.truncation_α + prior.truncation_β)

function from_bayesian_prior(prior::BayesianCouplingPrior)
    coupling_means = get_coupling_means(prior)
    coupling_means_clamped = clamp.(coupling_means, 0.0, 1.0)
    return CouplingTable(coupling_means_clamped, prior.amino_acids)
end

# ============================================================================
# SYNTHESIS SIMULATION
# ============================================================================

function generate_synthesised_sequence_with_truncation(peptide::String, 
                                                       coupling_table::CouplingTable,
                                                       θ_trunc::Float64)
    result = Char[]
    synth_order = reverse(peptide)

    for (i, aa) in enumerate(synth_order) 
        i == 1 && (push!(result, aa); continue)
        
        rand() > 1.0 - i * θ_trunc && break

        carboxyl_aa = synth_order[i-1]
        amine_aa = synth_order[i]
        rand() < get_coupling_prob(coupling_table, amine_aa, carboxyl_aa) && push!(result, amine_aa)
    end
    
    return String(result) |> reverse
end

function calculate_histogram_with_truncation(peptide::String, 
                                            coupling_table::CouplingTable,
                                            θ_trunc::Float64;
                                            num_simulations::Int=1000)
    sequence_counts = Dict{String, Int}()
    
    for _ in 1:num_simulations
        seq = generate_synthesised_sequence_with_truncation(peptide, coupling_table, θ_trunc)
        sequence_counts[seq] = get(sequence_counts, seq, 0) + 1
    end
    
    return sequence_counts
end

function sequence_to_mass_histogram(peptide::String,
                                   coupling_table::CouplingTable,
                                   θ_trunc::Float64;
                                   num_simulations::Int=1000)
    mass_counts = Dict{Float64, Int}()
    
    for _ in 1:num_simulations
        seq = generate_synthesised_sequence_with_truncation(peptide, coupling_table, θ_trunc)
        mass = calculate_peptide_mass(seq, include_termini=true)
        mass_counts[mass] = get(mass_counts, mass, 0) + 1
    end
    
    return mass_counts
end


# ============================================================================
# LIKELIHOOD CALCULATION
# ============================================================================

function compare_histograms(predicted::Dict{String,Int}, 
                           observed::Dict{String,Int})
    total_predicted = sum(values(predicted))
    log_ℓ = 0.0
    pseudocount = 1e-10

    n_keys = length(union(keys(predicted), keys(observed)))
    
    for (key, obs_count) in observed
        pred_count = get(predicted, key, 0)
        pred_prob = (pred_count + pseudocount) / (total_predicted + pseudocount * n_keys)
        log_ℓ += obs_count * log(pred_prob)
    end
    
    return log_ℓ
end

function compare_histograms(predicted::Dict{Float64,Int}, 
                           observed::Dict{Float64,Int},
                           kernel_bandwidth::Float64)
    total_predicted = sum(values(predicted))
    log_ℓ = 0.0
    min_prob = 1e-10
    
    for (obs_mass, obs_count) in observed
        weighted_count = 0.0
        
        for (pred_mass, pred_count) in predicted
            mass_diff = obs_mass - pred_mass
            kernel_weight = exp(-0.5 * (mass_diff / kernel_bandwidth)^2)
            weighted_count += pred_count * kernel_weight
        end
        
        pred_prob = max(weighted_count / total_predicted, min_prob)
        log_ℓ += obs_count * log(pred_prob)
    end
    
    return log_ℓ
end

function log_likelihood(obs::SequenceObservation,
                       coupling_table::CouplingTable,
                       θ_trunc::Float64;
                       num_simulations::Int=1000)
    predicted_hist = calculate_histogram_with_truncation(
        obs.target_sequence, coupling_table, θ_trunc,
        num_simulations=num_simulations
    )
    return compare_histograms(predicted_hist, obs.observed_histogram)
end

function log_likelihood(obs::MassSpecObservation,
                       coupling_table::CouplingTable,
                       θ_trunc::Float64;
                       num_simulations::Int=1000,
                       kernel_bandwidth::Float64=1.0)
    predicted_hist = sequence_to_mass_histogram(
        obs.target_sequence, coupling_table, θ_trunc,
        num_simulations=num_simulations
    )
    return compare_histograms(predicted_hist, obs.mass_histogram, kernel_bandwidth)
end

function log_likelihood(prior::BayesianCouplingPrior, 
                       observations::Vector{<:AbstractObservation};
                       num_simulations::Int=1000,
                       kernel_bandwidth::Float64=1.0)
    coupling_table = from_bayesian_prior(prior)
    θ_trunc = get_truncation_mean(prior)
    
    total_log_ℓ = 0.0
    for obs in observations
        if obs isa MassSpecObservation
            total_log_ℓ += log_likelihood(obs, coupling_table, θ_trunc, 
                                         num_simulations=num_simulations,
                                         kernel_bandwidth=kernel_bandwidth)
        else
            total_log_ℓ += log_likelihood(obs, coupling_table, θ_trunc, 
                                         num_simulations=num_simulations)
        end
    end
    
    return total_log_ℓ
end

# ============================================================================
# PRIOR DENSITY
# ============================================================================

function log_prior_density(prior::BayesianCouplingPrior,
                          coupling_probs::Matrix{Float64},
                          θ_trunc::Float64)
    (θ_trunc <= 0.0 || θ_trunc >= 1.0) && return -Inf
    any(p -> p <= 0.0 || p >= 1.0, coupling_probs) && return -Inf
    
    log_p = 0.0
    n = size(coupling_probs, 1)
    
    for i in 1:n, j in 1:n
        α = prior.coupling_α[i, j]
        β = prior.coupling_β[i, j]
        p = coupling_probs[i, j]
        log_p += (α - 1) * log(p) + (β - 1) * log(1 - p)
    end
    
    log_p += (prior.truncation_α - 1) * log(θ_trunc)
    log_p += (prior.truncation_β - 1) * log(1 - θ_trunc)
    
    return log_p
end

# ============================================================================
# POSTERIOR UPDATE (MAP ESTIMATION)
# ============================================================================

function update_posterior(prior::BayesianCouplingPrior,
                         observations::Vector{<:AbstractObservation};
                         method=:map,
                         num_likelihood_sims::Int=500,
                         max_iterations::Int=100,
                         belief_strength::Float64=100.0,
                         kernel_bandwidth::Float64=1.0)

    method != :map && error("Only MAP estimation currently implemented. Set method=:map")
    
    println("Starting MAP estimation...")
    println("  Prior coupling mean: $(round(mean(get_coupling_means(prior)), digits=4))")
    println("  Prior truncation mean: $(round(get_truncation_mean(prior), digits=6))")
    println("  Observations: $(length(observations))")
    println("  Likelihood simulations per evaluation: $num_likelihood_sims")
    
    n = length(prior.amino_acids)
    n_params = n * n + 1
    
    initial_params = clamp.(vcat(vec(get_coupling_means(prior)), get_truncation_mean(prior)), 0.002, 0.998)
    
    function neg_log_posterior(params)
        any(x -> x <= 0.0 || x >= 1.0, params) && return Inf
        
        coupling_matrix = reshape(params[1:n*n], n, n)
        θ_trunc = params[end]
        
        temp_prior = BayesianCouplingPrior(
            coupling_α=coupling_matrix .* belief_strength,
            coupling_β=(1.0 .- coupling_matrix) .* belief_strength,
            truncation_α=θ_trunc * belief_strength,
            truncation_β=(1.0 - θ_trunc) * belief_strength,
            amino_acids=prior.amino_acids
        )
        
        log_ℓ = log_likelihood(temp_prior, observations, 
                              num_simulations=num_likelihood_sims,
                              kernel_bandwidth=kernel_bandwidth)
        log_p = log_prior_density(prior, coupling_matrix, θ_trunc)
        
        return -(log_ℓ + log_p)
    end

    println("    Initial objective value: $(round(neg_log_posterior(initial_params), digits=2))")
    println("    Initial kernel bandwidth: $kernel_bandwidth Da")
    println("    Initial coupling mean: $(round(mean(initial_params[1:n*n]), digits=4))")
    println("    Initial truncation prob: $(round(initial_params[end], digits=6))")
    println("    Initial coupling range: [$(round(minimum(initial_params[1:n*n]), digits=4)), $(round(maximum(initial_params[1:n*n]), digits=4))]")

    lower = fill(0.001, n_params)
    upper = fill(0.999, n_params)
    
    result = optimize(
        neg_log_posterior, lower, upper, initial_params,
        ParticleSwarm(lower=lower, upper=upper, n_particles=min(50, 2*n_params)),
        Optim.Options(iterations=max_iterations, show_trace=true, show_every=1)
    )
    
    optimal_params = Optim.minimizer(result)
    optimal_coupling = reshape(optimal_params[1:n*n], n, n)
    optimal_θ_trunc = optimal_params[end]
     
    println("  Optimization Results:")
    println("    Converged: $(Optim.converged(result))")
    println("    Iterations: $(Optim.iterations(result)) / $(max_iterations)")
    println("    Function evaluations: $(Optim.f_calls(result))")
    println("    Final objective value: $(round(Optim.minimum(result), digits=2))")
    println("    Final coupling mean: $(round(mean(optimal_coupling), digits=4))")
    println("    Final coupling std: $(round(std(optimal_coupling), digits=4))")
    println("    Final truncation prob: $(round(optimal_θ_trunc, digits=6))")
    println("    Coupling range: [$(round(minimum(optimal_coupling), digits=4)), $(round(maximum(optimal_coupling), digits=4))]")
    
    prior_strength = prior.coupling_α .+ prior.coupling_β
    
    return BayesianCouplingPrior(
        coupling_α=optimal_coupling .* prior_strength,
        coupling_β=(1.0 .- optimal_coupling) .* prior_strength,
        truncation_α=optimal_θ_trunc * (prior.truncation_α + prior.truncation_β),
        truncation_β=(1.0 - optimal_θ_trunc) * (prior.truncation_α + prior.truncation_β),
        amino_acids=prior.amino_acids
    )
end

# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

function create_synthetic_observation(target_sequence::String, 
                                     coupling_table::CouplingTable;
                                     num_simulations::Int=1000,
                                     truncation_rate::Float64=0.001)
    histogram = calculate_histogram_with_truncation(
        target_sequence, coupling_table, truncation_rate,
        num_simulations=num_simulations
    )
    
    return SequenceObservation(
        target_sequence=target_sequence,
        observed_histogram=histogram,
        total_count=num_simulations
    )
end

function create_synthetic_mass_observation(target_sequence::String,
                                          coupling_table::CouplingTable;
                                          num_simulations::Int=1000,
                                          truncation_rate::Float64=0.001)
    histogram = sequence_to_mass_histogram(
        target_sequence, coupling_table, truncation_rate,
        num_simulations=num_simulations
    )
    
    return MassSpecObservation(
        target_sequence=target_sequence,
        mass_histogram=histogram,
        total_count=num_simulations
    )
end

