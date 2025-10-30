using Distributions
using Optim
using Statistics

"""
    BayesianCouplingPrior

Stores Beta prior/posterior parameters for coupling probabilities and truncation rate.

Fields:
- `coupling_alpha::Matrix{Float64}`: Beta distribution α parameters for 20×20 coupling matrix (successes)
- `coupling_beta::Matrix{Float64}`: Beta distribution β parameters for 20×20 coupling matrix (failures)
- `truncation_alpha::Float64`: Beta distribution α for per-position truncation probability
- `truncation_beta::Float64`: Beta distribution β for per-position truncation probability
- `amino_acids::Vector{Char}`: Amino acid ordering for matrix indexing
"""
@kwdef struct BayesianCouplingPrior
    coupling_alpha::Matrix{Float64}
    coupling_beta::Matrix{Float64}
    truncation_alpha::Float64
    truncation_beta::Float64
    amino_acids::Vector{Char}
    
    function BayesianCouplingPrior(coupling_alpha, coupling_beta, truncation_alpha, truncation_beta, amino_acids)
        @assert size(coupling_alpha) == size(coupling_beta)
        @assert size(coupling_alpha, 1) == size(coupling_alpha, 2) == length(amino_acids)
        @assert truncation_alpha > 0 && truncation_beta > 0
        new(coupling_alpha, coupling_beta, truncation_alpha, truncation_beta, amino_acids)
    end
end

Base.show(io::IO, prior::BayesianCouplingPrior) = begin
    n = length(prior.amino_acids)
    println(io, "BayesianCouplingPrior ($(n)×$(n) coupling matrix)")
    trunc_mean = prior.truncation_alpha / (prior.truncation_alpha + prior.truncation_beta)
    println(io, "  Truncation probability (mean): $(round(trunc_mean, digits=5))")
    println(io, "  Total prior strength (coupling): $(round(sum(prior.coupling_alpha + prior.coupling_beta) / (n*n), digits=1)) equiv. observations per cell")
end

"""
    initialize_prior_from_coupling_table(coupling_table::CouplingTable; 
                                        prior_strength::Float64=100.0,
                                        truncation_rate::Float64=0.001)

Convert a CouplingTable to a BayesianCouplingPrior using actual experimental counts.

Uses raw counts from Young et al. (1990) data when available, scaled to `prior_strength`.
Falls back to coupling probabilities for missing data 
    (since we have that fudge code already in the couplings.jl file).

# Arguments
- `coupling_table`: The coupling table to convert
- `prior_strength`: Target total of α + β for each cell (default: 100.0)
- `truncation_rate`: Prior mean for truncation probability
"""
function initialize_prior_from_coupling_table(coupling_table::CouplingTable; 
                                               prior_strength::Float64=100.0,
                                                truncation_rate::Float64=0.001)
    n = length(coupling_table.amino_acids)
    coupling_alpha = zeros(n, n)
    coupling_beta = zeros(n, n)
    
    for i in 1:n
        for j in 1:n
            aa_i = coupling_table.amino_acids[i]
            aa_j = coupling_table.amino_acids[j]
            
            raw_data = get(contextual_couplings_Young1990_raw, (aa_i, aa_j), nothing)
            
            if !isnothing(raw_data) && !ismissing(raw_data)
                incomplete, total = raw_data
                complete = total - incomplete
                # Scale raw counts to desired prior strength while preserving proportions
                # Add Laplace smoothing (+1 to each) before scaling
                total_with_smoothing = total + 2
                coupling_alpha[i, j] = (Float64(complete) + 1.0) * prior_strength / total_with_smoothing
                coupling_beta[i, j] = (Float64(incomplete) + 1.0) * prior_strength / total_with_smoothing
            else
                # Fallback for missing data: use coupling probability
                p = coupling_table.matrix[i, j]
                coupling_alpha[i, j] = p * prior_strength
                coupling_beta[i, j] = (1.0 - p) * prior_strength
            end
        end
    end
    
    trunc_alpha = truncation_rate * prior_strength
    trunc_beta = (1.0 - truncation_rate) * prior_strength
    
    return BayesianCouplingPrior(
        coupling_alpha=coupling_alpha,
        coupling_beta=coupling_beta,
        truncation_alpha=trunc_alpha,
        truncation_beta=trunc_beta,
        amino_acids=copy(coupling_table.amino_acids)
    )
end

"""
    get_coupling_means(prior::BayesianCouplingPrior) -> Matrix{Float64}

Extract mean coupling probabilities from Beta distributions: E[p] = α/(α+β).
"""
function get_coupling_means(prior::BayesianCouplingPrior)
    return prior.coupling_alpha ./ (prior.coupling_alpha .+ prior.coupling_beta)
end

"""
    get_truncation_mean(prior::BayesianCouplingPrior) -> Float64

Extract mean truncation probability from Beta distribution.
"""
function get_truncation_mean(prior::BayesianCouplingPrior)
    return prior.truncation_alpha / (prior.truncation_alpha + prior.truncation_beta)
end

"""
    from_bayesian_prior(prior::BayesianCouplingPrior) -> CouplingTable

Convert BayesianCouplingPrior back to CouplingTable using posterior means.
Clamps values to [0,1] to handle numerical precision issues.
"""
function from_bayesian_prior(prior::BayesianCouplingPrior)
    coupling_means = get_coupling_means(prior)
    # Clamp to [0,1] to handle numerical precision issues from optimization
    coupling_means_clamped = clamp.(coupling_means, 0.0, 1.0)
    return CouplingTable(coupling_means_clamped, prior.amino_acids)
end

"""
    log_likeligood(prior::BayesianCouplingPrior, 
                        observations::Vector{SynthesisObservation};
                        num_simulations::Int=1000)

Calculate log-likelihood of observed synthesis data given coupling parameters.

For each observation, we:
1. Generate synthetic histogram using current parameters
2. Compare to observed histogram using a distance metric
3. Convert distance to approximate likelihood

TODO: Implement more sophisticated likelihood estimation? Particle filtering? Importance sampling?
"""
function log_likeligood(prior::BayesianCouplingPrior, 
                             observations::Vector{SynthesisObservation};
                             num_simulations::Int=1000)
    coupling_table = from_bayesian_prior(prior)
    trunc_rate = get_truncation_mean(prior)
    
    total_log_likelihood = 0.0
    
    for obs in observations
        predicted_hist = calculate_histogram_with_truncation(
            obs.target_sequence, 
            coupling_table,
            trunc_rate,
            num_simulations=num_simulations
        )
        
        log_lik = compare_histograms(predicted_hist, obs.observed_histogram, obs.total_count)
        total_log_likelihood += log_lik
    end
    
    return total_log_likelihood
end

"""
    calculate_histogram_with_truncation(peptide::String, 
                                       coupling_table::CouplingTable,
                                       truncation_rate::Float64;
                                       num_simulations::Int=1000)

Modified histogram calculation with parameterized truncation rate.
"""
function calculate_histogram_with_truncation(peptide::String, 
                                            coupling_table::CouplingTable,
                                            truncation_rate::Float64;
                                            num_simulations::Int=1000)
    sequence_counts = Dict{String, Int}()
    
    for _ in 1:num_simulations
        synthesised_sequence = generate_synthesised_sequence_with_truncation(
            peptide, 
            coupling_table, 
            truncation_rate
        )
        sequence_counts[synthesised_sequence] = get(sequence_counts, synthesised_sequence, 0) + 1
    end
    
    return sequence_counts
end

"""
    generate_synthesised_sequence_with_truncation(peptide::String, 
                                                  coupling_table::CouplingTable,
                                                  truncation_rate::Float64)

Generate synthesis with parameterized truncation rate instead of fixed 0.001.
"""
function generate_synthesised_sequence_with_truncation(peptide::String, 
                                                       coupling_table::CouplingTable,
                                                       truncation_rate::Float64)
    result = Char[]
    synth_order = reverse(peptide)

    for (i, aa) in enumerate(synth_order) 
        if i == 1
            push!(result, aa) 
            continue 
        end
        
        if rand() > 1.0 - i * truncation_rate
            break
        end

        carboxyl_aa = synth_order[i-1]  # Previous aa (carboxyl end available for coupling)
        amine_aa = synth_order[i]       # Current aa being added (amine end couples)
        if rand() < get_coupling_prob(coupling_table, amine_aa, carboxyl_aa)
            push!(result, amine_aa)
        end
    end
    
    return String(result) |> reverse
end

"""
    compare_histograms(predicted::Dict{String,Int}, 
                      observed::Dict{String,Int},
                      total_observed::Int) -> Float64

Compare predicted and observed histograms and return approximate log-likelihood.

Uses multinomial likelihood approximation. For sequences with zero predicted probability,
adds pseudocount to avoid -Inf.
"""
function compare_histograms(predicted::Dict{String,Int}, 
                           observed::Dict{String,Int},
                           total_observed::Int)
    total_predicted = sum(values(predicted))
    log_likelihood = 0.0
    
    pseudocount = 1e-10
    
    for (seq, obs_count) in observed
        pred_count = get(predicted, seq, 0)
        pred_prob = (pred_count + pseudocount) / (total_predicted + pseudocount * length(predicted))
        
        log_likelihood += obs_count * log(pred_prob)
    end
    
    return log_likelihood
end

"""
    log_prior_density(prior::BayesianCouplingPrior,
                     coupling_probs::Matrix{Float64},
                     truncation_prob::Float64) -> Float64

Calculate log prior density for given parameters under Beta priors.
"""
function log_prior_density(prior::BayesianCouplingPrior,
                          coupling_probs::Matrix{Float64},
                          truncation_prob::Float64)
    log_prior = 0.0
    
    n = size(coupling_probs, 1)
    for i in 1:n
        for j in 1:n
            alpha = prior.coupling_alpha[i, j]
            beta = prior.coupling_beta[i, j]
            p = coupling_probs[i, j]
            
            if p <= 0.0 || p >= 1.0
                return -Inf
            end
            
            log_prior += (alpha - 1) * log(p) + (beta - 1) * log(1 - p)
        end
    end
    
    if truncation_prob <= 0.0 || truncation_prob >= 1.0
        return -Inf
    end
    
    log_prior += (prior.truncation_alpha - 1) * log(truncation_prob)
    log_prior += (prior.truncation_beta - 1) * log(1 - truncation_prob)
    
    return log_prior
end

"""
    update_posterior(prior::BayesianCouplingPrior,
                    observations::Vector{SynthesisObservation};
                    method=:map,
                    num_likelihood_sims::Int=500,
                    max_iterations::Int=100,
                    BELIEF_STRENGTH::Float64=100.0) -> BayesianCouplingPrior

Update coupling prior based on observed synthesis data using MAP estimation.

HERE BE DRAGONS.
"""
function update_posterior(prior::BayesianCouplingPrior,
                         observations::Vector{SynthesisObservation};
                         method=:map,
                         num_likelihood_sims::Int=500,
                         max_iterations::Int=100,
                         BELIEF_STRENGTH::Float64=100.0)

    if method != :map
        error("Only MAP estimation currently implemented. Set method=:map")
    end
    
    println("Starting MAP estimation...")
    println("  Prior coupling mean: $(round(mean(get_coupling_means(prior)), digits=4))")
    println("  Prior truncation mean: $(round(get_truncation_mean(prior), digits=6))")
    println("  Observations: $(length(observations))")
    println("  Likelihood simulations per evaluation: $num_likelihood_sims")
    
    initial_coupling_means = get_coupling_means(prior)
    initial_trunc = get_truncation_mean(prior)
    
    n = length(prior.amino_acids)
    n_params = n * n + 1
    
    initial_params = vcat(vec(initial_coupling_means), initial_trunc)
    initial_params = clamp.(initial_params, 0.002, 0.998)
    
    function neg_log_posterior(params)
        coupling_matrix = reshape(params[1:n*n], n, n)
        trunc_prob = params[end]
        
        if any(x -> x <= 0.0 || x >= 1.0, params) # blow up if probs hit (0,1)
            return Inf
        end
        
        temp_prior = BayesianCouplingPrior(
            coupling_alpha=coupling_matrix .* BELIEF_STRENGTH,
            coupling_beta=(1.0 .- coupling_matrix) .* BELIEF_STRENGTH,
            truncation_alpha=trunc_prob * BELIEF_STRENGTH,
            truncation_beta=(1.0 - trunc_prob) * BELIEF_STRENGTH,
            amino_acids=prior.amino_acids
        )
        
        log_lik = log_likeligood(temp_prior, observations, 
                                       num_simulations=num_likelihood_sims)
        log_prior = log_prior_density(prior, coupling_matrix, trunc_prob)
        
        return -(log_lik + log_prior)
    end
    
    lower = fill(0.001, n_params)
    upper = fill(0.999, n_params)
    
    result = optimize(
        neg_log_posterior,
        lower,
        upper,
        initial_params,
        ParticleSwarm(lower=lower, upper=upper, n_particles=min(50, 2*n_params)),
        Optim.Options(iterations=max_iterations, show_trace=false)
    )
    
    optimal_params = Optim.minimizer(result)
    optimal_coupling = reshape(optimal_params[1:n*n], n, n)
    optimal_trunc = optimal_params[end]
     
    println("  Optimization Results:")
    println("    Converged: $(Optim.converged(result))")
    println("    Iterations: $(Optim.iterations(result)) / $(max_iterations)")
    println("    Function evaluations: $(Optim.f_calls(result))")
    println("    Final objective value: $(round(Optim.minimum(result), digits=2))")
    println("    Final coupling mean: $(round(mean(optimal_coupling), digits=4))")
    println("    Final coupling std: $(round(std(optimal_coupling), digits=4))")
    println("    Final truncation prob: $(round(optimal_trunc, digits=6))")
    println("    Coupling range: [$(round(minimum(optimal_coupling), digits=4)), $(round(maximum(optimal_coupling), digits=4))]")
    
    prior_strength = prior.coupling_alpha .+ prior.coupling_beta
    
    updated_prior = BayesianCouplingPrior(
        coupling_alpha=optimal_coupling .* prior_strength,
        coupling_beta=(1.0 .- optimal_coupling) .* prior_strength,
        truncation_alpha=optimal_trunc * (prior.truncation_alpha + prior.truncation_beta),
        truncation_beta=(1.0 - optimal_trunc) * (prior.truncation_alpha + prior.truncation_beta),
        amino_acids=prior.amino_acids
    )
    
    return updated_prior
end

"""
    create_synthetic_observation(target_sequence::String, 
                                 coupling_table::CouplingTable;
                                 num_simulations::Int=1000,
                                 truncation_rate::Float64=0.001) -> SynthesisObservation

Create a synthetic observation by running Monte Carlo synthesis simulations.
Useful for testing and validation; these observations should NOT change the coupling table, 
as they are predictions from the coupling table!
"""
function create_synthetic_observation(target_sequence::String, 
                                     coupling_table::CouplingTable;
                                     num_simulations::Int=1000,
                                     truncation_rate::Float64=0.001)
    histogram = calculate_histogram_with_truncation(
        target_sequence,
        coupling_table,
        truncation_rate,
        num_simulations=num_simulations
    )
    
    return SynthesisObservation(
        target_sequence=target_sequence,
        observed_histogram=histogram,
        total_count=num_simulations
    )
end

