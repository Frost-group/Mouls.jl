"""
    Mouls

Predict synthesis of peptide sequences from Monte-Carlo sampling of coupling probabilities.
"""
module Mouls 

using Distributions
using Printf
using Statistics
using LinearAlgebra

include("couplings.jl")
include("synthesis_data.jl")
include("Bayesian_update.jl")

export contextual_couplings_Young1990_raw

export predict_synthesis, calculate_histogram, CouplingTable, calculate_peptide_mass, get_amino_acid_mass, get_coupling_prob
export BayesianCouplingPrior, AbstractObservation, SequenceObservation, MassSpecObservation
export initialize_prior_from_coupling_table, update_posterior, from_bayesian_prior
export create_synthetic_observation, create_synthetic_mass_observation
export get_coupling_means, get_truncation_mean
export create_coupling_table, validate_synthesis_observation, is_subsequence_of
export calculate_histogram_with_truncation, sequence_to_mass_histogram, compare_histograms, log_prior_density, log_likelihood

# Main MC bit 

"""
    calculate_histogram(peptide::String, coupling_table::CouplingTable; 
                       num_simulations::Int=1000)

Calculate expected histogram of sequences based on coupling table and synthesis model.
"""
function calculate_histogram(peptide::String, coupling_table::CouplingTable; 
                           num_simulations::Int=1000)
    
    sequence_counts = Dict{String, Int}()
    
    for _ in 1:num_simulations
        synthesised_sequence = generate_synthesised_sequence(peptide, coupling_table)
        sequence_counts[synthesised_sequence] = get(sequence_counts, synthesised_sequence, 0) + 1
    end
    
    return sequence_counts
end

"""
    generate_synthesised_sequence(peptide::String, coupling_table::CouplingTable)

Generate a single synthesised sequence based on coupling probabilities.
"""
function generate_synthesised_sequence(peptide::String, coupling_table::CouplingTable)
    result = Char[]
    
    synth_order = reverse(peptide)

    for (i, aa) in enumerate(synth_order) 
        if i==1
            push!(result, aa) 
            continue 
        end
        
        if rand()>1.0-i*0.001 
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
    predict_synthesis(peptide::String, coupling_table::CouplingTable; 
                     num_simulations::Int=1000, top::Int=20)

Main function to predict synthesis of peptide sequence.
"""
function predict_synthesis(peptide::String, coupling_table::CouplingTable; 
                         num_simulations::Int=1_000_000, top::Int=20)
    
    println("Predicting synthesis for peptide: $peptide")
    println("Target peptide mass: $(round(calculate_peptide_mass(peptide), digits=2)) Da")
    println("MC Synthesis Simulations: $num_simulations")
    
    histogram = calculate_histogram(peptide, coupling_table, 
                                  num_simulations=num_simulations)
    
    sorted_sequences = sort(collect(histogram), by=x->x[2], rev=true)
    
    println("\nTop $top yields:")
    println("Sequence\t\tCount\tProbability\tMass[inc. termini] (Da)\t2x\t0.5x")
    println("-" ^ 65)
    
    total_count = sum(values(histogram))
 
    total_prob = 0.0
    for (sequence, count) in sorted_sequences[1:min(top, length(sorted_sequences))]
        prob = count / total_count
        mass = calculate_peptide_mass(sequence, include_termini=true)
        @printf("%*s\t\t%d\t%.4f\t\t%.2f\t(%.2f,\t%.2f)\n", 
                length(peptide), sequence, count, prob, mass, 2*mass, mass/2)
        total_prob += prob
    end

    println("Total yield / probability of synthesis: $(round(total_prob, digits=4))")
    
    return histogram
end

end # module

