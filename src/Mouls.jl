"""
    Mouls

Predict synthesis of peptide sequences from Monte-Carlo sampling of coupling probabilities.
"""
module Mouls 

using Distributions
using Printf
using Statistics
using LinearAlgebra

const AMINO_ACID_MASSES = Dict{Char, Float64}(
    'A' => 71.03711,  # Alanine
    'R' => 156.10111, # Arginine
    'N' => 114.04293, # Asparagine
    'D' => 115.02694, # Aspartic acid
    'C' => 103.00919, # Cysteine
    'Q' => 128.05858, # Glutamine
    'E' => 129.04259, # Glutamic acid
    'G' => 57.02146,  # Glycine
    'H' => 137.05891, # Histidine
    'I' => 113.08406, # Isoleucine
    'L' => 113.08406, # Leucine
    'K' => 128.09496, # Lysine
    'M' => 131.04049, # Methionine
    'F' => 147.06841, # Phenylalanine
    'P' => 97.05276,  # Proline
    'S' => 87.03203,  # Serine
    'T' => 101.04768, # Threonine
    'W' => 186.07931, # Tryptophan
    'Y' => 163.06333, # Tyrosine
    'V' => 99.06841,   # Valine
    'n' => 1.007825, # H+
    'c' => 17.00274 # OH-
)

function calculate_peptide_mass(sequence::String; include_termini::Bool=true)
    mass = 0.0
    
    for aa in sequence
        if haskey(AMINO_ACID_MASSES, aa)
            mass += AMINO_ACID_MASSES[aa]
        else
            error("Unknown amino acid: $aa")
        end
    end
    
    if include_termini
        mass += AMINO_ACID_MASSES['n'] + AMINO_ACID_MASSES['c']
    end

    return mass
end

"""
    CouplingTable

Structure to store amino acid coupling probabilities.
"""
struct CouplingTable
    matrix::Matrix{Float64}
    amino_acids::Vector{Char}
    
    function CouplingTable(matrix::Matrix{Float64}, amino_acids::Vector{Char})
        @assert size(matrix, 1) == size(matrix, 2) == length(amino_acids)
        @assert all(x -> 0 ≤ x ≤ 1, matrix)
        new(matrix, amino_acids)
    end
end

Base.show(io::IO, ct::CouplingTable) = begin
    aas = ct.amino_acids
    println(io, "CouplingTable (size $(length(aas))×$(length(aas)))")
    print(io, "    ")
    for aa in aas
        print(io, rpad(aa, 6))
    end
    println(io)
    for (i, aa) in enumerate(aas)
        print(io, rpad(aa, 4))
        for j in 1:length(aas)
            print(io, @sprintf("%.3f ", ct.matrix[i, j]))
        end
        println(io)
    end
end

"""
    get_coupling_prob(coupling_table::CouplingTable, aa1::Char, aa2::Char)

Get coupling probability between two amino acids.
"""
function get_coupling_prob(coupling_table::CouplingTable, aa1::Char, aa2::Char)
    i = findfirst(isequal(aa1), coupling_table.amino_acids)
    j = findfirst(isequal(aa2), coupling_table.amino_acids)
    
    if isnothing(i) || isnothing(j)
        return 0.0
    end
    
    return coupling_table.matrix[i, j]
end

include("couplings.jl")
include("synthesis_data.jl")
include("Bayesian_update.jl")

export contextual_couplings_Young1990_raw

export predict_synthesis, calculate_histogram, CouplingTable, calculate_peptide_mass, get_amino_acid_mass, get_coupling_prob
export BayesianCouplingPrior, SynthesisObservation
export initialize_prior_from_coupling_table, update_posterior, from_bayesian_prior
export load_synthesis_data_json, save_synthesis_data_json, create_synthetic_observation
export get_coupling_means, get_truncation_mean
export create_coupling_table, validate_synthesis_observation, is_subsequence_of
export calculate_histogram_with_truncation, compare_histograms, log_prior_density

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

