using Statistics
using LinearAlgebra


"""
    Mouls

Predict synthesis of peptide sequences from Monte-Carlo sampling of coupling probabilities.
"""
module Mouls 

using Distributions
using Printf

# Amino acid mass lookup table (monoisotopic masses in Da)
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

    # Terminal groups
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
        @assert size(matrix, 1) == size(matrix, 2) == length(amino_acids) # gotta catch them all
        @assert all(x -> 0 ≤ x ≤ 1, matrix) # probabilities, innit 
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

export predict_synthesis, calculate_histogram, CouplingTable, calculate_peptide_mass, get_coupling_prob
export BayesianCouplingPrior, AbstractObservation, SequenceObservation, MassSpecObservation
export initialize_prior_from_coupling_table, update_posterior, from_bayesian_prior
export create_synthetic_observation, create_synthetic_mass_observation
export get_coupling_means, get_truncation_mean
export create_coupling_table, validate_synthesis_observation, is_subsequence_of
export calculate_histogram_with_truncation, sequence_to_mass_histogram
export log_likelihood, log_prior_density

"""
    calculate_histogram(peptide::String, coupling_table::CouplingTable; 
                       num_simulations::Int=1000)

Calculate expected histogram of sequences based on coupling table and synthesis model.
"""
function calculate_histogram(peptide::String, coupling_table::CouplingTable; 
                           num_simulations::Int=1000)
    
    # Initialize histogram
    sequence_counts = Dict{String, Int}()
    
    for _ in 1:num_simulations
        # Generate synthesised sequence
        synthesised_sequence = generate_synthesised_sequence(peptide, coupling_table)
        sequence_counts[synthesised_sequence] = get(sequence_counts, synthesised_sequence, 0) + 1 # uses get to set default 0 if not in dict
    end
    
    return sequence_counts
end

"""
    generate_synthesised_sequence(peptide::String, coupling_table::CouplingTable)

Generate a single synthesised sequence based on coupling probabilities.
"""
function generate_synthesised_sequence(peptide::String, coupling_table::CouplingTable)
    result = Char[]
    
    synth_order = reverse(peptide) # N-to-C order; how our synthesiser runs

    for (i, aa) in enumerate(synth_order) 
        if i==1 # always add the first aa; avoids issue with i-1 index
            push!(result, aa) 
            continue 
        end
        
        # TRUNCATION:
        # Young1990, about a 1% chance of 'incomplete' with each additional aa
        if rand()>1.0-i*0.001 
            break
        end

        # COUPLING EFFICIENCY
        # tables (potentially) context-dependent 
        # (i.e. carboxylic given amine you are adding onto)
        amine_aa=synth_order[i-1]
        carboxyl_aa=synth_order[i]
        if rand() < get_coupling_prob(coupling_table, amine_aa, carboxyl_aa)
            push!(result, carboxyl_aa) # coupling successful!
        end
    end
    
    return String(result) |> reverse # back to FASTA order
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
    
    # Calculate histogram
    histogram = calculate_histogram(peptide, coupling_table, 
                                  num_simulations=num_simulations)
    
    # Sort by frequency (yield)
    sorted_sequences = sort(collect(histogram), by=x->x[2], rev=true)
    
    println("\nTop $top yields:")
    println("Sequence\t\tCount\tProbability\tMass[inc. termini] (Da)\t2x\0.5x")
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

using ArgParse
if abspath(PROGRAM_FILE) == @__FILE__
    using .Mouls
    
    s = ArgParseSettings(description="Monte Carlo prediction of peptide synthesis", version="0.0.1", add_version=true)
    @add_arg_table! s begin
        "peptides"
            help = "Peptide sequence(s) to predict"
            nargs = '+'
            required = true
        "--simulations", "-n"
            help = "Number of MC simulations"
            arg_type = Int
            default = 1_000_000
        "--coupling", "-c"
            help = "Coupling table: contextfree_couplings, contextfree_couplings_Young1990, contextual_couplings_Young1990"
            arg_type = String
            default = "contextual_couplings_Young1990"
        "--top"
            help = "Number of top sequences to display"
            arg_type = Int
            default = 20
    end
    
    args = parse_args(s)
    coupling_table = Mouls.create_coupling_table(args["coupling"])
    
    for peptide in args["peptides"]
        result = predict_synthesis(peptide, coupling_table, num_simulations=args["simulations"], top=args["top"])
        println("\nTotal unique sequences: $(length(result))")
    end
end 

