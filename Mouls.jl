using Statistics
using LinearAlgebra


"""
    Mouls

Predict synthesis of peptide sequences from Monte-Carlo sampling of coupling probabilities.
"""
module Mouls 

using Distributions
using Printf

include("couplings.jl")

export predict_synthesis, calculate_histogram, CouplingTable, calculate_peptide_mass, get_amino_acid_mass


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
    
    for (i, aa) in enumerate(peptide |> reverse) # reverse for N-to-C actual synthesis order
        
        # TRUNCATION
        if rand()>1.0-i*0.001 # from Young1990, about a 1% chance of 'incomplete' with each additional aa
            break
        end

        # COUPLING EFFICIENCY
        if rand() < get_coupling_prob(coupling_table, aa, aa)
            push!(result, aa) # coupling successful!
        end
    end
    
    return String(result) |> reverse # back to FASTA order
end

"""
    predict_synthesis(peptide::String, coupling_table::CouplingTable; 
                     num_simulations::Int=1000)

Main function to predict synthesis of peptide sequence.
"""
function predict_synthesis(peptide::String, coupling_table::CouplingTable; 
                         num_simulations::Int=1_000_000)
    
    println("Predicting synthesis for peptide: $peptide")
    println("Original peptide mass: $(round(calculate_peptide_mass(peptide), digits=2)) Da")
    println("Simulations: $num_simulations")
    
    # Calculate histogram
    histogram = calculate_histogram(peptide, coupling_table, 
                                  num_simulations=num_simulations)
    
    # Sort by frequency (yield)
    sorted_sequences = sort(collect(histogram), by=x->x[2], rev=true)
    
    println("\nTop 20 yields:")
    println("Sequence\t\tCount\tProbability\tMass[inc. termini] (Da)")
    println("-" ^ 65)
    
    total_count = sum(values(histogram))
 
    total_prob = 0.0
    for (sequence, count) in sorted_sequences[1:min(20, length(sorted_sequences))]
        prob = count / total_count
        mass = calculate_peptide_mass(sequence, include_termini=true)
        println("$(lpad(sequence,length(peptide)))\t\t$count\t$(round(prob, digits=4))\t$(round(mass, digits=2))")
        total_prob += prob

        
    end

    println("Total yield / probability of synthesis: $(round(total_prob, digits=4))")
    
    return histogram
end


end # module

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    using .Mouls
    
    # Create example coupling table
    coupling_table = Mouls.create_coupling_table()
    
    println("CouplingTable structure:")
    println(coupling_table)
    
    # Test peptide
    KB09 = "ILKKLKKLAAPAL"
    KB11 = "LILKPLKLLKCLKKL"
    testpep=KB11

    # Predict synthesis
    result = predict_synthesis(testpep, coupling_table, num_simulations=10_000_000)
    println("\nTotal unique sequences generated: $(length(result))")
end 

