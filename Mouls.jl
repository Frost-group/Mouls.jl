using Statistics
using LinearAlgebra

"""
    Mouls

Predict synthesis of peptide sequences from Monte-Carlo sampling of coupling probabilities.
"""
module Mouls 

using Distributions
using Printf

export predict_synthesis, calculate_histogram, CouplingTable

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
        print(io, rpad(aa, 4))
    end
    println(io)
    for (i, aa) in enumerate(aas)
        print(io, rpad(aa, 4))
        for j in 1:length(aas)
            print(io, @sprintf("%.2f ", ct.matrix[i, j]))
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
                         num_simulations::Int=1000_000)
    
    println("Predicting synthesis for peptide: $peptide")
    println("Simulations: $num_simulations")
    
    # Calculate histogram
    histogram = calculate_histogram(peptide, coupling_table, 
                                  num_simulations=num_simulations)
    
    # Sort by frequency (yield)
    sorted_sequences = sort(collect(histogram), by=x->x[2], rev=true)
    
    println("\nTop 10 yields:")
    println("Sequence\t\tCount\tProbability")
    println("-" ^ 50)
    
    total_count = sum(values(histogram))
 
    total_prob = 0.0
    for (sequence, count) in sorted_sequences[1:min(10, length(sorted_sequences))]
        prob = count / total_count
        println("$sequence\t\t$count\t$(round(prob, digits=4))")
        total_prob += prob
    end

    println("Total yield / probability of synthesis: $(round(total_prob, digits=4))")
    
    return histogram
end

# Context free coupling probability 
contextfree_couplings = Dict(
    'A' => 0.99,  # Alanine
    'R' => 0.98,  # Arginine
    'N' => 0.98,  # Asparagine
    'D' => 0.98,  # Aspartic acid
    'C' => 0.95,  # Cysteine - oxidation ?
    'Q' => 0.98,  # Glutamine
    'E' => 0.98,  # Glutamic acid
    'G' => 0.99,  # Glycine
    'H' => 0.97,  # Histidine
    'I' => 0.99,  # Isoleucine
    'L' => 0.99,  # Leucine
    'K' => 0.98,  # Lysine
    'M' => 0.96,  # Methionine - oxidation sensitive ?
    'F' => 0.99,  # Phenylalanine
    'P' => 0.92,  # Proline - ring strain ?
    'S' => 0.99,  # Serine
    'T' => 0.98,  # Threonine
    'W' => 0.95,  # Tryptophan - indole ring sensitive ?
    'Y' => 0.98,  # Tyrosine - phenol group ?
    'V' => 0.99   # Valine
)

function create_coupling_table()
    # Standard amino acids; used to order matrix indices, dereferenced by a lookup
    amino_acids = "ARNDCQEGHILKMFPSTWYV" |> collect 
    n = length(amino_acids)
    coupling_matrix = zeros(n, n)

    # Set coupling probabilities based on amino acid synthesis difficulty
    for i in 1:n
        for j in 1:n
            # coupling_matrix[i, j] represents the probability of successfully adding amino acid i when the previous amino acid (context) is j.
            # For now, using flat probabilities (context-independent)
            coupling_matrix[i, j] = get(contextfree_couplings, amino_acids[i], 0.90)  # Default to 0.90 if not in dict
        end
    end
    
    return CouplingTable(coupling_matrix, amino_acids)
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

    # Predict synthesis
    result = predict_synthesis(KB09, coupling_table, num_simulations=1_000_000)
    println("\nTotal unique sequences generated: $(length(result))")
end 

