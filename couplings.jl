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

# Young1990 - Figure 2; high incomplete
contextfree_couplings_Young1990 = Dict(
    'H' => 0.75,  # Histidine
    'T' => 0.86,  # Threonine
    'R' => 0.87,  # Arginine
    'V' => 0.87,   # Valine
    'I' => 0.87,  # Isoleucine
    'Q' => 0.90,  # Glutamine
    'Y' => 0.91,  # Tyrosine - phenol group ?
    'N' => 0.92,  # Asparagine
    'W' => 0.92,  # Tryptophan - indole ring sensitive ?
    'E' => 0.92,  # Glutamic acid
    'P' => 0.92,  # Proline - ring strain ?
    'K' => 0.94,  # Lysine
    'L' => 0.94,  # Leucine
    'S' => 0.94,  # Serine
    'F' => 0.94,  # Phenylalanine
    'A' => 0.94,  # Alanine
    'C' => 0.95,  # Cysteine - oxidation ?
    'M' => 0.95,  # Methionine - oxidation sensitive ?
    'D' => 0.97,  # Aspartic acid
    'G' => 0.97  # Glycine
)



# Coupling data digitized from Young et al. (1990) Figure 2
# Key: (amine, carboxyl), Value: (number_incomplete, total_number)
# ND (not determined) entries are represented as `missing`

const contextual_couplings_Young1990 = Dict{Tuple{Char,Char}, Union{Float64,Missing}}(
    # Q row
    ('Q', 'H') => 4/6,
    ('Q', 'T') => 7/20,
    ('Q', 'R') => 7/16,
    ('Q', 'V') => 5/12,
    ('Q', 'I') => 4/8,
    ('Q', 'Q') => 11/19,
    ('Q', 'Y') => 2/8,
    ('Q', 'N') => 4/14,
    ('Q', 'W') => 2/5,
    ('Q', 'E') => 4/16,
    ('Q', 'P') => 5/17,
    ('Q', 'K') => 1/5,
    ('Q', 'L') => 6/27,
    ('Q', 'S') => 5/14,
    ('Q', 'F') => 2/7,
    ('Q', 'A') => 8/19,
    ('Q', 'C') => 1/4,
    ('Q', 'M') => 0/1,
    ('Q', 'D') => 4/12,
    ('Q', 'G') => 2/20,
    # L row
    ('L', 'H') => 7/11,
    ('L', 'T') => 16/29,
    ('L', 'R') => 17/36,
    ('L', 'V') => 5/15,
    ('L', 'I') => 3/10,
    ('L', 'Q') => 10/26,
    ('L', 'Y') => 1/8,
    ('L', 'N') => 6/21,
    ('L', 'W') => 5/9,
    ('L', 'E') => 8/32,
    ('L', 'P') => 12/31,
    ('L', 'K') => 9/24,
    ('L', 'L') => 14/40,
    ('L', 'S') => 5/29,
    ('L', 'F') => 0/7,
    ('L', 'A') => 7/26,
    ('L', 'C') => 2/12,
    ('L', 'M') => 0/4,
    ('L', 'D') => 5/27,
    ('L', 'G') => 9/29,
    # A row
    ('A', 'H') => 1/2,
    ('A', 'T') => 6/16,
    ('A', 'R') => 11/22,
    ('A', 'V') => 3/10,
    ('A', 'I') => 6/10,
    ('A', 'Q') => 8/16,
    ('A', 'Y') => 4/14,
    ('A', 'N') => 6/13,
    ('A', 'W') => 1/4,
    ('A', 'E') => 9/21,
    ('A', 'P') => 9/30,
    ('A', 'K') => 9/28,
    ('A', 'L') => 8/29,
    ('A', 'S') => 4/29,
    ('A', 'F') => 1/11,
    ('A', 'A') => 13/32,
    ('A', 'C') => 4/14,
    ('A', 'M') => 1/4,
    ('A', 'D') => 5/20,
    ('A', 'G') => 3/30,
    # R row
    ('R', 'H') => 4/6,
    ('R', 'T') => 8/19,
    ('R', 'R') => 25/54,
    ('R', 'V') => 9/16,
    ('R', 'I') => 4/19,
    ('R', 'Q') => 2/17,
    ('R', 'Y') => 3/9,
    ('R', 'N') => 3/12,
    ('R', 'W') => 1/8,
    ('R', 'E') => 7/28,
    ('R', 'P') => 11/24,
    ('R', 'K') => 6/32,
    ('R', 'L') => 10/36,
    ('R', 'S') => 9/33,
    ('R', 'F') => 6/16,
    ('R', 'A') => 6/24,
    ('R', 'C') => 6/13,
    ('R', 'M') => 3/6,
    ('R', 'D') => 4/29,
    ('R', 'G') => 7/41,
    # I row
    ('I', 'H') => 1/4,
    ('I', 'T') => 5/11,
    ('I', 'R') => 15/22,
    ('I', 'V') => 5/10,
    ('I', 'I') => 2/8,
    ('I', 'Q') => 7/15,
    ('I', 'Y') => 2/19,
    ('I', 'N') => 3/10,
    ('I', 'W') => 0/3,
    ('I', 'E') => 2/10,
    ('I', 'P') => 5/13,
    ('I', 'K') => 3/21,
    ('I', 'L') => 3/13,
    ('I', 'S') => 1/15,
    ('I', 'F') => 2/4,
    ('I', 'A') => 0/13,
    ('I', 'C') => 0/6,
    ('I', 'M') => 2/5,
    ('I', 'D') => 8/23,
    ('I', 'G') => 2/14,
    # K row
    ('K', 'H') => 5/8,
    ('K', 'T') => 7/17,
    ('K', 'R') => 9/29,
    ('K', 'V') => 7/11,
    ('K', 'I') => 8/16,
    ('K', 'Q') => 4/11,
    ('K', 'Y') => 6/19,
    ('K', 'N') => 6/19,
    ('K', 'W') => 1/6,
    ('K', 'E') => 8/30,
    ('K', 'P') => 5/29,
    ('K', 'K') => 16/43,
    ('K', 'L') => 8/36,
    ('K', 'S') => 5/38,
    ('K', 'F') => 1/7,
    ('K', 'A') => 7/29,
    ('K', 'C') => 2/16,
    ('K', 'M') => 2/10,
    ('K', 'D') => 0/14,
    ('K', 'G') => 7/23,
    # V row
    ('V', 'H') => 3/4,
    ('V', 'T') => 7/17,
    ('V', 'R') => 7/15,
    ('V', 'V') => 4/17,
    ('V', 'I') => 6/20,
    ('V', 'Q') => 5/9,
    ('V', 'Y') => 2/3,
    ('V', 'N') => 1/6,
    ('V', 'W') => 0/3,
    ('V', 'E') => 8/17,
    ('V', 'P') => 6/31,
    ('V', 'K') => 6/24,
    ('V', 'L') => 4/17,
    ('V', 'S') => 6/21,
    ('V', 'F') => 1/4,
    ('V', 'A') => 0/9,
    ('V', 'C') => 0/6,
    ('V', 'M') => 1/2,
    ('V', 'D') => 0/15,
    ('V', 'G') => 5/21,
    # M row
    ('M', 'H') => 1/3,
    ('M', 'T') => 2/5,
    ('M', 'R') => 1/8,
    ('M', 'V') => 1/6,
    ('M', 'I') => 3/6,
    ('M', 'Q') => 1/4,
    ('M', 'Y') => 3/7,
    ('M', 'N') => 0/4,
    ('M', 'W') => 1/5,
    ('M', 'E') => 1/3,
    ('M', 'P') => 1/8,
    ('M', 'K') => 2/10,
    ('M', 'L') => 3/9,
    ('M', 'S') => 2/3,
    ('M', 'F') => 4/7,
    ('M', 'A') => 1/7,
    ('M', 'C') => 0/3,
    ('M', 'M') => 1/4,
    ('M', 'D') => 0/2,
    ('M', 'G') => 1/7,
    # E row
    ('E', 'H') => 3/8,
    ('E', 'T') => 4/22,
    ('E', 'R') => 7/23,
    ('E', 'V') => 10/18,
    ('E', 'I') => 4/18,
    ('E', 'Q') => 5/20,
    ('E', 'Y') => 7/12,
    ('E', 'N') => 1/18,
    ('E', 'W') => 0/5,
    ('E', 'E') => 7/34,
    ('E', 'P') => 4/26,
    ('E', 'K') => 8/22,
    ('E', 'L') => 14/38,
    ('E', 'S') => 2/15,
    ('E', 'F') => 1/6,
    ('E', 'A') => 5/19,
    ('E', 'C') => 3/12,
    ('E', 'M') => 3/14,
    ('E', 'D') => 3/17,
    ('E', 'G') => 2/13,
    # T row
    ('T', 'H') => 0/4,
    ('T', 'T') => 6/19,
    ('T', 'R') => 5/25,
    ('T', 'V') => 8/14,
    ('T', 'I') => 10/28,
    ('T', 'Q') => 0/8,
    ('T', 'Y') => 4/13,
    ('T', 'N') => 4/12,
    ('T', 'W') => 2/6,
    ('T', 'E') => 7/11,
    ('T', 'P') => 2/21,
    ('T', 'K') => 6/19,
    ('T', 'L') => 6/22,
    ('T', 'S') => 8/34,
    ('T', 'F') => 0/8,
    ('T', 'A') => 3/12,
    ('T', 'C') => 3/11,
    ('T', 'M') => 1/7,
    ('T', 'D') => 1/11,
    ('T', 'G') => 2/19,
    # F row
    ('F', 'H') => 1/1,
    ('F', 'T') => 1/4,
    ('F', 'R') => 3/7,
    ('F', 'V') => 4/8,
    ('F', 'I') => 4/4,
    ('F', 'Q') => 1/5,
    ('F', 'Y') => 1/2,
    ('F', 'N') => 2/9,
    ('F', 'W') => 0/3,
    ('F', 'E') => 2/7,
    ('F', 'P') => 3/4,
    ('F', 'K') => 1/8,
    ('F', 'L') => 2/20,
    ('F', 'S') => 2/14,
    ('F', 'F') => 2/5,
    ('F', 'A') => 2/11,
    ('F', 'C') => 1/8,
    ('F', 'M') => 0/4,
    ('F', 'D') => 1/12,
    ('F', 'G') => 5/18,
    # H row
    ('H', 'H') => 2/3,
    ('H', 'T') => 0/4,
    ('H', 'R') => 4/8,
    ('H', 'V') => 2/3,
    ('H', 'I') => 1/6,
    ('H', 'Q') => 2/3,
    ('H', 'Y') => 1/4,
    ('H', 'N') => 3/9,
    ('H', 'W') => 0/5,
    ('H', 'E') => 3/9,
    ('H', 'P') => 1/3,
    ('H', 'K') => 0/11,
    ('H', 'L') => 0/6,
    ('H', 'S') => 2/4,
    ('H', 'F') => 1/4,
    ('H', 'A') => 0/4,
    ('H', 'C') => 1/4,
    ('H', 'M') => 1/3,
    ('H', 'D') => 1/4,
    ('H', 'G') => 0/9,
    # G row
    ('G', 'H') => 5/9,
    ('G', 'T') => 9/23,
    ('G', 'R') => 16/34,
    ('G', 'V') => 5/24,
    ('G', 'I') => 4/15,
    ('G', 'Q') => 2/11,
    ('G', 'Y') => 6/27,
    ('G', 'N') => 4/21,
    ('G', 'W') => 1/1,
    ('G', 'E') => 5/20,
    ('G', 'P') => 5/30,
    ('G', 'K') => 4/29,
    ('G', 'L') => 4/23,
    ('G', 'S') => 7/26,
    ('G', 'F') => 0/7,
    ('G', 'A') => 2/26,
    ('G', 'C') => 4/21,
    ('G', 'M') => 0/8,
    ('G', 'D') => 5/27,
    ('G', 'G') => 5/29,
    # S row
    ('S', 'H') => 2/8,
    ('S', 'T') => 10/25,
    ('S', 'R') => 8/21,
    ('S', 'V') => 5/20,
    ('S', 'I') => 4/14,
    ('S', 'Q') => 5/19,
    ('S', 'Y') => 3/8,
    ('S', 'N') => 8/19,
    ('S', 'W') => 3/7,
    ('S', 'E') => 5/32,
    ('S', 'P') => 3/27,
    ('S', 'K') => 8/26,
    ('S', 'L') => 4/31,
    ('S', 'S') => 9/39,
    ('S', 'F') => 3/15,
    ('S', 'A') => 6/36,
    ('S', 'C') => 2/14,
    ('S', 'M') => 2/8,
    ('S', 'D') => 3/26,
    ('S', 'G') => 3/32,
    # C row
    ('C', 'H') => 0/4,
    ('C', 'T') => 7/16,
    ('C', 'R') => 2/13,
    ('C', 'V') => 2/12,
    ('C', 'I') => 1/4,
    ('C', 'Q') => 1/5,
    ('C', 'Y') => 2/6,
    ('C', 'N') => 1/13,
    ('C', 'W') => 2/4,
    ('C', 'E') => 2/10,
    ('C', 'P') => 4/14,
    ('C', 'K') => 3/17,
    ('C', 'L') => 2/14,
    ('C', 'S') => 4/12,
    ('C', 'F') => 1/6,
    ('C', 'A') => 1/3,
    ('C', 'C') => 1/8,
    ('C', 'M') => 0/0,
    ('C', 'D') => 0/5,
    ('C', 'G') => 2/3,
    # W row
    ('W', 'H') => 4/10,
    ('W', 'T') => 2/2,
    ('W', 'R') => 1/6,
    ('W', 'V') => 0/2,
    ('W', 'I') => 1/2,
    ('W', 'Q') => 2/6,
    ('W', 'Y') => 0/3,
    ('W', 'N') => 1/5,
    ('W', 'W') => 0/4,
    ('W', 'E') => 1/4,
    ('W', 'P') => 2/5,
    ('W', 'K') => 0/3,
    ('W', 'L') => 1/5,
    ('W', 'S') => 0/4,
    ('W', 'F') => 0/0,
    ('W', 'A') => 0/7,
    ('W', 'C') => 0/1,
    ('W', 'M') => 0/0,
    ('W', 'D') => 1/3,
    ('W', 'G') => 0/5,
    # Y row
    ('Y', 'H') => 0/5,
    ('Y', 'T') => 7/11,
    ('Y', 'R') => 1/15,
    ('Y', 'V') => 3/15,
    ('Y', 'I') => 1/7,
    ('Y', 'Q') => 5/11,
    ('Y', 'Y') => 2/10,
    ('Y', 'N') => 4/15,
    ('Y', 'W') => 1/3,
    ('Y', 'E') => 2/14,
    ('Y', 'P') => 1/7,
    ('Y', 'K') => 2/20,
    ('Y', 'L') => 4/17,
    ('Y', 'S') => 3/14,
    ('Y', 'F') => 1/5,
    ('Y', 'A') => 1/9,
    ('Y', 'C') => 2/7,
    ('Y', 'M') => 0/5,
    ('Y', 'D') => 2/15,
    ('Y', 'G') => 2/11,
    # N row
    ('N', 'H') => 3/6,
    ('N', 'T') => 5/12,
    ('N', 'R') => 4/16,
    ('N', 'V') => 2/12,
    ('N', 'I') => 0/7,
    ('N', 'Q') => 2/17,
    ('N', 'Y') => 2/9,
    ('N', 'N') => 6/14,
    ('N', 'W') => 1/4,
    ('N', 'E') => 3/17,
    ('N', 'P') => 1/13,
    ('N', 'K') => 5/19,
    ('N', 'L') => 0/15,
    ('N', 'S') => 4/21,
    ('N', 'F') => 2/7,
    ('N', 'A') => 0/10,
    ('N', 'C') => 0/7,
    ('N', 'M') => 1/7,
    ('N', 'D') => 5/17,
    ('N', 'G') => 2/19,
    # D row
    ('D', 'H') => 1/3,
    ('D', 'T') => 8/22,
    ('D', 'R') => 3/17,
    ('D', 'V') => 2/20,
    ('D', 'I') => 4/15,
    ('D', 'Q') => 1/11,
    ('D', 'Y') => 0/1,
    ('D', 'N') => 2/12,
    ('D', 'W') => 1/4,
    ('D', 'E') => 0/21,
    ('D', 'P') => 0/14,
    ('D', 'K') => 2/24,
    ('D', 'L') => 4/29,
    ('D', 'S') => 7/24,
    ('D', 'F') => 2/9,
    ('D', 'A') => 3/19,
    ('D', 'C') => 2/7,
    ('D', 'M') => 0/12,
    ('D', 'D') => 2/22,
    ('D', 'G') => 2/30,
    # P row
    ('P', 'H') => missing,
    ('P', 'T') => missing,
    ('P', 'R') => missing,
    ('P', 'V') => missing,
    ('P', 'I') => missing,
    ('P', 'Q') => missing,
    ('P', 'Y') => missing,
    ('P', 'N') => missing,
    ('P', 'W') => missing,
    ('P', 'E') => missing,
    ('P', 'P') => missing,
    ('P', 'K') => missing,
    ('P', 'L') => missing,
    ('P', 'S') => missing,
    ('P', 'F') => missing,
    ('P', 'A') => missing,
    ('P', 'C') => missing,
    ('P', 'M') => missing,
    ('P', 'D') => missing,
    ('P', 'G') => missing
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
            #coupling_matrix[i, j] = get(contextfree_couplings, amino_acids[i], 0.90)  # Default to 0.90 if not in dict

            #coupling_matrix[i, j] = get(contextfree_couplings_Young1990, amino_acids[i], 0.90)^0.5  # Default to 0.90 if not in dict
            
            raw_value = get(contextual_couplings_Young1990, (amino_acids[i], amino_acids[j]), 0.5)  # Default to 0.50 if not in dict
            ismissing(raw_value) ? raw_value = 0.5 : raw_value = raw_value
            isfinite(raw_value) ? raw_value = raw_value : raw_value = 0.5
            
            # @printf("raw coupling_matrix[%d %s, %d %s] = %f\n", i, amino_acids[i], j, amino_acids[j], raw_value)
            
            coupling_matrix[i, j] = 1-raw_value*0.1
        end
    end
    
    return CouplingTable(coupling_matrix, amino_acids)
end


