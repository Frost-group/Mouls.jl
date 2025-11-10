# ============================================================================
# OBSERVATION TYPES
# ============================================================================

abstract type AbstractObservation end

@kwdef struct SequenceObservation <: AbstractObservation
    target_sequence::String
    observed_histogram::Dict{String,Int}
    total_count::Int
    
    function SequenceObservation(target_sequence, observed_histogram, total_count)
        @assert total_count > 0
        @assert sum(values(observed_histogram)) == total_count
        @assert all(count > 0 for count in values(observed_histogram))
        new(target_sequence, observed_histogram, total_count)
    end
end

Base.show(io::IO, obs::SequenceObservation) = begin
    println(io, "SequenceObservation")
    println(io, "  Target: $(obs.target_sequence)")
    println(io, "  Total count: $(obs.total_count)")
    println(io, "  Unique sequences: $(length(obs.observed_histogram))")
    
    sorted_seqs = sort(collect(obs.observed_histogram), by=x->x[2], rev=true)
    top_n = min(5, length(sorted_seqs))
    println(io, "  Top $top_n sequences:")
    for (seq, count) in sorted_seqs[1:top_n]
        prob = count / obs.total_count
        println(io, "    $seq: $count ($(round(prob*100, digits=2))%)")
    end
end

@kwdef struct MassSpecObservation <: AbstractObservation
    target_sequence::String
    mass_histogram::Dict{Float64,Int}
    total_count::Int
    
    function MassSpecObservation(target_sequence, mass_histogram, total_count)
        @assert total_count > 0
        @assert sum(values(mass_histogram)) == total_count
        @assert all(count > 0 for count in values(mass_histogram))
        new(target_sequence, mass_histogram, total_count)
    end
end

Base.show(io::IO, obs::MassSpecObservation) = begin
    println(io, "MassSpecObservation")
    println(io, "  Target: $(obs.target_sequence)")
    println(io, "  Total count: $(obs.total_count)")
    println(io, "  Unique masses: $(length(obs.mass_histogram))")
    
    sorted_masses = sort(collect(obs.mass_histogram), by=x->x[2], rev=true)
    top_n = min(5, length(sorted_masses))
    println(io, "  Top $top_n masses:")
    for (mass, count) in sorted_masses[1:top_n]
        prob = count / obs.total_count
        println(io, "    $(round(mass, digits=2)) Da: $count ($(round(prob*100, digits=2))%)")
    end
end

# ============================================================================
# VALIDATION
# ============================================================================

function validate_synthesis_observation(obs::SequenceObservation)
    valid_amino_acids = Set("ARNDCQEGHILKMFPSTWYV")
    
    !all(aa in valid_amino_acids for aa in obs.target_sequence) && (@warn "Invalid amino acids in target"; return false)
    
    for (seq, count) in obs.observed_histogram
        !all(aa in valid_amino_acids for aa in seq) && (@warn "Invalid amino acids in sequence: $seq"; return false)
        !is_subsequence_of(seq, obs.target_sequence) && @warn "Sequence '$seq' not a subsequence of target"
    end
    
    return true
end

function validate_synthesis_observation(obs::MassSpecObservation)
    valid_amino_acids = Set("ARNDCQEGHILKMFPSTWYV")
    
    !all(aa in valid_amino_acids for aa in obs.target_sequence) && (@warn "Invalid amino acids in target"; return false)
    all(mass > 0.0 for mass in keys(obs.mass_histogram)) || (@warn "Non-positive masses found"; return false)
    
    return true
end

function is_subsequence_of(subseq::String, seq::String)
    length(subseq) > length(seq) && return false
    isempty(subseq) && return true
    
    j = 1
    for i in 1:length(seq)
        j > length(subseq) && return true
        seq[i] == subseq[j] && (j += 1)
    end
    
    return j > length(subseq)
end

