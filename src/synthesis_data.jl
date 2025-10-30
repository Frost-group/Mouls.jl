using JSON

"""
    SynthesisObservation

Stores observed synthesis data for a single target peptide sequence.

Fields:
- `target_sequence::String`: The intended peptide sequence
- `observed_histogram::Dict{String,Int}`: Mapping of observed sequences to their counts
- `total_count::Int`: Total number of synthesis attempts observed
"""
@kwdef struct SynthesisObservation
    target_sequence::String
    observed_histogram::Dict{String,Int}
    total_count::Int
    
    function SynthesisObservation(target_sequence, observed_histogram, total_count)
        @assert total_count > 0 "total_count must be positive"
        @assert sum(values(observed_histogram)) == total_count "Histogram counts must sum to total_count"
        @assert all(count > 0 for count in values(observed_histogram)) "All counts must be positive"
        new(target_sequence, observed_histogram, total_count)
    end
end

Base.show(io::IO, obs::SynthesisObservation) = begin
    println(io, "SynthesisObservation")
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

"""
    validate_synthesis_observation(obs::SynthesisObservation) -> Bool

Validate that a SynthesisObservation is internally consistent and contains valid amino acid sequences.
"""
function validate_synthesis_observation(obs::SynthesisObservation)
    valid_amino_acids = Set("ARNDCQEGHILKMFPSTWYV")
    
    if !all(aa in valid_amino_acids for aa in obs.target_sequence)
        @warn "Target sequence contains invalid amino acids: $(obs.target_sequence)"
        return false
    end
    
    for (seq, count) in obs.observed_histogram
        if !all(aa in valid_amino_acids for aa in seq)
            @warn "Observed sequence contains invalid amino acids: $seq"
            return false
        end
        
        if !is_subsequence_of(seq, obs.target_sequence)
            @warn "Observed sequence '$seq' is not a valid subsequence of target '$(obs.target_sequence)'"
        end
    end
    
    return true
end

"""
    is_subsequence_of(subseq::String, seq::String) -> Bool

Check if subseq could result from incomplete synthesis of seq (considering deletions but not insertions).
"""
function is_subsequence_of(subseq::String, seq::String)
    if length(subseq) > length(seq)
        return false
    end
    
    if isempty(subseq)
        return true
    end
    
    j = 1
    for i in 1:length(seq)
        if j > length(subseq)
            return true
        end
        if seq[i] == subseq[j]
            j += 1
        end
    end
    
    return j > length(subseq)
end

"""
    load_synthesis_data_json(filepath::String) -> Vector{SynthesisObservation}

Load synthesis observations from a JSON file.

Expected JSON format:
```json
[
    {
        "target_sequence": "ACDEFG",
        "observed_histogram": {"ACDEFG": 100, "ACDEF": 50, "ACDE": 20},
        "total_count": 170
    },
    ...
]
```
"""
function load_synthesis_data_json(filepath::String)
    data = JSON.parsefile(filepath)
    observations = SynthesisObservation[]
    
    for entry in data
        obs = SynthesisObservation(
            target_sequence=entry["target_sequence"],
            observed_histogram=Dict{String,Int}(entry["observed_histogram"]),
            total_count=entry["total_count"]
        )
        
        if validate_synthesis_observation(obs)
            push!(observations, obs)
        else
            @warn "Skipping invalid observation for target: $(entry["target_sequence"])"
        end
    end
    
    return observations
end

"""
    save_synthesis_data_json(observations::Vector{SynthesisObservation}, filepath::String)

Save synthesis observations to a JSON file.
"""
function save_synthesis_data_json(observations::Vector{SynthesisObservation}, filepath::String)
    data = []
    
    for obs in observations
        push!(data, Dict(
            "target_sequence" => obs.target_sequence,
            "observed_histogram" => obs.observed_histogram,
            "total_count" => obs.total_count
        ))
    end
    
    open(filepath, "w") do io
        JSON.print(io, data, 2)
    end
end


