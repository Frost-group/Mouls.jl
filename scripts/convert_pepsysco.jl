#!/usr/bin/env julia

using CSV
using DataFrames

function convert_pepsysco_to_julia(input_csv::String, output_jl::String; total_count::Int=500)
    println("Reading CSV file: $input_csv")
    df = CSV.read(input_csv, DataFrame)
    
    rename!(df, "Peptide Sequence" => :sequence, "Mass" => :mass, "Rel Abundance" => :abundance)
    
    dropmissing!(df, [:sequence, :mass, :abundance])
    
    function safe_parse_abundance(s)
        try
            s_clean = replace(string(s), "%" => "", "#" => "")
            return parse(Float64, s_clean) / 100.0
        catch
            return missing
        end
    end
    
    df.abundance = safe_parse_abundance.(df.abundance)
    dropmissing!(df, :abundance)
    
    peptide_groups = groupby(df, :sequence)
    
    println("Found $(length(peptide_groups)) unique peptide sequences")
    
    observations = []
    
    for group in peptide_groups
        sequence = group.sequence[1]
        
        total_abundance = sum(group.abundance)
        total_abundance == 0.0 && continue
        
        mass_histogram = Dict{Float64, Int}()
        for row in eachrow(group)
            normalized_abundance = row.abundance / total_abundance
            count = round(Int, normalized_abundance * total_count)
            count > 0 && (mass_histogram[row.mass] = count)
        end
        
        isempty(mass_histogram) && continue
        
        actual_total = sum(values(mass_histogram))
        
        push!(observations, (sequence=sequence, histogram=mass_histogram, total=actual_total))
    end
    
    println("Writing $(length(observations)) observations")
    
    open(output_jl, "w") do file
        println(file, "# Generated from pepsysco.csv")
        println(file, "# Total peptides: $(length(observations))")
        println(file, "")
        println(file, "using Mouls")
        println(file, "")
        println(file, "[")
        
        for (idx, obs) in enumerate(observations)
            println(file, "    MassSpecObservation(")
            println(file, "        target_sequence=\"$(obs.sequence)\",")
            println(file, "        mass_histogram=Dict{Float64,Int}(")
            
            sorted_masses = sort(collect(keys(obs.histogram)))
            for (i, mass) in enumerate(sorted_masses)
                count = obs.histogram[mass]
                comma = i < length(sorted_masses) ? "," : ""
                println(file, "            $mass => $count$comma")
            end
            
            println(file, "        ),")
            println(file, "        total_count=$(obs.total)")
            
            comma = idx < length(observations) ? "," : ""
            println(file, "    )$comma")
        end
        
        println(file, "]")
    end
    
    println("Wrote output to: $output_jl")
end

if abspath(PROGRAM_FILE) == @__FILE__
    input_file = joinpath(@__DIR__, "..", "data", "pepsysco.csv")
    output_file = joinpath(@__DIR__, "..", "data", "pepsysco_observations.jl")
    
    if !isfile(input_file)
        error("Input file not found: $input_file")
    end
    
    convert_pepsysco_to_julia(input_file, output_file)
    println("\nConversion complete!")
    println("Load the observations with: observations = include(\"$output_file\")")
end
