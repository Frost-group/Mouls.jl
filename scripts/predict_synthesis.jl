#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Mouls
using ArgParse

s = ArgParseSettings(
    description="Monte Carlo prediction of peptide synthesis",
    version="0.1.0",
    add_version=true
)

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
coupling_table = create_coupling_table(args["coupling"])

for peptide in args["peptides"]
    result = predict_synthesis(
        peptide,
        coupling_table,
        num_simulations=args["simulations"],
        top=args["top"]
    )
    println("\nTotal unique sequences: $(length(result))")
end

