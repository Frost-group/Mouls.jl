using Test
using Mouls
using Statistics

@testset "Bayesian Update Tests" begin
    
    @testset "BayesianCouplingPrior creation and conversion" begin
        coupling_table = create_coupling_table("contextual_couplings_Young1990")
        
        # Test with default prior_strength
        prior = initialize_prior_from_coupling_table(coupling_table, prior_strength=100.0)
        
        @test prior isa BayesianCouplingPrior
        @test size(prior.coupling_alpha) == (20, 20)
        @test size(prior.coupling_beta) == (20, 20)
        @test prior.truncation_alpha > 0
        @test prior.truncation_beta > 0
        @test length(prior.amino_acids) == 20
        
        # Verify scaling: α + β should equal prior_strength for all cells
        @test mean(prior.coupling_alpha .+ prior.coupling_beta) ≈ 100.0
        
        coupling_means = get_coupling_means(prior)
        @test size(coupling_means) == (20, 20)
        @test all(0.0 .<= coupling_means .<= 1.0)
        
        trunc_mean = get_truncation_mean(prior)
        @test 0.0 < trunc_mean < 1.0
        @test trunc_mean ≈ 0.001 atol=0.01
        
        reconstructed_table = from_bayesian_prior(prior)
        @test reconstructed_table isa CouplingTable
        @test size(reconstructed_table.matrix) == (20, 20)
        @test all(0.0 .<= reconstructed_table.matrix .<= 1.0)
        
        # Test with different prior_strength values
        prior_weak = initialize_prior_from_coupling_table(coupling_table, prior_strength=10.0)
        @test mean(prior_weak.coupling_alpha .+ prior_weak.coupling_beta) ≈ 10.0
        
        prior_strong = initialize_prior_from_coupling_table(coupling_table, prior_strength=1000.0)
        @test mean(prior_strong.coupling_alpha .+ prior_strong.coupling_beta) ≈ 1000.0
        
        # Verify proportions are preserved across different strengths
        for i in 1:20
            for j in 1:20
                p_default = prior.coupling_alpha[i,j] / (prior.coupling_alpha[i,j] + prior.coupling_beta[i,j])
                p_weak = prior_weak.coupling_alpha[i,j] / (prior_weak.coupling_alpha[i,j] + prior_weak.coupling_beta[i,j])
                @test p_default ≈ p_weak atol=0.001
            end
        end
    end
    
    @testset "SynthesisObservation creation and validation" begin
        histogram = Dict("ACDE" => 100, "ACD" => 50, "AC" => 20)
        total = 170
        
        obs = SynthesisObservation(
            target_sequence="ACDE",
            observed_histogram=histogram,
            total_count=total
        )
        
        @test obs.target_sequence == "ACDE"
        @test obs.total_count == 170
        @test length(obs.observed_histogram) == 3
        
        @test validate_synthesis_observation(obs)
        
        @test_throws AssertionError SynthesisObservation(
            target_sequence="ACDE",
            observed_histogram=Dict("ACDE" => 100),
            total_count=200
        )
        
        @test_throws AssertionError SynthesisObservation(
            target_sequence="ACDE",
            observed_histogram=Dict("ACDE" => 100),
            total_count=0
        )
    end
    
    @testset "Subsequence validation" begin
        @test is_subsequence_of("ACD", "ACDE")
        @test is_subsequence_of("ACE", "ACDE")
        @test is_subsequence_of("AD", "ACDE")
        @test is_subsequence_of("", "ACDE")
        @test is_subsequence_of("ACDE", "ACDE")
        
        @test !is_subsequence_of("ACDEX", "ACDE")
        @test !is_subsequence_of("XYZ", "ACDE")
    end
    
    @testset "Synthetic observation generation" begin
        coupling_table = create_coupling_table("contextual_couplings_Young1990")
        
        obs = create_synthetic_observation(
            "ACDE",
            coupling_table,
            num_simulations=100,
            truncation_rate=0.001
        )
        
        @test obs isa SynthesisObservation
        @test obs.target_sequence == "ACDE"
        @test obs.total_count == 100
        @test sum(values(obs.observed_histogram)) == 100
        @test validate_synthesis_observation(obs)
    end
    
    @testset "Histogram calculation with truncation" begin
        coupling_table = create_coupling_table("contextual_couplings_Young1990")
        
        histogram = calculate_histogram_with_truncation(
            "AAA",
            coupling_table,
            0.001,
            num_simulations=1000
        )
        
        @test histogram isa Dict{String, Int}
        @test sum(values(histogram)) == 1000
        @test all(count > 0 for count in values(histogram))
        
        histogram_high_trunc = calculate_histogram_with_truncation(
            "AAAAA",
            coupling_table,
            0.1,
            num_simulations=1000
        )
        
        @test sum(values(histogram_high_trunc)) == 1000
    end
    
    @testset "Likelihood calculation smoke test" begin
        coupling_table = create_coupling_table("contextual_couplings_Young1990")
        prior = initialize_prior_from_coupling_table(coupling_table, prior_strength=50.0)
        
        obs = create_synthetic_observation(
            "AAA",
            coupling_table,
            num_simulations=100,
            truncation_rate=0.001
        )
        
        log_lik = calculate_likelihood(prior, [obs], num_simulations=100)
        
        @test log_lik isa Float64
        @test isfinite(log_lik)
        @test log_lik < 0.0
    end
    
    @testset "Histogram comparison" begin
        pred = Dict("AAA" => 80, "AA" => 15, "A" => 5)
        obs = Dict("AAA" => 85, "AA" => 10, "A" => 5)
        
        log_lik = compare_histograms(pred, obs, 100)
        
        @test log_lik isa Float64
        @test isfinite(log_lik)
        @test log_lik < 0.0
        
        identical = Dict("AAA" => 100)
        log_lik_perfect = compare_histograms(identical, identical, 100)
        @test log_lik_perfect > log_lik
    end
    
    @testset "Prior density calculation" begin
        coupling_table = create_coupling_table("contextual_couplings_Young1990")
        prior = initialize_prior_from_coupling_table(coupling_table, prior_strength=50.0)
        
        coupling_means = get_coupling_means(prior)
        trunc_mean = get_truncation_mean(prior)
        
        log_prior = log_prior_density(prior, coupling_means, trunc_mean)
        
        @test log_prior isa Float64
        @test isfinite(log_prior)
        
        bad_couplings = fill(0.0, 20, 20)
        log_prior_bad = log_prior_density(prior, bad_couplings, trunc_mean)
        @test log_prior_bad == -Inf
        
        log_prior_bad2 = log_prior_density(prior, coupling_means, 0.0)
        @test log_prior_bad2 == -Inf
    end
    
    @testset "MAP estimation smoke test" begin
        coupling_table = create_coupling_table("contextual_couplings_Young1990")
        prior = initialize_prior_from_coupling_table(coupling_table, prior_strength=50.0)
        
        obs1 = create_synthetic_observation(
            "AAA",
            coupling_table,
            num_simulations=200,
            truncation_rate=0.001
        )
        
        obs2 = create_synthetic_observation(
            "AAA",
            coupling_table,
            num_simulations=200,
            truncation_rate=0.001
        )
        
        posterior = update_posterior(
            prior,
            [obs1, obs2],
            method=:map,
            num_likelihood_sims=100,
            max_iterations=10
        )
        
        @test posterior isa BayesianCouplingPrior
        @test size(posterior.coupling_alpha) == (20, 20)
        @test all(posterior.coupling_alpha .> 0)
        @test all(posterior.coupling_beta .> 0)
        @test posterior.truncation_alpha > 0
        @test posterior.truncation_beta > 0
        
        posterior_coupling_means = get_coupling_means(posterior)
        @test all(0.0 .< posterior_coupling_means .< 1.0)
        
        updated_table = from_bayesian_prior(posterior)
        @test updated_table isa CouplingTable
    end
    
    @testset "JSON save and load" begin
        mktempdir() do tmpdir
            filepath = joinpath(tmpdir, "test_data.json")
            
            obs1 = SynthesisObservation(
                target_sequence="ACDE",
                observed_histogram=Dict("ACDE" => 100, "ACD" => 50),
                total_count=150
            )
            
            obs2 = SynthesisObservation(
                target_sequence="FGH",
                observed_histogram=Dict("FGH" => 80, "FG" => 20),
                total_count=100
            )
            
            observations = [obs1, obs2]
            save_synthesis_data_json(observations, filepath)
            
            @test isfile(filepath)
            
            loaded_obs = load_synthesis_data_json(filepath)
            
            @test length(loaded_obs) == 2
            @test loaded_obs[1].target_sequence == "ACDE"
            @test loaded_obs[1].total_count == 150
            @test loaded_obs[2].target_sequence == "FGH"
            @test loaded_obs[2].total_count == 100
        end
    end
    
end

println("\nAll tests passed! ✓")

