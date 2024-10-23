@testset "ConvLSTMCell" begin
    rng = Xoshiro(42)
    @testset "peephole weights" begin
        x = rand(rng, Float32, 8, 8, 1, 1)
        layer = ConvLSTMCell((3, 3), (3, 3), 1 => 2; peephole=true)
        ps, st = Lux.setup(rng, layer)
        for k in (:Wc_i, :Wc_f, :Wc_o)
            @test haskey(ps, k)
        end
        layer = ConvLSTMCell((3, 3), (3, 3), 1 => 2; peephole=false)
        ps, st = Lux.setup(rng, layer)
        for k in (:Wc_i, :Wc_f, :Wc_o)
            @test !haskey(ps, k)
        end
    end

    @testset "output size" begin
        x = rand(rng, Float32, 8, 8, 1, 3)
        layer = ConvLSTMCell((3, 3), (3, 3), 1 => 2)
        ps, st = Lux.setup(rng, layer)
        (output, carry), st2 = layer(x, ps, st)
        @test size(output) == (8, 8, 2, 3)
    end

    @testset "carry" begin
        x = rand(rng, Float32, 8, 8, 1, 3)
        layer = ConvLSTMCell((3, 3), (3, 3), 1 => 2)
        ps, st = Lux.setup(rng, layer)
        (output, (hidden, memory)), st2 = layer(x, ps, st)
        @test output ≈ hidden
    end
end
