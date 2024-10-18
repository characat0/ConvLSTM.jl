@testset "ConvLSTMCell" begin
    rng = Xoshiro(42)
    @testset "cpu" begin
        x = rand(rng, Float32, 8, 8, 1, 1)
        layer = ConvLSTMCell((3, 3), (3, 3), 1 => 2; peephole=true)
        ps, st = Lux.setup(rng, layer)
        for k in (:Wc_i, :Wc_f, :Wc_o)
            @test haskey(ps, k)
        end

    end
end
