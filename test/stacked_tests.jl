@testset "StackedCell" begin
    rng = Xoshiro(42)
    @testset "initialization" begin
        M = 3
        dims = [1, 4, 4, 4]
        layer = StackedCell(
            [
                ConvLSTMCell((3, 3), (3, 3), dims[i] => dims[i+1])
                for i in 1:M
            ]...
        )
        @test length(layer.layers) == M
    end

    @testset "output shape" begin
        x = rand(rng, Float32, 8, 8, 1, 3)
        M = 3
        dims = [1, 2, 3, 4]
        layer = StackedCell(
            [
                ConvLSTMCell((3, 3), (3, 3), dims[i] => dims[i+1])
                for i in 1:M
            ]...
        )
        ps, st = Lux.setup(rng, layer)
        (output, carry), st2 = layer(x, ps, st)
        @test size(output) == (8, 8, 4, 3)
        layer = StackedCell(
            [
                ConvLSTMCell((3, 3), (3, 3), dims[i] => dims[i+1])
                for i in 1:M
            ]...;
            concatenate=true
        )
        (output, carry), st2 = layer(x, ps, st)
        @test size(output) == (8, 8, 2+3+4, 3)
    end
end
