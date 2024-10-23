@testset "SequenceToSequenceConvLSTM" begin
    rng = Xoshiro(42)
    x = rand(rng, Float32, 8, 8, 1, 6, 2)

    @testset "$mode" for mode in (:conditional_teacher, :generative, :conditional)
        model = SequenceToSequenceConvLSTM((3, 3), (3, 3), 1, (2, 4), 3, mode)

        ps, st = Lux.setup(Lux.replicate(rng), model)
        y, st2 = model(x, ps, st)
        @test size(y) == (8, 8, 3, 2)

        __f = (x, ps) -> sum(first(model(x, ps, st)))
        @test_gradients(__f, x, ps; atol=1.0f-3, rtol=1.0f-3)
    end

end
