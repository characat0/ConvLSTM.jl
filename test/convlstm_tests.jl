@testset "SequenceToSequenceConvLSTM" begin
    rng = Xoshiro(42)
    @testset "models" begin
        x = rand(rng, Float32, 8, 8, 1, 6, 2)

        for mode in (:generative, :conditional, :conditional_teacher), bias in (false, true), peephole in (false, true)
            model = SequenceToSequenceConvLSTM((3, 3), (3, 3), 1, (2, 4), 3, mode, bias, peephole)
            ps, st = Lux.setup(Lux.replicate(rng), model)
            model(x, ps, st)
        end
    end

end
