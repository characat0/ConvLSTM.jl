using ConvLSTM
using Test
using Lux
using Random

@testset "ConvLSTM.jl" begin
    include("./convlstm_tests.jl")
    include("./stacked_tests.jl")
end
