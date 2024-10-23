using ConvLSTM
using Test
using Lux
using Random

@testset "ConvLSTM.jl" begin
    include("./convlstmcell_tests.jl")
    include("./stacked_tests.jl")
    include("./convlstm_tests.jl")
end
