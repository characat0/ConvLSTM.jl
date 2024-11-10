using Lux

function Decoder(
    k_x::NTuple{N},
    k_h::NTuple{N},
    in_dims,
    hidden_dims::NTuple{M},
    use_bias::NTuple{M},
    peephole::NTuple{M},
    activation=σ,
    k_out=1,
) where {N, M}
    dims = vcat([in_dims], hidden_dims...)
    lstm = StackedCell(
        [
            ConvLSTMCell(k_x, k_h, dims[i] => dims[i+1], peephole=peephole[i], use_bias=use_bias[i])
            for i in 1:M
        ]...,
        concatenate=True(),
    )
    conv = Conv(ntuple(Returns(k_out), N), sum(hidden_dims) => in_dims, activation, use_bias=false, pad=SamePad())
    return Chain(lstm, Parallel(nothing, conv, NoOpLayer()))
end

