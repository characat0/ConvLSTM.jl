
function Encoder(
    k_x::NTuple{N},
    k_h::NTuple{N},
    in_dims,
    hidden_dims::NTuple{M},
    use_bias::NTuple{M},
    peephole::NTuple{M},
) where {N, M}
    dims = vcat([in_dims], hidden_dims...)
    return CarryRecurrence(
        StackedCell(
            [
                ConvLSTMCell(k_x, k_h, dims[i] => dims[i+1], peephole=peephole[i], use_bias=use_bias[i])
                for i in 1:M
            ]...
        ),
    )
end
