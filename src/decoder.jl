using Lux

function Decoder(
    k_x::NTuple{N},
    k_h::NTuple{N},
    in_dims,
    hidden_dims::NTuple{M},
    use_bias::NTuple{M},
    peephole::NTuple{M},
    activation=σ,
) where {N, M}
    dims = vcat([in_dims], hidden_dims...)
    Lux.@compact(
        lstm=StackedCell(
            [
                ConvLSTMCell(k_x, k_h, dims[i] => dims[i+1], peephole=peephole[i], use_bias=use_bias[i])
                for i in 1:M
            ]...,
            concatenate=True(),
        ),
        conv=Conv(ntuple(Returns(1), N), sum(hidden_dims) => in_dims, activation, use_bias=false),
    ) do X
        (X2, carry) = lstm(X)
        X3 = conv(X2)
        @return (X3, carry)
    end
end

