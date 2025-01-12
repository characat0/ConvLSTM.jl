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
    dropout_p=0.0,
) where {N, M}
    dims = vcat([in_dims], hidden_dims...)
    cells = [
        ConvLSTMCell(k_x, k_h, dims[i] => dims[i+1], peephole=peephole[i], use_bias=use_bias[i])
        for i in 1:M
    ]
    if (dropout_p > 0) && (length(cells) > 1)
        cells = vcat([
            Chain(c, Parallel(nothing, Dropout(dropout_p), NoOpLayer()))
            for c in cells[begin:end-1]
        ], cells[end:end])
    end
    lstm = StackedCell(
        cells...,
        concatenate=True(),
    )
    conv = Conv(ntuple(Returns(k_out), N), sum(hidden_dims) => in_dims, activation, use_bias=false, pad=SamePad())
    return Chain(lstm, Parallel(nothing, conv, NoOpLayer()))
end

