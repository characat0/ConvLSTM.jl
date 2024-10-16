module ConvLSTM

    include("./convlstmcell.jl")
    include("./recurrence.jl")
    include("./stacked.jl")
    include("./encoder.jl")
    include("./decoder.jl")

    struct SequenceToSequenceConvLSTM{Teacher, E, D} <: Lux.AbstractLuxContainerLayer{(:encoder, :decoder)}
        teacher::Teacher
        encoder::E
        decoder::D
        steps
    end


    function SequenceToSequenceConvLSTM(
        k_x::NTuple{N},
        k_h::NTuple{N},
        in_dims, 
        hidden_dims::NTuple{M}, 
        steps,
        teacher::Lux.BoolType,
        use_bias::NTuple{M},
        peephole::NTuple{M},
        activation=σ,
    ) where {N, M}
        return SequenceToSequenceConvLSTM(
            static(teacher),
            Encoder(
                k_x, k_h, in_dims, hidden_dims, use_bias, peephole,
            ),
            Decoder(
                k_x, k_h, in_dims, hidden_dims, use_bias, peephole, activation,
            ),
            steps
        )
    end

    SequenceToSequenceConvLSTM(
        k_x::NTuple{N},
        k_h::NTuple{N},
        in_dims, 
        hidden_dims::NTuple{M}, 
        steps,
        teacher::Lux.BoolType,
        use_bias::Bool = false,
        peephole::Bool = true,
        activation=σ,
    ) where {N, M} = SequenceToSequenceConvLSTM(
        k_x,
        k_h,
        in_dims, 
        hidden_dims, 
        steps,
        teacher,
        ntuple(Returns(use_bias), M),
        ntuple(Returns(peephole), M),
        activation,
    )

    function (c::SequenceToSequenceConvLSTM)(x::AbstractArray{T, N}, ps::NamedTuple, st::NamedTuple) where {T, N}
        if Lux.known(c.teacher)
            X = selectdim(x, N-1, 1:(size(x, N-1) - c.steps))
        else
            X = x
        end
        (_, carry), st_encoder = c.encoder(X, ps.encoder, st.encoder)
        # Last frame
        Xi = selectdim(x, N-1, size(X, N-1) + 0)
    
        (output, carry), st_decoder = c.decoder((Xi, carry), ps.decoder, st.decoder)
        out = output
        for i in 1:c.steps-1
            # Autoregressive part
            if Lux.known(c.teacher)
                Xi = selectdim(x, N-1, size(X, N-1) + i)
            else
                Xi = output
            end
            (output, carry), st_decoder = c.decoder((Xi, carry), ps.decoder, st_decoder)
            out = cat(out, output; dims=Val(N-2))
        end
        return out, merge(st, (encoder=st_encoder, decoder=st_decoder))
    end

    export SequenceToSequenceConvLSTM

end
