module ConvLSTM

    include("./convlstmcell.jl")
    include("./recurrence.jl")
    include("./stacked.jl")
    include("./encoder.jl")
    include("./decoder.jl")

    struct SequenceToSequenceConvLSTM{Mode, E, D} <: Lux.AbstractLuxContainerLayer{(:encoder, :decoder)}
        mode::Val{Mode}
        encoder::E
        decoder::D
        steps::Int
    end

    Lux.initialstates(rng::AbstractRNG, l::SequenceToSequenceConvLSTM) =
    (;
        rng = Lux.Utils.sample_replicate(rng),
        encoder=Lux.initialstates(Lux.Utils.sample_replicate(rng), l.encoder),
        decoder=Lux.initialstates(Lux.Utils.sample_replicate(rng), l.decoder),
        training=Val(true),
    )

    function SequenceToSequenceConvLSTM(
        k_x::NTuple{N},
        k_h::NTuple{N},
        in_dims, 
        hidden_dims::NTuple{M}, 
        steps,
        mode::Symbol,
        use_bias::NTuple{M},
        peephole::NTuple{M},
        activation=σ,
    ) where {N, M}
        if mode ∉ (:conditional_teacher, :conditional, :generative)
            error("mode should be one of (:conditional_teacher, :conditional, :generative)")
        end
        return SequenceToSequenceConvLSTM(
            Val(mode),
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
        mode::Symbol,
        use_bias::Bool = false,
        peephole::Bool = true,
        activation=σ,
    ) where {N, M} = SequenceToSequenceConvLSTM(
        k_x,
        k_h,
        in_dims, 
        hidden_dims, 
        steps,
        mode,
        ntuple(Returns(use_bias), M),
        ntuple(Returns(peephole), M),
        activation,
    )

    function (c::SequenceToSequenceConvLSTM{Mode})(x::AbstractArray{T, N}, ps::NamedTuple, st::NamedTuple) where {Mode, T, N}
        rng = Lux.replicate(st.rng)
        if (Mode == :conditional_teacher) && st.training
            X = selectdim(x, N-1, 1:(size(x, N-1) - c.steps))
        else
            X = x
        end
        (_, carry), st_encoder = c.encoder(X, ps.encoder, st.encoder)
        if (Mode == :conditional_teacher) || (Mode == :conditional)
            # Last frame
            Xi = selectdim(x, N-1, size(X, N-1) + 0)
        elseif (Mode == :generative)
            Xi = glorot_uniform(rng, T, size(X)[1:N-2]..., size(X, N))
        end
        (output, carry), st_decoder = c.decoder((Xi, carry), ps.decoder, st.decoder)
        out = output
        for i in 1:c.steps-1
            if (Mode == :conditional_teacher)
                if st.training
                    Xi = selectdim(x, N-1, size(X, N-1) + i)
                else
                    Xi = output
                end
            elseif Mode == :conditional
                Xi = output
            elseif Mode == :generative
                Xi = glorot_uniform(rng, T, size(X)[1:N-2]..., size(X, N))
            end
            (output, carry), st_decoder = c.decoder((Xi, carry), ps.decoder, st_decoder)
            out = cat(out, output; dims=Val(N-2))
        end
        return out, merge(st, (; rng, encoder=st_encoder, decoder=st_decoder))
    end

    export SequenceToSequenceConvLSTM
    export ConvLSTMCell
    export StackedCell

end
