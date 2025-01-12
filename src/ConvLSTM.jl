module ConvLSTM

    include("./convlstmcell.jl")
    include("./recurrence.jl")
    include("./stacked.jl")
    include("./encoder.jl")
    include("./decoder.jl")

    

    struct ConditionalTeachingSequenceToSequenceConvLSTM{E, D} <: Lux.AbstractLuxContainerLayer{(:encoder, :decoder)}
        encoder::E
        decoder::D
        steps::Int
    end

    struct ConditionalSequenceToSequenceConvLSTM{E, D} <: Lux.AbstractLuxContainerLayer{(:encoder, :decoder)}
        encoder::E
        decoder::D
        steps::Int
    end

    struct GenerativeSequenceToSequenceConvLSTM{E, D} <: Lux.AbstractLuxContainerLayer{(:encoder, :decoder)}
        encoder::E
        decoder::D
        steps::Int
    end

    Lux.initialstates(rng::AbstractRNG, l::Union{ConditionalSequenceToSequenceConvLSTM, GenerativeSequenceToSequenceConvLSTM, ConditionalTeachingSequenceToSequenceConvLSTM}) =
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
        k_last=1,
        dropout_p=0.0,
    ) where {N, M}
        if mode ∉ (:conditional, :generative, :conditional_teaching)
            error("mode should be one of (:conditional, :generative, :conditional_teaching)")
        end
        encoder = Encoder(
            k_x, k_h, in_dims, hidden_dims, use_bias, peephole, dropout_p,
        )
        decoder = Decoder(
            k_x, k_h, in_dims, hidden_dims, use_bias, peephole, activation, k_last, dropout_p
        )
        if mode == :conditional
            return ConditionalSequenceToSequenceConvLSTM(
                encoder,
                decoder,
                steps
            )
        elseif mode == :conditional_teaching
            return ConditionalTeachingSequenceToSequenceConvLSTM(
                encoder,
                decoder,
                steps
            )
        elseif mode == :generative
            return GenerativeSequenceToSequenceConvLSTM(
                encoder,
                decoder,
                steps
            )
        end
    end

    SequenceToSequenceConvLSTM(
        k_x::NTuple{N},
        k_h::NTuple{N},
        in_dims, 
        hidden_dims::NTuple{M}, 
        steps,
        mode::Symbol;
        use_bias::Bool = false,
        peephole::Bool = true,
        activation=σ,
        dropout_p=0.0,
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
        1,
        dropout_p,
    )

    function (c::ConditionalSequenceToSequenceConvLSTM)(x::AbstractArray{T, N}, ps::NamedTuple, st::NamedTuple) where {T, N}
        rng = Lux.replicate(st.rng)
        (_, carry), st_encoder = c.encoder(x, ps.encoder, st.encoder)
        # Last frame
        Xi = selectdim(x, N-1, size(x, N-1))
        (output, carry), st_decoder = c.decoder((Xi, carry), ps.decoder, st.decoder)
        out = output
        for i in 1:c.steps-1
            Xi = output
            (output, carry), st_decoder = c.decoder((Xi, carry), ps.decoder, st_decoder)
            out = cat(out, output; dims=Val(N-2))
        end
        return out, merge(st, (; rng, encoder=st_encoder, decoder=st_decoder))
    end

    function (c::ConditionalTeachingSequenceToSequenceConvLSTM)(x::AbstractArray{T, N}, ps::NamedTuple, st::NamedTuple) where {T, N}
        rng = Lux.replicate(st.rng)
        if Lux.known(st.training)
            X_teach = selectdim(x, N-1, (size(x, N-1) - c.steps)+1:size(x, N-1))
            x = selectdim(x, N-1, 1:(size(x, N-1) - c.steps))
        end
        (_, carry), st_encoder = c.encoder(x, ps.encoder, st.encoder)
        # Last frame
        Xi = selectdim(x, N-1, size(x, N-1))
        (output, carry), st_decoder = c.decoder((Xi, carry), ps.decoder, st.decoder)
        out = output
        for i in 1:c.steps-1
            if Lux.known(st.training)
                Xi = selectdim(X_teach, N-1, i)
            else
                Xi = output
            end
            (output, carry), st_decoder = c.decoder((Xi, carry), ps.decoder, st_decoder)
            out = cat(out, output; dims=Val(N-2))
        end
        return out, merge(st, (; rng, encoder=st_encoder, decoder=st_decoder))
    end

    function (c::GenerativeSequenceToSequenceConvLSTM)(x::AbstractArray{T, N}, ps::NamedTuple, st::NamedTuple) where {T, N}
        rng = Lux.replicate(st.rng)
        (_, carry), st_encoder = c.encoder(x, ps.encoder, st.encoder)
        # Last frame
        Xi = glorot_uniform(rng, T, size(x)[1:N-2]..., size(x, N)) |> Lux.get_device(x)
        (output, carry), st_decoder = c.decoder((Xi, carry), ps.decoder, st.decoder)
        out = output
        for _ in 1:c.steps-1
            Xi = glorot_uniform(rng, T, size(x)[1:N-2]..., size(x, N)) |> Lux.get_device(x)
            (output, carry), st_decoder = c.decoder((Xi, carry), ps.decoder, st_decoder)
            out = cat(out, output; dims=Val(N-2))
        end
        return out, merge(st, (; rng, encoder=st_encoder, decoder=st_decoder))
    end

    export SequenceToSequenceConvLSTM
    export ConvLSTMCell
    export StackedCell

end
