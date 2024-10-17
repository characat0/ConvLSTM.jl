using Lux
using Lux: False, True, StaticBool

Lux.@concrete struct StackedCell <: AbstractLuxWrapperLayer{:layers}
    concatenate <: Lux.StaticBool
    layers <: NamedTuple
end

function StackedCell(cells...; concatenate::Lux.BoolType = False())
    StackedCell(static(concatenate), Chain(cells...).layers)
end


Lux.initialstates(rng::AbstractRNG, stacked::StackedCell) =
    Lux.initialstates(rng, stacked.layers)

Lux.initialparameters(rng::AbstractRNG, stacked::StackedCell) =
    Lux.initialparameters(rng, stacked.layers)

(s::StackedCell)(x, ps, st::NamedTuple) =
    applystacked(s.layers, s.concatenate, x, ps, st)

@generated function applystacked(
    layers::NamedTuple{fields},
    ::Lux.StaticBool{concat},
    x::AbstractArray{T,ND},
    ps,
    st::NamedTuple{fields},
) where {fields,concat,ND,T}
    N = length(fields)
    x_symbols = vcat([:x], [gensym() for _ = 1:N])
    c_symbols = [gensym() for _ = 1:N]
    st_symbols = [gensym() for _ = 1:N]
    calls = [
        :(
            (($(x_symbols[i+1]), $(c_symbols[i])), $(st_symbols[i])) =
                @inline Lux.apply(
                    layers.$(fields[i]),
                    $(x_symbols[i]),
                    ps.$(fields[i]),
                    st.$(fields[i]),
                )
        ) for i = 1:N
    ]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    if concat
        push!(
            calls,
            :(return (cat($(x_symbols[2:end]...); dims = ND - 1), ($(c_symbols...),)), st),
        )
    else
        push!(calls, :(return ($(x_symbols[N+1]), ($(c_symbols...),)), st))
    end
    return Expr(:block, calls...)
end


@generated function applystacked(
    layers::NamedTuple{fields},
    ::Lux.StaticBool{concat},
    inp::Tuple{AbstractArray{T,ND},Any},
    ps,
    st::NamedTuple{fields},
) where {fields,concat,ND,T}
    N = length(fields)
    x_symbols = vcat([:(inp[1])], [gensym() for _ = 1:N])
    c_symbols = [gensym() for _ = 1:N]
    st_symbols = [gensym() for _ = 1:N]

    calls = [
        :(
            (($(x_symbols[i+1]), $(c_symbols[i])), $(st_symbols[i])) =
                @inline Lux.apply(
                    layers.$(fields[i]),
                    ($(x_symbols[i]), inp[2][$(i)]),
                    ps.$(fields[i]),
                    st.$(fields[i]),
                )
        ) for i = 1:N
    ]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    if concat
        push!(
            calls,
            :(return (cat($(x_symbols[2:end]...); dims = ND - 1), ($(c_symbols...),)), st),
        )
    else
        push!(calls, :(return ($(x_symbols[N+1]), ($(c_symbols...),)), st))
    end
    return Expr(:block, calls...)
end


Lux.Functors.children(x::StackedCell) = Lux.Functors.children(x.layers)

function Base.show(io::IO, stacked::StackedCell)
    print(io, "StackedCell(\n")
    for (k, c) in pairs(Lux.Functors.children(stacked))
        Lux.PrettyPrinting.big_show(io, c, 4, k)
    end
    print(io, ")")
end
