using Lux

Lux.@concrete struct CarryRecurrence{R <: Lux.StaticBool} <: Lux.AbstractLuxWrapperLayer{:cell}
    cell <: Union{<:Lux.AbstractRecurrentCell, <:Lux.AbstractDebugRecurrentCell}
    ordering <: Lux.AbstractTimeSeriesDataBatchOrdering
    return_sequence::R
end

function CarryRecurrence(cell; ordering::Lux.AbstractTimeSeriesDataBatchOrdering=BatchLastIndex(),
    return_sequence::Bool=false)
    return CarryRecurrence(cell, ordering, static(return_sequence))
end

function (r::CarryRecurrence)(x::AbstractArray, ps, st::NamedTuple)
    return Lux.apply(r, Lux.safe_eachslice(x, r.ordering), ps, st)
end

function (r::CarryRecurrence{False})(x::Union{AbstractVector, NTuple}, ps, st::NamedTuple)
    (out, carry), st = Lux.apply(r.cell, first(x), ps, st)
    for xᵢ in x[(begin + 1):end]
        (out, carry), st = Lux.apply(r.cell, (xᵢ, carry), ps, st)
    end
    return (out, carry), st
end

function (r::CarryRecurrence{True})(x::Union{AbstractVector, NTuple}, ps, st::NamedTuple)
    function recur_op(::Nothing, input)
        (out, carry), state = apply(r.cell, input, ps, st)
        return [out], carry, state
    end
    function recur_op((outputs, carry, state), input)
        (out, carry), state = apply(r.cell, (input, carry), ps, state)
        return vcat(outputs, [out]), carry, state
    end
    results = Lux.private_foldl_init(recur_op, x)
    return first(results), last(results)
end





