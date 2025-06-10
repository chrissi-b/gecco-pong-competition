module listnumber_fromimg
using StatsBase

import UTCGP: FunctionBundle, append_method!, FunctionWrapper, listfloat_caster
import UTCGP: CONSTRAINED, SMALL_ARRAY, NANO_ARRAY, BIG_ARRAY
import UTCGP: SImageND
import ImageCore: Normed
import LinearAlgebra: triu
using Distances

fallback(args...) = Float64[] # should be number
bundle_listnumber_from_img = FunctionBundle(listfloat_caster, fallback)
VECTORNUM = Vector{<:Number}

function pool4(img, mw, mh)
    a = @view img[1:mw, 1:mh]
    b = @view img[1:mw, (mh+1):end]
    c = @view img[(mw+1):end, 1:mh]
    d = @view img[(mw+1):end, (mh+1):end]
    a, b, cd
end
function get_middle_point(img)
    w, h, = size(img)
    mw = trunc(Int, w / 2)
    mh = trunc(Int, h / 2)
    mw, mh
end

########################
# Quadrant #############
########################

function max_pool4(from::SImageND{S,T,D,C}, args...) where {S,T<:Normed,D,C}
    mw, mh = get_middle_point(from)
    a, b, c, d = pool4(from, mw, mh)
    return maximum.([a, b, c, d])
end
function min_pool4(from::SImageND{S,T,D,C}, args...) where {S,T<:Normed,D,C}
    mw, mh = get_middle_point(from)
    a, b, c, d = pool4(from, mw, mh)
    return minimum.([a, b, c, d])
end
function mean_pool4(from::SImageND{S,T,D,C}, args...) where {S,T<:Normed,D,C}
    mw, mh = get_middle_point(from)
    a, b, c, d = pool4(from, mw, mh)
    return mean.([a, b, c, d])
end


# h argmax 
function hargmax_pool4(from::SImageND{S,T,D,C}, args...) where {S,T<:Normed,D,C}
    mw, mh = get_middle_point(from)
    a, b, c, d = pool4(from, mw, mh)
    v = argmax.([a, b, c, d])
    return convert.(Float64, map(x -> x[1], v))
end
function vargmax_pool4(from::SImageND{S,T,D,C}, args...) where {S,T<:Normed,D,C}
    mw, mh = get_middle_point(from)
    a, b, c, d = pool4(from, mw, mh)
    v = argmax.([a, b, c, d])
    return convert.(Float64, map(x -> x[2], v))
end

###################################
# ARGMAX per ROW /COL #############
###################################
function argmax_per_row(from::SImageND{S,T,D,C}, args...) where {S,T<:Normed,D,C}
    argmax.([from[ax, :] for ax in axes(from, 1)])
end

function argmax_per_col(from::SImageND{S,T,D,C}, args...) where {S,T<:Normed,D,C}
    argmax.([from[ax, :] for ax in axes(from, 2)])
end


###################################
# PEAK DISTANCE       #############
###################################
function _correct_npeaks(npeaks)
    npeaks = trunc(Int, npeaks)
    npeaks = clamp(npeaks, 2, 10)
    npeaks
end
function _get_n_peaks(img, npeaks)
    peaks_indices = sortperm(reduce(vcat, img), rev=true)[1:npeaks]
    vs = [img[idx] for idx in peaks_indices]
    peaks_cartesian_indices = findall(x -> x in vs, img)
    map(x -> collect(x.I), peaks_cartesian_indices)
end

function peaks_mindist(from::SImageND{S,T,D,C}, n::Number, args...) where {S,T<:Normed,D,C}
    npeaks = _correct_npeaks(n)
    cartesian_indices_for_peaks = _get_n_peaks(from, npeaks)
    d = pairwise(Euclidean(), cartesian_indices_for_peaks, cartesian_indices_for_peaks) # Square D matrix
    @show size(d)
    utri = triu(trues(size(d)))
    sort(reduce(vcat, d[utri]))[1:npeaks]
end
# function peaks_meandist(from::SImageND{S,T,D,C}, n::Number, args...) where {S,T<:Normed,D,C}
#     npeaks = _correct_npeaks(n)
#     cartesian_indices_for_peaks = _get_n_peaks(from, npeaks)
#     d = pairwise(Euclidean(), cartesian_indices_for_peaks, cartesian_indices_for_peaks) # Square D matrix
#     mean.()
# end
function peaks_maxdist(from::SImageND{S,T,D,C}, n::Number, args...) where {S,T<:Normed,D,C}
    npeaks = _correct_npeaks(n)
    cartesian_indices_for_peaks = _get_n_peaks(from, npeaks)
    d = pairwise(Euclidean(), cartesian_indices_for_peaks, cartesian_indices_for_peaks) # Square D matrix
    @show size(d)
    utri = triu(trues(size(d)))
    sort(reduce(vcat, d[utri]), rev=true)[1:npeaks]
end

# quadrants
append_method!(bundle_listnumber_from_img, max_pool4)
append_method!(bundle_listnumber_from_img, min_pool4)
append_method!(bundle_listnumber_from_img, mean_pool4)
append_method!(bundle_listnumber_from_img, hargmax_pool4)
append_method!(bundle_listnumber_from_img, vargmax_pool4)

# rows cols
append_method!(bundle_listnumber_from_img, argmax_per_row)
append_method!(bundle_listnumber_from_img, argmax_per_col)

# peak
# append_method!(bundle_listnumber_from_img, peaks_mindist)
# append_method!(bundle_listnumber_from_img, peaks_meandist)
# append_method!(bundle_listnumber_from_img, peaks_maxdist)
end


module tupleintint_2Dposition

using StatsBase

import UTCGP: FunctionBundle, append_method!, FunctionWrapper, listfloat_caster
import UTCGP: CONSTRAINED, SMALL_ARRAY, NANO_ARRAY, BIG_ARRAY
import UTCGP: SImageND
import ImageCore: Normed

fallback(args...) = (0, 0) # should be number
bundle_tupleintint_2Dposition = FunctionBundle(fallback)

function argmax_position(from::SImageND{S,T,D,C}, args...) where {S,T<:Normed,D,C}
    return argmax(from).I
end
function argmin_position(from::SImageND{S,T,D,C}, args...) where {S,T<:Normed,D,C}
    return argmin(from).I
end

function argmaxposition_from(from::SImageND{S,T,D,C}, w::Float64, h::Float64, args...) where {S,T<:Normed,D,C}
    fw, fh = clamp(w, 0.0, 1.0), clamp(h, 0.0, 1.0)
    wimg, himg = size(from)
    tw, th = trunc(Int, fw * wimg), trunc(Int, fh * himg)
    c = (@view from[tw:end, th:end]) |> argmax
    c.I .+ (tw, th) .- (1, 1)
end

function argmaxposition_to(from::SImageND{S,T,D,C}, w::Float64, h::Float64, args...) where {S,T<:Normed,D,C}
    fw, fh = clamp(w, 0.0, 1.0), clamp(h, 0.0, 1.0)
    wimg, himg = size(from)
    tw, th = trunc(Int, fw * wimg), trunc(Int, fh * himg)
    c = (@view from[tw:end, th:end]) |> argmax
    c.I
end

function center_of_mass(from::SImageND{S,T,D,C}, args...) where {S,T<:Normed,D,C}
    CI = CartesianIndices(from)
    center_x, center_y = 0.0, 0.0
    d1, d2 = 0.0, 0.0
    for ci in CI
        center_x += ci.I[1] * from[ci] # / mx
        center_y += ci.I[2] * from[ci]   #/ my
        d1 += from[ci]
        d2 += from[ci]
    end

    center_x /= d1
    center_y /= d2
    return trunc(Int, center_x), trunc(Int, center_y)
end


function direction(t1::Tuple{Int,Int}, t2::Tuple{Int,Int}, args...) where {S,T<:Normed,D,C}
    t2 .- t1
end


append_method!(bundle_tupleintint_2Dposition, argmax_position)
append_method!(bundle_tupleintint_2Dposition, argmin_position)
append_method!(bundle_tupleintint_2Dposition, argmaxposition_from)
append_method!(bundle_tupleintint_2Dposition, argmaxposition_to)

append_method!(bundle_tupleintint_2Dposition, center_of_mass)
append_method!(bundle_tupleintint_2Dposition, direction)

end


module float_comparetuples

using StatsBase

import UTCGP: FunctionBundle, append_method!, FunctionWrapper, listfloat_caster
import UTCGP: CONSTRAINED, SMALL_ARRAY, NANO_ARRAY, BIG_ARRAY
import UTCGP: SImageND, float_caster
import ImageCore: Normed

fallback(args...) = 0.0 # should be number
bundle_float_comparetuples = FunctionBundle(float_caster, fallback)

function true_less(t1::Tuple{Int,Int}, t2::Tuple{Int,Int}, args...) where {S,T<:Normed,D,C}
    if t1[1] < t2[1] && t1[2] < t2[2]
        return 1.0
    end
    return 0.0
end
function true_gt(t1::Tuple{Int,Int}, t2::Tuple{Int,Int}, args...) where {S,T<:Normed,D,C}
    if t1[1] > t2[1] && t1[2] > t2[2]
        return 1.0
    end
    return 0.0
end
function true_lesseq(t1::Tuple{Int,Int}, t2::Tuple{Int,Int}, args...) where {S,T<:Normed,D,C}
    if t1[1] <= t2[1] && t1[2] <= t2[2]
        return 1.0
    end
    return 0.0
end
function true_gteq(t1::Tuple{Int,Int}, t2::Tuple{Int,Int}, args...) where {S,T<:Normed,D,C}
    if t1[1] >= t2[1] && t1[2] >= t2[2]
        return 1.0
    end
    return 0.0
end

# OR
function less_on_one(t1::Tuple{Int,Int}, t2::Tuple{Int,Int}, args...) where {S,T<:Normed,D,C}
    if t1[1] < t2[1] || t1[2] < t2[2]
        return 1.0
    end
    return 0.0
end

function gt_on_one(t1::Tuple{Int,Int}, t2::Tuple{Int,Int}, args...) where {S,T<:Normed,D,C}
    if t1[1] > t2[1] || t1[2] > t2[2]
        return 1.0
    end
    return 0.0
end

# Distances
function dist_first(t1::Tuple{Int,Int}, t2::Tuple{Int,Int}, args...) where {S,T<:Normed,D,C}
    t1[1] - t2[1]
end
function dist_second(t1::Tuple{Int,Int}, t2::Tuple{Int,Int}, args...) where {S,T<:Normed,D,C}
    t1[2] - t2[2]
end

function dist(t1::Tuple{Int,Int}, t2::Tuple{Int,Int}, args...) where {S,T<:Normed,D,C}
    sqrt(sum((t1 .- t2) .^ 2))
end

append_method!(bundle_float_comparetuples, true_less)
append_method!(bundle_float_comparetuples, true_gt)
append_method!(bundle_float_comparetuples, true_lesseq)
append_method!(bundle_float_comparetuples, true_gteq)

append_method!(bundle_float_comparetuples, less_on_one)
append_method!(bundle_float_comparetuples, gt_on_one)

append_method!(bundle_float_comparetuples, dist_first)
append_method!(bundle_float_comparetuples, dist_second)
append_method!(bundle_float_comparetuples, dist)

end
