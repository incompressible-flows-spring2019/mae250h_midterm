# Grid mappings

export xmap, ymap

"""
    xmap(i,u::GridData,p::Parameters) -> Real
    ymap(j,u::GridData,p::Parameters) -> Real

Map a given index `i` or `j` in the grid to its physical coordinate, based on
the supplied type of data in `u`. Grid mapping parameters are provided in
`p`.
"""
xmap(i::Int,u::NodeData,p::Parameters) = xnode(i,p)
ymap(j::Int,u::NodeData,p::Parameters) = ynode(j,p)
xmap(i::Int,u::CellData,p::Parameters) = xcell(i,p)
ymap(j::Int,u::CellData,p::Parameters) = ycell(j,p)
xmap(i::Int,u::XEdgeData,p::Parameters) = xedgex(i,p)
ymap(j::Int,u::XEdgeData,p::Parameters) = yedgex(j,p)
xmap(i::Int,u::YEdgeData,p::Parameters) = xedgey(i,p)
ymap(j::Int,u::YEdgeData,p::Parameters) = yedgey(j,p)

# these allow the functions to be evaluated on sets of indices
"""
    xmap(ir::Range,u::GridData,p::Parameters) -> Array
    ymap(jr::Range,u::GridData,p::Parameters) -> Array

Map a range of indices to their physical coordinates.
"""
xmap(ir::AbstractArray{<:Int},u...) = map(i -> xmap(i,u...),ir)
ymap(jr::AbstractArray{<:Int},u...) = map(j -> ymap(j,u...),jr)

# These are ultimately called by xmap and ymap
xnode(i_n,p::Parameters) = p.x0 + (i_n - 1)*p.Δx
ynode(j_n,p::Parameters) = p.y0 + (j_n - 1)*p.Δx
xcell(i_c,p::Parameters) = p.x0 + (i_c - 1.5)*p.Δx
ycell(j_c,p::Parameters) = p.y0 + (j_c - 1.5)*p.Δx
xedgex(i_e,p::Parameters) = xnode(i_e,p) # x component of x edge
yedgex(j_e,p::Parameters) = ycell(j_e,p) # y component of x edge
xedgey(i_e,p::Parameters) = xcell(i_e,p) # x component of y edge
yedgey(j_e,p::Parameters) = ynode(j_e,p) # y component of y edge
