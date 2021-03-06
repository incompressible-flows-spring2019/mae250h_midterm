#=
Here we will set up operations to be carried out on our grid data. Note that
all of these operations are to be carried out in index space.
divergence
grad
curl
rot
Laplacian
translation (e.g. translating from cell data to edge data)
dot products for edge data and node data
=#

# The LinearAlgebra package contains some useful functions on arrays
# We import some of its operations to extend them to our data here
using LinearAlgebra
import LinearAlgebra:dot, norm

export diff!
export divergence, gradient, rot, curl, laplacian, interpolate!
export restrict!,prolong!

#============  INNER PRODUCTS AND NORMS ===============#

# To compute inner products, we will extend the Julia function `dot`. Note that
# we exclude the ghost cells from the dot product.
"""
    dot(p1::CellData,p2::CellData) -> Real

Computes the inner product between two sets of cell-centered data on the same grid.
"""
function dot(p1::CellData{NX,NY},p2::CellData{NX,NY}) where {NX,NY}
  return dot(p1[2:NX+1,2:NY+1],p2[2:NX+1,2:NY+1])/(NX*NY)
end

"""
    dot(p1::NodeData,p2::NodeData) -> Real

Computes the inner product between two sets of node data on the same grid.
"""
function dot(p1::NodeData{NX,NY},p2::NodeData{NX,NY}) where {NX,NY}

  # interior
  tmp = dot(p1[2:NX,2:NY],p2[2:NX,2:NY])

  # boundaries
  tmp += 0.5*dot(p1[1,2:NY],   p2[1,2:NY])
  tmp += 0.5*dot(p1[NX+1,2:NY],p2[NX+1,2:NY])
  tmp += 0.5*dot(p1[2:NX,1],   p2[2:NX,1])
  tmp += 0.5*dot(p1[2:NX,NY+1],p2[2:NX,NY+1])

  # corners
  tmp += 0.25*(p1[1,1]*p2[1,1]       + p1[NX+1,1]*p2[NX+1,1] +
               p1[1,NY+1]*p2[1,NY+1] + p1[NX+1,NY+1]*p2[NX+1,NY+1])

  return tmp/(NX*NY)
end

"""
    dot(p1::EdgeData,p2::EdgeData) -> Real

Computes the inner product between two sets of edge data on the same grid.
"""
function dot(p1::EdgeData{NX,NY},p2::EdgeData{NX,NY}) where {NX,NY}

  # interior
  tmp = dot(p1.qx[2:NX,2:NY+1],p2.qx[2:NX,2:NY+1]) +
        dot(p1.qy[2:NX+1,2:NY],p2.qy[2:NX+1,2:NY])

  # boundaries
  tmp += 0.5*dot(p1.qx[1,2:NY+1],   p2.qx[1,2:NY+1])
  tmp += 0.5*dot(p1.qx[NX+1,2:NY+1],p2.qx[NX+1,2:NY+1])
  tmp += 0.5*dot(p1.qy[2:NX+1,1],   p2.qy[2:NX+1,1])
  tmp += 0.5*dot(p1.qy[2:NX+1,NY+1],p2.qy[2:NX+1,NY+1])

  return tmp/(NX*NY)
end

"""
    norm(p::GridData) -> Real

Computes the L2 norm of data on a grid.
"""
norm(p::GridData) = sqrt(dot(p,p))

# This function computes an integral by just taking the inner product with
# another set of cell data uniformly equal to 1
"""
    integrate(p::CellData) -> Real

Computes a numerical quadrature of the cell-centered data.
"""
function integrate(p::CellData{NX,NY}) where {NX,NY}
  p2 = CellData(p)
  fill!(p2.data,1) # fill it with ones
  return dot(p,p2)
end

"""
    integrate(p::NodeData) -> Real

Computes a numerical quadrature of the node data.
"""
function integrate(p::NodeData{NX,NY}) where {NX,NY}
  p2 = NodeData(p)
  fill!(p2.data,1) # fill it with ones
  return dot(p,p2)
end

#=============== 1-D DIFFERENCING OPERATIONS ==================#

# edges to cells, cells to edges

"""
    diff!(p::CellData,u::XEdgeData)

Compute the x difference of x-edge data `u` at cell centers and place the result
in `p`. Only interior cells are updated.
"""
function diff!(p::CellData{NX,NY},u::XEdgeData{NX,NY}) where {NX,NY}
    p[2:NX+1,2:NY+1] .= diff(u[:,2:NY+1],dims=1)
    return p
end

"""
    diff!(p::CellData,v::YEdgeData)

Compute the x difference of y-edge data `v` at cell centers and place the result
in `p`. Only interior cells are updated.
"""
function diff!(p::CellData{NX,NY},v::YEdgeData{NX,NY}) where {NX,NY}
    p[2:NX+1,2:NY+1] .= diff(v[2:NX+1,:],dims=2)
    return p
end

"""
    diff!(u::XEdgeData,p::CellData)

Compute the x difference of cell data `p` at x edges and place the result
in `u`.
"""
function diff!(u::XEdgeData{NX,NY},p::CellData{NX,NY}) where {NX,NY}
    u .= diff(p,dims=1)
    return u
end

"""
    diff!(v::YEdgeData,p::CellData)

Compute the x difference of cell data `p` at y edges and place the result
in `v`.
"""
function diff!(v::YEdgeData{NX,NY},p::CellData{NX,NY}) where {NX,NY}
    v .= diff(p,dims=2)
    return v
end

# edges to nodes, nodes to edges

"""
    diff!(s::NodeData,u::XEdgeData)

Compute the y difference of x-edge data `u` at nodes and place the result
in `s`.
"""
function diff!(s::NodeData{NX,NY},u::XEdgeData{NX,NY}) where {NX,NY}
    s .= diff(u,dims=2)
    return s
end

"""
    diff!(s::NodeData,v::YEdgeData)

Compute the x difference of y-edge data `v` at nodes and place the result
in `s`.
"""
function diff!(s::NodeData{NX,NY},v::YEdgeData{NX,NY}) where {NX,NY}
    s .= diff(v,dims=1)
    return s
end

"""
    diff!(u::XEdgeData,s::NodeData)

Compute the y difference of node data `s` at x edges and place the result
in `u`. Note that only interior edges are updated.
"""
function diff!(u::XEdgeData{NX,NY},s::NodeData{NX,NY}) where {NX,NY}
    u[:,2:NY+1] .= diff(s,dims=2)
    return u
end

"""
    diff!(v::YEdgeData,s::NodeData)

Compute the x difference of node data `s` at y edges and place the result
in `v`. Note that only interior edges are updated.
"""
function diff!(v::YEdgeData{NX,NY},s::NodeData{NX,NY}) where {NX,NY}
    v[2:NX+1,:] .= diff(s,dims=1)
    return v
end

#=============== 2-D DIFFERENCING OPERATIONS ==================#


"""
    divergence(q::EdgeData) -> CellData

Compute the discrete divergence of edge data `q`, returning cell-centered
data on the same grid.
"""
function divergence(q::EdgeData{NX,NY}) where {NX,NY}
   p = CellData(q)
   # Loop over interior cells
   for j in 2:NY+1, i in 2:NX+1
     p[i,j] = q.qx[i,j] - q.qx[i-1,j] + q.qy[i,j] - q.qy[i,j-1]
   end
   return p

end

"""
    rot(q::EdgeData) -> NodeData

Compute the discrete rot of edge data `q`, returning node
data on the same grid. Can also be called as `curl(q)`.
"""
function rot(q::EdgeData{NX,NY}) where {NX,NY}
    w = NodeData(q)
    # Loop over all nodes
    for j in 1:NY+1, i in 1:NX+1
      w[i,j] = q.qx[i,j] - q.qx[i,j+1] + q.qy[i+1,j] - q.qy[i,j]
    end
    return w
end

# We can also call this curl, if we feel like it...
curl(q::EdgeData) = rot(q)

"""
    gradient(p::CellData) -> EdgeData

Compute the discrete gradient of cell-centered data `p`, returning edge
data on the same grid.
"""
function gradient(p::CellData{NX,NY}) where {NX,NY}
    q = EdgeData(p)
    diff!(q.qx,p)
    diff!(q.qy,p)
    return q
end

"""
    curl(s::NodeData) -> EdgeData

Compute the discrete curl of node data `s`, returning edge
data on the same grid.
"""
function curl(s::NodeData{NX,NY}) where {NX,NY}
    q = EdgeData(s)
    diff!(q.qx,s)
    diff!(q.qy,s)
    q.qy .= -q.qy
    return q
end

"""
    laplacian(p::CellData) -> CellData

Compute the discrete Laplacian of the cell-centered data `p`, using
ghost cells where needed.
"""
function laplacian(p::CellData{NX,NY}) where {NX,NY}
  lap = CellData(p)
  for j in 2:NY+1, i in 2:NX+1
    lap[i,j] = -4*p[i,j] + p[i-1,j] + p[i+1,j] + p[i,j-1] + p[i,j+1]
  end
  return lap
end

"""
    laplacian(p::CellData,dir::Integer) -> CellData

Compute the discrete second derivative, in the direction specified by `dir` (1 or 2)
of the cell-centered data `p`, using ghost cells where needed.
"""
function laplacian(p::CellData{NX,NY},dir::Integer) where {NX,NY}
  lap = CellData(p)
  if dir == 1
    for j in 2:NY+1, i in 2:NX+1
      lap[i,j] = -2*p[i,j] + p[i-1,j] + p[i+1,j]
    end
  elseif dir == 2
    for j in 2:NY+1, i in 2:NX+1
      lap[i,j] = -2*p[i,j] + p[i,j-1] + p[i,j+1]
    end
  end
  return lap
end

"""
    laplacian(s::NodeData) -> NodeData

Compute the discrete Laplacian of the node data `s` at its interior nodes.
"""
function laplacian(s::NodeData{NX,NY}) where {NX,NY}
  lap = NodeData(s)
  for j in 2:NY, i in 2:NX
    lap[i,j] = -4*s[i,j] + s[i-1,j] + s[i+1,j] + s[i,j-1] + s[i,j+1]
  end
  return lap
end

"""
    laplacian(s::NodeData,dir::Integer) -> NodeData

Compute the discrete second derivative, in the direction specified by `dir` (1 or 2),
of the node data `s` at its interior nodes.
"""
function laplacian(s::NodeData{NX,NY},dir::Integer) where {NX,NY}
  lap = NodeData(s)
  if dir == 1
    for j in 2:NY, i in 2:NX
      lap[i,j] = -2*s[i,j] + s[i-1,j] + s[i+1,j]
    end
  elseif dir == 2
    for j in 2:NY, i in 2:NX
      lap[i,j] = -2*s[i,j] + s[i,j-1] + s[i,j+1]
    end
  end
  return lap
end

"""
    laplacian(q::EdgeData) -> EdgeData

Compute the discrete Laplacian of both components of the edge data
`q` at its interior edges.
"""
function laplacian(q::EdgeData{NX,NY}) where {NX,NY}
  lap = EdgeData(q)
  for j in 2:NY+1, i in 2:NX
    lap.qx[i,j] = -4*q.qx[i,j] +
                    q.qx[i-1,j] + q.qx[i+1,j] + q.qx[i,j-1] + q.qx[i,j+1]
  end
  for j in 2:NY, i in 2:NX+1
    lap.qy[i,j] = -4*q.qy[i,j] +
                    q.qy[i-1,j] + q.qy[i+1,j] + q.qy[i,j-1] + q.qy[i,j+1]
  end
  return lap
end

"""
    laplacian(q::EdgeData,dir::Integer) -> EdgeData

Compute the discrete second derivative, in the direction specified by `dir` (1
  or 2) of both components of the edge data `q` at its interior edges
"""
function laplacian(q::EdgeData{NX,NY},dir::Int) where {NX,NY}
  lap = EdgeData(q)
  if dir == 1
    for j in 2:NY+1, i in 2:NX
      lap.qx[i,j] = -2*q.qx[i,j] + q.qx[i-1,j] + q.qx[i+1,j]
    end
    for j in 2:NY, i in 2:NX+1
      lap.qy[i,j] = -2*q.qy[i,j] + q.qy[i-1,j] + q.qy[i+1,j]
    end
  elseif dir == 2
    for j in 2:NY+1, i in 2:NX
      lap.qx[i,j] = -2*q.qx[i,j] + q.qx[i,j-1] + q.qx[i,j+1]
    end
    for j in 2:NY, i in 2:NX+1
      lap.qy[i,j] = -2*q.qy[i,j] + q.qy[i,j-1] + q.qy[i,j+1]
    end
  end
  return lap
end


#=============== INTERPOLATING OPERATIONS ==================#
#=
Note that we call this interpolate!, because the first argument
is changed by the function.
=#

# Interpolate from cells to edges
function interpolate!(qx::XEdgeData{NX,NY},p::CellData{NX,NY}) where {NX,NY}
    # This interpolation from cells to x-edges proceeds horizontally.
    # Loop over all interior, boundary and ghost edges.
    for j in 1:NY+2, i in 1:NX+1
      qx[i,j] = 0.5*(p[i+1,j] + p[i,j])
    end
    return qx
end
function interpolate!(qy::YEdgeData{NX,NY},p::CellData{NX,NY}) where {NX,NY}
    # This interpolation from cells to y-edges proceeds vertically.
    # Loop over all interior, boundary and ghost edges.
    for j in 1:NY+1, i in 1:NX+2
      qy[i,j] = 0.5*(p[i,j+1] + p[i,j])
    end
    return qy
end

"""
    interpolate!(q::EdgeData,p::CellData)

Translate (by simply averaging) cell data `p` into edge data `q`
on the same grid.
"""
function interpolate!(q::EdgeData{NX,NY},p::CellData{NX,NY}) where {NX,NY}
    interpolate!(q.qx,p)
    interpolate!(q.qy,p)
    return q
end

# Interpolate from edges to cell centers
function interpolate!(p::CellData{NX,NY},qx::XEdgeData{NX,NY}) where {NX,NY}
    # This interpolation from x-edges to cells proceeds horizontally.
    # Loop over all interior cells.
    for j in 2:NY+1, i in 2:NX+1
      p[i,j] = 0.5*(qx[i,j] + qx[i-1,j])
    end
    return p
end
function interpolate!(p::CellData{NX,NY},qy::YEdgeData{NX,NY}) where {NX,NY}
    # This interpolation from y-edges to cells proceeds vertically.
    # Loop over all interior cells.
    for j in 2:NY+1, i in 2:NX+1
      p[i,j] = 0.5*(qy[i,j] + qy[i,j-1])
    end
    return p
end


# Interpolate from nodes to edges
function interpolate!(qx::XEdgeData{NX,NY},s::NodeData{NX,NY}) where {NX,NY}
    # This interpolation from nodes to x-edges proceeds vertically.
    # Loop over all interior and boundary edges.
    for j in 2:NY+1, i in 1:NX+1
      qx[i,j] = 0.5*(s[i,j] + s[i,j-1])
    end
    return qx
end
function interpolate!(qy::YEdgeData{NX,NY},s::NodeData{NX,NY}) where {NX,NY}
    # This interpolation from nodes to y-edges proceeds horizontally.
    # Loop over all interior and boundary edges.
    for j in 1:NY+1, i in 2:NX+1
      qy[i,j] = 0.5*(s[i,j] + s[i-1,j])
    end
    return qy
end

"""
    interpolate!(q::EdgeData,s::NodaData)

Translate (by simply averaging) node data `s` into edge data `q`
on the same grid.
"""
function interpolate!(q::EdgeData{NX,NY},s::NodeData{NX,NY}) where {NX,NY}
    interpolate!(q.qx,s)
    interpolate!(q.qy,s)
    return q
end

# Interpolate from edges to nodes

function interpolate!(s::NodeData{NX,NY},qx::XEdgeData{NX,NY}) where {NX,NY}
    # This interpolation from nodes to x-edges proceeds vertically.
    # Loop over all nodes.
    for j in 1:NY+1, i in 1:NX+1
      s[i,j] = 0.5*(qx[i,j+1] + qx[i,j])
    end
    return s
end

function interpolate!(s::NodeData{NX,NY},qy::YEdgeData{NX,NY}) where {NX,NY}
    # This interpolation from nodes to y-edges proceeds horizontally.
    # Loop over all nodes.
    for j in 1:NY+1, i in 1:NX+1
      s[i,j] = 0.5*(qy[i+1,j] + qy[i,j])
    end
    return s
end


#=============== RESTRICTION AND PROLONGATION OPERATIONS ==================#
"""
    restrict!(target::CellData,src::CellData)

Restrict (by 16-point stencil) the cell data `src` onto a grid of half the number
of cells in each direction.
"""
function restrict!(target::CellData{NXhalf,NYhalf},src::CellData{NX,NY}) where {NXhalf,NYhalf,NX,NY}

  @assert NX == 2NXhalf && NY == 2NYhalf "Incompatible grid sizes for restriction"

  # Loop over all interior cells in the half-size grid
  for jhalf = 2:NYhalf+1, ihalf = 2:NXhalf+1
      target[ihalf,jhalf] = 9/64*(src[2ihalf-1,2jhalf-1] + src[2ihalf-2,2jhalf-1]
                                + src[2ihalf-1,2jhalf-2] + src[2ihalf-2,2jhalf-2])+
                            3/64*(src[2ihalf  ,2jhalf-1] + src[2ihalf  ,2jhalf-2]
                                + src[2ihalf-3,2jhalf-1] + src[2ihalf-3,2jhalf-2]
                                + src[2ihalf-1,2jhalf  ] + src[2ihalf-2,2jhalf  ]
                                + src[2ihalf-1,2jhalf-3] + src[2ihalf-2,2jhalf-3])+
                            1/64*(src[2ihalf  ,2jhalf  ] + src[2ihalf-3,2jhalf  ]
                                + src[2ihalf  ,2jhalf-3] + src[2ihalf-3,2jhalf-3])
  end
  return target
end

"""
    restrict!(target::NodeData,src::NodeData)

Restrict (by 9-point stencil) the node data `src` onto a grid of half the number
of cells in each direction.
"""
function restrict!(target::NodeData{NXhalf,NYhalf},src::NodeData{NX,NY}) where {NXhalf,NYhalf,NX,NY}

  @assert NX == 2NXhalf && NY == 2NYhalf "Incompatible grid sizes for restriction"

  # Loop over all interior nodes in the half-size grid
  for jhalf = 2:NYhalf, ihalf = 2:NXhalf
      target[ihalf,jhalf] = 0.25*src[2ihalf-1,2jhalf-1] +
                            0.125*(src[2ihalf-2,2jhalf-1]+src[2ihalf  ,2jhalf-1]
                                 + src[2ihalf-1,2jhalf-2]+src[2ihalf-1,2jhalf  ])+
                           0.0625*(src[2ihalf-2,2jhalf-2]+src[2ihalf  ,2jhalf-2]
                                 + src[2ihalf-2,2jhalf  ]+src[2ihalf  ,2jhalf  ])
  end
  return target
end

"""
    prolong!(target::CellData,src::CellData)

Prolong (by 4-point stencil) the cell data `src` onto a grid of twice the number
of cells in each direction.
"""
function prolong!(target::CellData{NX,NY},src::CellData{NXhalf,NYhalf}) where {NX,NY,NXhalf,NYhalf}

    @assert NX == 2NXhalf && NY == 2NYhalf "Incompatible grid sizes for prolongation"

    for jhalf = 1:NYhalf+1, ihalf = 1:NXhalf+1
        target[2ihalf-1,2jhalf-1] = 9/16*src[ihalf,jhalf  ]+3/16*src[ihalf+1,jhalf  ]+
                                    3/16*src[ihalf,jhalf+1]+1/16*src[ihalf+1,jhalf+1]
        target[2ihalf  ,2jhalf-1] = 3/16*src[ihalf,jhalf  ]+9/16*src[ihalf+1,jhalf  ]+
                                    1/16*src[ihalf,jhalf+1]+3/16*src[ihalf+1,jhalf+1]
        target[2ihalf-1,2jhalf  ] = 3/16*src[ihalf,jhalf  ]+1/16*src[ihalf+1,jhalf  ]+
                                    9/16*src[ihalf,jhalf+1]+3/16*src[ihalf+1,jhalf+1]
        target[2ihalf  ,2jhalf  ] = 1/16*src[ihalf,jhalf  ]+3/16*src[ihalf+1,jhalf  ]+
                                    3/16*src[ihalf,jhalf+1]+9/16*src[ihalf+1,jhalf+1]
    end

end

"""
    prolong!(target::NodeData,src::NodeData)

Prolong the node data `src` onto a grid of twice the number
of cells in each direction. Boundary nodes are also affected.
"""
function prolong!(target::NodeData{NX,NY},src::NodeData{NXhalf,NYhalf}) where {NX,NY,NXhalf,NYhalf}

    @assert NX == 2NXhalf && NY == 2NYhalf "Incompatible grid sizes for prolongation"

    # The nodes that coincide on both grids
    for jhalf = 1:NYhalf+1, ihalf = 1:NXhalf+1
        target[2ihalf-1,2jhalf-1] = src[ihalf,jhalf]
    end

    # Fine nodes halfway between coarse nodes in x, aligned in y
    for jhalf = 1:NYhalf+1, ihalf = 1:NXhalf
        target[2ihalf,2jhalf-1] = 0.5*(src[ihalf,jhalf] + src[ihalf+1,jhalf])
    end

    # Fine nodes halfway between coarse nodes in y, aligned in x
    for jhalf = 1:NYhalf, ihalf = 1:NXhalf+1
        target[2ihalf-1,2jhalf] = 0.5*(src[ihalf,jhalf] + src[ihalf,jhalf+1])
    end

    # Fine nodes halfway between coarse nodes in y, aligned in x
    for jhalf = 1:NYhalf, ihalf = 1:NXhalf
        target[2ihalf,2jhalf] = 0.25*(src[ihalf  ,jhalf] + src[ihalf  ,jhalf+1]
                                    + src[ihalf+1,jhalf] + src[ihalf+1,jhalf+1])
    end

end
