# Boundary condition functions


export apply_bc!, apply_dirichlet_bc!, apply_neumann_bc!

# This function takes as input a set of edge data and returns the same
# data with the boundary and ghost values set as needed to enforce Dirichlet
# boundary conditions.
"""
    apply_bc!(u::EdgeData,t,p<:VectorDirichletParameters) -> EdgeData

Apply Dirichlet boundary conditions (specified in the Parameters structure `p`) at time
`t` in the given edge data `u`, overwriting the boundary and ghost values
in `u` with new values.
"""
function apply_bc!(u::EdgeData{NX,NY},t::Real,p::VectorDirichletParameters) where {NX,NY}

    # get the index ranges needed
    i_x, i_y = indices(u,1)
    j_x, j_y = indices(u,2)

    ### x components
    x = xmap(i_x,u.qx,p)
    y = ymap(j_x,u.qx,p)

    # on left, right sides, set x edges directly
    u.qx[1,   j_x] .= p.uL.(y,t)
    u.qx[NX+1,j_x] .= p.uR.(y,t)

    # on bottom, top sides, set x edges via averaging with ghost edges
    u.qx[i_x,1]    .= -u.qx[i_x,2]    + 2*p.uB.(x,t)
    u.qx[i_x,NY+2] .= -u.qx[i_x,NY+1] + 2*p.uT.(x,t)

    ### y components
    x = xmap(i_y,u.qy,p)
    y = ymap(j_y,u.qy,p)

    # on left, right sides, set y edges via averaging with ghost edges
    u.qy[1,   j_y] .= -u.qy[2,   j_y] + 2*p.vL.(y,t)
    u.qy[NX+2,j_y] .= -u.qy[NX+1,j_y] + 2*p.vR.(y,t)

    # on bottom, top sides, set y edges directly
    u.qy[i_y,1]    .= p.vB.(x,t)
    u.qy[i_y,NY+1] .= p.vT.(x,t)

    return u
end

"""
    apply_bc!(u::CellData,t,p::ScalarDirichletParameters) -> CellData

Apply Dirichlet boundary conditions (specified in the Parameters structure `p`) at time
`t` in the given cell data `u`, overwriting the ghost values
in `u` with new values.
"""
function apply_bc!(u::CellData{NX,NY},t::Real,p::ScalarDirichletParameters) where {NX,NY}

    # get the index ranges needed (interior cells)
    i = indices(u,1)
    j = indices(u,2)

    x = xmap(i,u,p)
    y = ymap(j,u,p)

    # on all sides, set zero boundary values via averaging with ghost cells
    u[1,   j] .= -u[2   ,j] + 2*p.uL.(y,t)
    u[NX+2,j] .= -u[NX+1,j] + 2*p.uR.(y,t)

    u[i,1]    .= -u[i,2   ] + 2*p.uB.(x,t)
    u[i,NY+2] .= -u[i,NY+1] + 2*p.uT.(x,t)

    # zero the corner ghosts
    u[1   ,1   ] = 0
    u[NX+2,1   ] = 0
    u[1   ,NY+2] = 0
    u[NX+2,NY+2] = 0

    return u
end

"""
    apply_dirichlet_bc!(u::EdgeData) -> EdgeData

Apply zero Dirichlet boundary conditions in the given edge data `u`, overwriting
the ghost and boundary values in `u` with new values.
"""
function apply_dirichlet_bc!(u::EdgeData{NX,NY}) where {NX,NY}

    # get the index ranges needed
    i_x, i_y = indices(u,1)
    j_x, j_y = indices(u,2)

    ### x components

    # on left, right sides, set x edges directly
    u.qx[1,   j_x] .= 0
    u.qx[NX+1,j_x] .= 0

    # on bottom, top sides, set x edges via averaging with ghost edges
    u.qx[i_x,1]    .= -u.qx[i_x,2]
    u.qx[i_x,NY+2] .= -u.qx[i_x,NY+1]

    ### y components

    # on left, right sides, set y edges via averaging with ghost edges
    u.qy[1,   j_y] .= -u.qy[2,   j_y]
    u.qy[NX+2,j_y] .= -u.qy[NX+1,j_y]

    # on bottom, top sides, set y edges directly
    u.qy[i_y,1]    .= 0
    u.qy[i_y,NY+1] .= 0

    return u
end

"""
    apply_dirichlet_bc!(u::CellData) -> CellData

Apply zero Dirichlet boundary conditions in the given cell data `u`, overwriting
the ghost values in `u` with new values.
"""
function apply_dirichlet_bc!(u::CellData{NX,NY}) where {NX,NY}

    # get the index ranges needed (interior cells)
    i = indices(u,1)
    j = indices(u,2)

    # on all sides, set zero boundary values via averaging with ghost cells
    u[1,   j] .= -u[2   ,j]
    u[NX+2,j] .= -u[NX+1,j]

    u[i,1]    .= -u[i,2   ]
    u[i,NY+2] .= -u[i,NY+1]

    # zero the corner ghosts
    u[1   ,1   ] = 0
    u[NX+2,1   ] = 0
    u[1   ,NY+2] = 0
    u[NX+2,NY+2] = 0

    return u
end

"""
    apply_dirichlet_bc!(u::NodeData) -> NodeData

Apply zero Dirichlet boundary conditions in the given node data `u`, overwriting
the boundary values in `u` with new values.
"""
function apply_dirichlet_bc!(u::NodeData{NX,NY}) where {NX,NY}

    # get the index ranges needed (interior cells)
    i = indices(u,1)
    j = indices(u,2)

    # on all sides, set zero boundary values via averaging with ghost cells
    u[1,   j] .= 0
    u[NX+1,j] .= 0

    u[i,1]    .= 0
    u[i,NY+1] .= 0

    return u
end

"""
    apply_bc!(u::CellData,t,p::ScalarNeumannParameters) -> EdgeData

Apply Neumann boundary conditions (specified in the Parameters structure `p`) at time
`t` in the given cell data `u`, overwriting the ghost values
in `u` with new values.
"""
function apply_bc!(u::CellData{NX,NY},t::Real,p::ScalarNeumannParameters) where {NX,NY}

    # get the index ranges needed (interior cells)
    i = indices(u,1)
    j = indices(u,2)

    x = xmap(i,u,p)
    y = ymap(j,u,p)

    # on all sides, set zero boundary values via averaging with ghost cells
    u[1,   j] .= u[2   ,j] + p.Δx*p.qL.(y,t)
    u[NX+2,j] .= u[NX+1,j] + p.Δx*p.qR.(y,t)

    u[i,1]    .= u[i,2   ] + p.Δx*p.qB.(x,t)
    u[i,NY+2] .= u[i,NY+1] + p.Δx*p.qT.(x,t)

    # zero the corner ghosts
    u[1   ,1   ] = 0
    u[NX+2,1   ] = 0
    u[1   ,NY+2] = 0
    u[NX+2,NY+2] = 0

    return u
end

"""
    apply_neumann_bc!(u::CellData) -> CellData

Apply zero Neumann boundary conditions in the given cell data `u`, overwriting
the ghost values in `u` with new values.
"""
function apply_neumann_bc!(u::CellData{NX,NY}) where {NX,NY}

    # get the index ranges needed (interior cells)
    i = indices(u,1)
    j = indices(u,2)

    # on all sides, set zero Neumann boundary values via differencing with ghost cells
    u[1,   j] .= u[2   ,j]
    u[NX+2,j] .= u[NX+1,j]

    u[i,1]    .= u[i,2   ]
    u[i,NY+2] .= u[i,NY+1]

    # zero the corner ghosts
    u[1   ,1   ] = 0
    u[NX+2,1   ] = 0
    u[1   ,NY+2] = 0
    u[NX+2,NY+2] = 0

    return u
end
