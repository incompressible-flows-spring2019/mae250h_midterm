export diffuse2d_dirichlet_cn, diffuse2d_dirichlet_cn_delta


# Crank-Nicolson for diffusion on edge data
"""
    diffuse2d_dirichlet_cn(t::Real,u::EdgeData,rhs::EdgeData,p::NSParameters) -> Real, EdgeData

Advance the 2-d diffusion problem, based on a Crank-Nicolson time discretization,
from time `t` to time `t + Δt`, where `Δt` is provided in the parameters structure `p`.
The function is called with the non-diffusion part of the right-hand side of the problem given in `rhs`,
and the solution data at time `t` provided in `u`. The new time level and the new
solution data are output.
"""
function diffuse2d_dirichlet_cn(t::Real,u::EdgeData{NX,NY},rhs::EdgeData{NX,NY},p::NSParameters) where {NX,NY}

    i_x, i_y = indices(u,1)
    j_x, j_y = indices(u,2)

    # get grid Fourier number from problem parameters
    Fo = p.Fo

    # This will hold the right-hand side at first, but ultimately hold
    # the solution
    unp1 = deepcopy(u)

    # Add 1/2*Fo*L(u), after setting the current boundary values
    # in u
    apply_bc!(u,t,p)
    unp1 .+= 0.5Fo*laplacian(u,1) # in the x direction
    unp1 .+= 0.5Fo*laplacian(unp1,2)  # in the y direction

    # Now apply the boundary conditions at t+Δt. For this, we set up
    # blank edge data and then apply the boundary conditions to the
    # ghosts, boundaries
    ubc   = EdgeData(u)
    apply_bc!(ubc,t+p.Δt,p)
    unp1 .+= 0.5Fo*laplacian(ubc)

    # Add the existing right-hand side
    unp1 .+= rhs

    # now unp1 holds the right-hand side. Now set up the tridiagonal
    # solution, first in the x direction, then in the y direction
    axlap, bxlap, cxlap = 1,-2,1
    ax, bx, cx = -0.5Fo*axlap, 1-0.5Fo*bxlap, -0.5Fo*cxlap

    aylap, bylap, cylap = 1*ones(NX-1), -2*ones(NX), 1*ones(NX-1)
    bylap[1] -= 1
    bylap[NX] -= 1
    ay, by, cy = -0.5Fo*aylap, 1 .- 0.5Fo*bylap, -0.5Fo*cylap


    for j in j_x # Loop through the interior x edge rows
        unp1.qx[i_x,j] .= trisolve(ax,bx,cx,unp1.qx[i_x,j],"regular")
    end
    for j in j_y # Loop through the interior y edge rows
        unp1.qy[i_y,j] .= trisolve(ay,by,cy,unp1.qy[i_y,j],"regular")
    end

    # now unp1 contains the solution in the x direction. This
    # supplies the right-hand side in the y direction. But first,
    # we need to transpose the data

    qxtmp = deepcopy(transpose(unp1.qx))
    qytmp = deepcopy(transpose(unp1.qy))

    axlap, bxlap, cxlap = 1*ones(NY-1), -2*ones(NY), 1*ones(NY-1)
    bxlap[1] -= 1
    bxlap[NY] -= 1
    ax, bx, cx = -0.5Fo*axlap, 1 .- 0.5Fo*bxlap, -0.5Fo*cxlap

    aylap, bylap, cylap = 1,-2,1
    ay, by, cy = -0.5Fo*aylap, 1-0.5Fo*bylap, -0.5Fo*cylap

    for i in i_x # Loop through the interior x edge columns
        qxtmp[j_x,i] .= trisolve(ax,bx,cx,qxtmp[j_x,i],"regular")
    end
    for i in i_y # Loop through the interior y edge columns
        qytmp[j_y,i] .= trisolve(ay,by,cy,qytmp[j_y,i],"regular")
    end

    # now transpose back
    unp1.qx .= transpose(qxtmp)
    unp1.qy .= transpose(qytmp)

    # now place boundary values in the appropriate places
    apply_bc!(unp1,t+p.Δt,p)

    return t+p.Δt, unp1
end

"""
    diffuse2d_dirichlet_cn_delta(t::Real,u::EdgeData,rhs::EdgeData,p::NSParameters) -> Real, EdgeData

Advance the 2-d diffusion problem, based on a delta formulation of the Crank-Nicolson time discretization,
from time `t` to time `t + Δt`, where `Δt` is provided in the parameters structure `p`.
The function is called with the non-diffusion part of the right-hand side of the problem given in `rhs`,
and the solution data at time `t` provided in `u`. The new time level and the change in the
solution data from `t` to `t+Δt` are output.
"""
function diffuse2d_dirichlet_cn_delta(t::Real,u::EdgeData{NX,NY},rhs::EdgeData{NX,NY},p::NSParameters) where {NX,NY}

    i_x, i_y = indices(u,1)
    j_x, j_y = indices(u,2)

    # get grid Fourier number from problem parameters
    Fo = p.Fo

    # This will hold the right-hand side at first, but ultimately hold
    # the solution
    δu = EdgeData(u)

    # Add Fo*L(u), after setting zero boundary values
    # in u. (We account for the actual bcs below.)
    # No need to factorize right hand side because the left-hand side
    # operator acts on δu, so the error term is already O(Δt^3)
    apply_dirichlet_bc!(u)
    δu .= Fo*laplacian(u)

    # Now apply the boundary conditions at t and t+Δt. For this, we set up
    # blank edge data and then apply the boundary conditions to the
    # ghosts, boundaries
    δubc   = EdgeData(u)
    apply_bc!(δubc,t,p) # at t
    δu .+= 0.5Fo*laplacian(δubc)
    apply_bc!(δubc,t+p.Δt,p) # at t+Δt
    δu .+= 0.5Fo*laplacian(δubc)

    # Add the existing right-hand side
    δu .+= rhs

    # now unp1 holds the right-hand side. Now set up the tridiagonal
    # solution, first in the x direction, then in the y direction
    axlap, bxlap, cxlap = 1,-2,1
    ax, bx, cx = -0.5Fo*axlap, 1-0.5Fo*bxlap, -0.5Fo*cxlap

    aylap, bylap, cylap = 1*ones(NX-1), -2*ones(NX), 1*ones(NX-1)
    bylap[1] -= 1
    bylap[NX] -= 1
    ay, by, cy = -0.5Fo*aylap, 1 .- 0.5Fo*bylap, -0.5Fo*cylap


    for j in j_x # Loop through the interior x edge rows
        δu.qx[i_x,j] .= trisolve(ax,bx,cx,δu.qx[i_x,j],"regular")
    end
    for j in j_y # Loop through the interior y edge rows
        δu.qy[i_y,j] .= trisolve(ay,by,cy,δu.qy[i_y,j],"regular")
    end

    # now unp1 contains the solution in the x direction. This
    # supplies the right-hand side in the y direction. But first,
    # we need to transpose the data

    qxtmp = deepcopy(transpose(δu.qx))
    qytmp = deepcopy(transpose(δu.qy))

    axlap, bxlap, cxlap = 1*ones(NY-1), -2*ones(NY), 1*ones(NY-1)
    bxlap[1] -= 1
    bxlap[NY] -= 1
    ax, bx, cx = -0.5Fo*axlap, 1 .- 0.5Fo*bxlap, -0.5Fo*cxlap

    aylap, bylap, cylap = 1,-2,1
    ay, by, cy = -0.5Fo*aylap, 1-0.5Fo*bylap, -0.5Fo*cylap

    for i in i_x # Loop through the interior x edge columns
        qxtmp[j_x,i] .= trisolve(ax,bx,cx,qxtmp[j_x,i],"regular")
    end
    for i in i_y # Loop through the interior y edge columns
        qytmp[j_y,i] .= trisolve(ay,by,cy,qytmp[j_y,i],"regular")
    end

    # now transpose back
    δu.qx .= transpose(qxtmp)
    δu.qy .= transpose(qytmp)

    # now place zero boundary values in the appropriate places
    apply_dirichlet_bc!(δu)

    return t+p.Δt, δu
end
