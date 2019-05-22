# Poisson equation routines

using FFTW

export smooth!, mg!, mgcycle!, RelaxationParameters, MGParameters, poisson_neumann_fft!

abstract type PoissonSolverParameters end

"""
    RelaxationParameters

Set parameters for relaxation solution.

# Constructors
- `RelaxationParameters(apply_bc!,niter,type)`
"""
struct RelaxationParameters <: PoissonSolverParameters

  "Boundary condition function"
  apply_bc!

  "Number of iterations"
  niter :: Int

  "Solver type"
  solver_type :: String

end

"""
    smooth!(u::CellData,f::CellData,p::RelaxationParameters) -> Float64

Perform `p.niter` iterations of Gauss-Seidel smoothing of the cell data in
`u`. The right-hand side of the discrete Laplace equation is in `f`. Specifically,
this computes

u[i,j] = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])/4 - f[i,j]/4

at each iteration, using updated values on the right-hand side when available.
Note that the physical grid spacing is not used in this equation, so it must
be accounted for in `f` in the function call. The function returns the smoothed
version of `u` in place. It also returns the norm of the residual.

The argument `p.solver_type` sets the relaxation method. It can be set to `"GS"`
for basic Gauss-Seidel (the default), or the `"checkerboard"` for checkerboard
Gauss-Seidel.
"""
function smooth!(u::ScalarData{NX,NY},f::ScalarData{NX,NY},p::RelaxationParameters) where {NX,NY}

  if p.solver_type == "GS"
    return gs!(u,f,p)
  elseif p.solver_type == "checkerboard"
    return gs_checkerboard!(u,f,p)
  end

end

function gs!(u::CellData{NX,NY},f::CellData{NX,NY},p::RelaxationParameters) where {NX,NY}

    resid_norm = 1.0
    for iter = 1:p.niter
      p.apply_bc!(u)
      for j = 2:NY+1, i = 2:NX+1
        u[i,j] = 0.25*(u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]) - 0.25*f[i,j]
      end
      resid_norm = norm(f - laplacian(u))
    end

    return resid_norm
end

function gs!(u::NodeData{NX,NY},f::NodeData{NX,NY},p::RelaxationParameters) where {NX,NY}

    resid_norm = 1.0
    for iter = 1:p.niter
      p.apply_bc!(u)
      for j = 2:NY, i = 2:NX
        u[i,j] = 0.25*(u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]) - 0.25*f[i,j]
      end
      resid_norm = norm(f - laplacian(u))
    end

    return resid_norm
end


function gs_checkerboard!(u::CellData{NX,NY},f::CellData{NX,NY},p::RelaxationParameters) where {NX,NY}

  tmp = zero(transpose(u))
  resid_norm = 1.0
  for iter = 1:p.niter
    for color = 0:1
      p.apply_bc!(u)
      stag = color
      for j = 2:NY+1 # interior rows
        u[2+stag:2:NX+stag,j] .= -0.25*f[2+stag:2:NX+stag,j] +
                                  0.25*(u[1+stag:2:NX-1+stag,j] + u[3+stag:2:NX+1+stag,j])
        stag = 1-stag # switch center color from one row to the next
      end

      # now transpose to include the (i,j-1) and (i,j+1) elements
      tmp .= transpose(u)
      stag = color
      for i = 2:NX+1 # interior columns
        tmp[2+stag:2:NY+stag,i] .+= 0.25*(tmp[1+stag:2:NY-1+stag,i]+tmp[3+stag:2:NY+1+stag,i])
        stag = 1-stag # switch center color from one column to the next
      end
      u .= transpose(tmp)

    end
    resid_norm = norm(f - laplacian(u))
  end
  return resid_norm

end

function gs_checkerboard!(u::NodeData{NX,NY},f::NodeData{NX,NY},p::RelaxationParameters) where {NX,NY}

  tmp = zero(transpose(u))
  resid_norm = 1.0
  for iter = 1:p.niter
    for color = 0:1
      p.apply_bc!(u)
      stag = color
      for j = 2:NY # interior rows
        u[2+stag:2:NX-stag,j] .= -0.25*f[2+stag:2:NX-stag,j] +
                                  0.25*(u[1+stag:2:NX-1-stag,j] + u[3+stag:2:NX+1-stag,j])
        stag = 1-stag # switch center color from one row to the next
      end

      # now transpose to include the (i,j-1) and (i,j+1) elements
      tmp .= transpose(u)
      stag = color
      for i = 2:NX # interior columns
        tmp[2+stag:2:NY-stag,i] .+= 0.25*(tmp[1+stag:2:NY-1-stag,i]+tmp[3+stag:2:NY+1-stag,i])
        stag = 1-stag # switch center color from one column to the next
      end
      u .= transpose(tmp)

    end
    resid_norm = norm(f - laplacian(u))
  end
  return resid_norm

end

# this version is not quite working.
function gs_checkerboard_2!(u::CellData{NX,NY},f::CellData{NX,NY};niter=1) where {NX,NY}

  resid_norm = 1.0
  I = CellData(u)
  I .= 1:NX+2
  I .+= transpose(1:NY+2)
  red = mod.(I,2)

  for iter = 1:niter
    u .= u .* red
    u .=  -0.25*(f - laplacian(u)).*(1 .- red)
    apply_bc!(u)
    u .+= -0.25*(f - laplacian(u)).*red
    resid_norm = norm(f - laplacian(u))
  end
  apply_bc!(u)
  return resid_norm

end

####### MULTIGRID #########

"""
    MGParameters

Set parameters for multigrid solution.

# Constructors
- `MGParameters(apply_bc!,niteri,niter1,niter2,niterf,maxlev,gstol,gsmaxiter,maxcycle,tol)`
"""
struct MGParameters <: PoissonSolverParameters

  "Boundary condition function"
  apply_bc!

  "Number of initial sweeps on finest grid"
  niteri :: Int

  "Number of smoothing sweeps after each restriction"
  niter1 :: Int

  "Number of smoothing sweeps after each prolongation"
  niter2 :: Int

  "Number of final sweeps on finest grid"
  niterf :: Int

  "Number of coarseness levels to descend. Finest grid at level 1"
  maxlev :: Int

  "GS tolerance"
  gstol :: Float64

  "Maximum number of GS iterations"
  gsmaxiter :: Int

  "Maximum number of multigrid cycles"
  maxcycle :: Int

  "Overall tolerance"
  tol :: Float64
end

"""
    mg!(u::ScalarData,f::ScalarData,p::MGParameters) -> Float64

Uses multigrid to solve the Poisson system Lu = f, with the right-hand side `f`
and initial guess `u`, and multigrid parameters provided in `p`. The function returns
the final solution `u` in place. It also returns the final residual norm.
"""
function mg!(u::ScalarData{NX,NY},f::ScalarData{NX,NY},p::MGParameters) where {NX,NY}

  T = eval(nameof(typeof(u))) # returns grid data type, e.g., CellData, without parameters

  uh = T(u)
  u2h = T(Int(NX/2),Int(NY/2))
  for iter = 1:p.maxcycle
    uh .= f - laplacian(u)
    p.apply_bc!(uh)
    resid_norm = mgcycle!(u2h,2,uh,p)
    prolong!(uh,u2h)
    u .+= uh
    global resid_norm = smooth!(u,f,RelaxationParameters(p.apply_bc!,p.niterf,"GS"))

    if resid_norm < p.tol
        #println("Residual norm = ",resid_norm)
        #println("Number of iterations = ",iter)
        break
    end
  end
  return resid_norm

end

"""
    mgcycle!(u::ScalarData,lev::Int,r::ScalarData,p::MGParameters) -> Float64

Takes the residual `r` from a finer grid at level `lev-1` and returns the
smoothed correction `u` at level `lev` as well as the residual norm at that level.
"""
function mgcycle!(u::ScalarData{NXhalf,NYhalf},lev::Int,r::ScalarData{NX,NY},p::MGParameters) where {NXhalf,NYhalf,NX,NY}

  T = eval(nameof(typeof(u))) # returns grid data type, e.g., CellData, without parameters

  f = T(u)
  uh = T(u)

  restrict!(f,r)

  if lev < p.maxlev
    # Perform smoothing at this level. Start with zero correction, since we
    # are only computing what will be added to the next-finer grid's
    # correction after prolongation
    fill!(u,0)
    # note that factor of 4 below accounts for twice the grid spacing at level lev
    resid_norm = smooth!(u,4*f,RelaxationParameters(p.apply_bc!,p.niter1,"GS"))


    # Compute residual (held in this temporary variable uh)
    # note that factor of 0.25 below accounts for twice the grid spacing at level lev
    uh .= f - 0.25*laplacian(u)

    # Apply zero conditions to uh
    p.apply_bc!(uh)

    # Cycle through next coarser level
    u2h = T(Int(NXhalf/2),Int(NYhalf/2))
    mgcycle!(u2h,lev+1,uh,p)

    # Prolong the coarser cycle data and add it (as a correction) to this
    # level's correction
    prolong!(uh,u2h)
    u .+= uh

    # Smooth it some more
    resid_norm = smooth!(u,4*f,RelaxationParameters(p.apply_bc!,p.niter2,"GS"))

  else

    # Solve it at the coarsest level to completion
    for iter = 1:p.gsmaxiter
        resid_norm = smooth!(u,4*f,RelaxationParameters(p.apply_bc!,1,"GS"))
        #println("residual at coarsest = ",resid_norm)
        #println(size(u))
        if resid_norm < p.gstol
          break
        end
    end

  end
  return resid_norm
end

"""
    poisson_neumann_fft!(u::CellData) -> CellData

Uses a discrete cosine transform to solve a Neumann problem. The right-hand
side should be supplied as the sole argument, and the solution is returned in
its place.
"""
function poisson_neumann_fft!(u::CellData{NX,NY}) where {NX,NY}

  f = view(u,2:NX+1,2:NY+1)
  f .= FFTW.dct(f,1)
  f .= FFTW.dct(f,2)

  Λ = zero(f)
  Λ .= cos.(pi*(0:NX-1)'/NX) .+ cos.(pi*(0:NY-1)/NY) .- 2ones(NX,NY)
  Λ[1,1] = 1
  f[1,1] = 0
  f .= 0.5*f./Λ

  f .= FFTW.idct(f,1)
  f .= FFTW.idct(f,2)

  return u

end
