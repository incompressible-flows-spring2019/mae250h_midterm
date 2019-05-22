export convection

"""
    convection(q::EdgeData) -> EdgeData

Compute the non-linear convective term, in divergence form. Note that this
does not scale the result by grid spacing.
"""
function convection(q::EdgeData{NX,NY}) where {NX,NY}

  nlq = EdgeData(q)

  tmpc = CellData(q)
  tmpnu = NodeData(q)
  tmpnv = NodeData(q)
  tmpnl = EdgeData(q)

  interpolate!(tmpc,q.qx)
  diff!(nlq.qx,tmpc*tmpc) # du^2/dx

  interpolate!(tmpnu,q.qx)
  interpolate!(tmpnv,q.qy)
  diff!(tmpnl.qx,tmpnu*tmpnv) # dvu/dy
  nlq.qx .+= tmpnl.qx

  interpolate!(tmpc,q.qy)
  diff!(nlq.qy,tmpc*tmpc) # dv^2/dy

  diff!(tmpnl.qy,tmpnu*tmpnv) # duv/dx
  nlq.qy .+= tmpnl.qy


  return nlq

end
