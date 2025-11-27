import torch
from torch import nn
from torch.optim import LBFGS

# Suppose:
# - STLDataClass is your data wrapper class (e.g. STL2DKernel)
# - st_op is an operator such that st_op.apply(DC).to_flatten()
#   returns a 1D tensor of scattering coefficients.
# - target is your reference map of shape (1,1,128,128)
#   and r is the corresponding scattering vector.


class ScatteringMatchModel(nn.Module):
    def __init__(self, st_op, STLDataClass, init_shape, device=None, dtype=None):
        super().__init__()
        self.st_op = st_op
        self.STLDataClass = STLDataClass

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float32

        # Learnable field u
        self.u = STLDataClass(torch.randn(init_shape, device=device, dtype=dtype))
        w_op = self.u.get_wavelet_op()
        self.u = nn.Parameter(w_op.apply_smooth(self.u,copy=True).array)
        

    def forward(self):
        DC_u = self.STLDataClass(self.u)
        st_u = self.st_op.apply(DC_u) #,norm='load_ref')
        s_flat_u = st_u.to_flatten()
        return s_flat_u


def optimize_scattering_LBFGS(
    target,
    STLDataClass,
    SO_class,
    max_iter=100,
    lr=1.0,
    history_size=50,
    verbose=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.as_tensor(target, device=device, dtype=torch.float32)

    # Reference scattering
    DC_target = STLDataClass(target)
    st_op = SO_class(DC_target)
    with torch.no_grad():
        r = st_op.apply(DC_target).to_flatten() #,norm='store_ref').to_flatten()
    r = r.detach()

    # Model with learnable u
    model = ScatteringMatchModel(
        st_op=st_op,
        STLDataClass=STLDataClass,
        init_shape=target.shape,
        device=device,
        dtype=target.dtype,
    )

    optimizer = LBFGS(
        [model.u],
        lr=lr,
        max_iter=max_iter,        # <-- le nombre d'itérations internes LBFGS
        history_size=history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12, 
        tolerance_change=1e-15
    )

    loss_history = []

    def closure():
        optimizer.zero_grad()
        s_flat_u = model()
        loss = ((s_flat_u - r).abs() ** 2).sum()
        loss.backward()

        # Log à chaque appel interne
        loss_val = loss.item()
        loss_history.append(loss_val)
        if verbose:
            print(f"[LBFGS] inner iter {len(loss_history)}, loss = {loss_val:.6e}")

        return loss

    # Un seul appel : toutes les itérations LBFGS internes sont faites ici
    optimizer.step(closure)

    u_opt = model.u.detach()
    return u_opt, loss_history

