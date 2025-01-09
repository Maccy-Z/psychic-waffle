import torch
from utils_sparse import plot_sparsity
from graph_grid.graph_utils import plot_interp_graph


torch.set_printoptions(precision=3, sci_mode=False)
save_dict = torch.load("jacobian_residuals.pt")
jac = save_dict["jacobian"]
resid = save_dict["residuals"]
div = save_dict["aux_input"]
mask = save_dict["mask"]
graph_dict = torch.load("graph.pth")
Xs, grad_mask = graph_dict["Xs"], graph_dict["grad_mask"]

# plot_sparsity(jac)

print(f'{jac.shape = }, {resid.shape = }, {div.shape = }')
A = jac.to_dense()
b = torch.zeros_like(resid)
b[mask] = div#resid.to_dense()
Ab = torch.hstack([A, b.unsqueeze(-1)])
# print(f'{div.abs().max() = }')

A = A.double()
b = b.double()
Ab = Ab.double()

print("Ranks")
print("A", torch.linalg.matrix_rank(A).cpu())
print("Ab", torch.linalg.matrix_rank(Ab).cpu())

print()
us = torch.linalg.solve(A, b)
error = torch.max(torch.abs(A @ us - b)).cpu()
print("Error solve = ", error.item())
# print(x)

us_all = torch.zeros_like(Xs[:, 0])
us_all[grad_mask] = us.squeeze().float()
plot_interp_graph(Xs, us_all)

print(f'{us.shape = }')
us2 = torch.linalg.lstsq(A, b).solution
error = torch.max(torch.abs(A @ us2 - b)).cpu().item()
print(f'Error lstsq {error = }')