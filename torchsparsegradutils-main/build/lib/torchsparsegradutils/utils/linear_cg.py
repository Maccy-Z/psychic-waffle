# MIT-licensed code imported from https://github.com/cornellius-gp/linear_operator
# Minor modifications for torchsparsegradutils to remove dependencies

import warnings

import torch

from typing import NamedTuple


class LinearCGSettings(NamedTuple):
    max_cg_iterations: int = 1000  # The maximum number of conjugate gradient iterations to perform (when computing
    # matrix solves). A higher value rarely results in more accurate solves -- instead, lower the CG tolerance.
    max_lanczos_quadrature_iterations: int = (
        20  # The maximum number of Lanczos iterations to perform when doing stochastic
    )
    # Lanczos quadrature. This is ONLY used for log determinant calculations and
    # computing Tr(K^{-1}dK/d\theta)
    cg_tolerance: float = 1  # Relative residual tolerance to use for terminating CG.
    terminate_cg_by_size: bool = False  # If set to true, cg will terminate after n iterations for an n x n matrix.
    verbose_linalg: bool = False  # Print out information whenever running an expensive linear algebra routine


def _default_preconditioner(x):
    return x.clone()


@torch.jit.script
def _jit_linear_cg_updates(
    result, alpha, residual_inner_prod, eps, beta, residual, precond_residual, mul_storage, is_zero, curr_conjugate_vec
):
    # # Update result
    # # result_{k} = result_{k-1} + alpha_{k} p_vec_{k-1}
    result = torch.addcmul(result, alpha, curr_conjugate_vec, out=result)

    # beta_{k} = (precon_residual{k}^T r_vec_{k}) / (precon_residual{k-1}^T r_vec_{k-1})
    beta.resize_as_(residual_inner_prod).copy_(residual_inner_prod)
    torch.mul(residual, precond_residual, out=mul_storage)
    torch.sum(mul_storage, -2, keepdim=True, out=residual_inner_prod)

    # Do a safe division here
    torch.lt(beta, eps, out=is_zero)
    beta.masked_fill_(is_zero, 1)
    torch.div(residual_inner_prod, beta, out=beta)
    beta.masked_fill_(is_zero, 0)

    # Update curr_conjugate_vec
    # curr_conjugate_vec_{k} = precon_residual{k} + beta_{k} curr_conjugate_vec_{k-1}
    curr_conjugate_vec.mul_(beta).add_(precond_residual)


@torch.jit.script
def _jit_linear_cg_updates_no_precond(
    mvms,
    result,
    has_converged,
    alpha,
    residual_inner_prod,
    eps,
    beta,
    residual,
    precond_residual,
    mul_storage,
    is_zero,
    curr_conjugate_vec,
):
    torch.mul(curr_conjugate_vec, mvms, out=mul_storage)
    torch.sum(mul_storage, dim=-2, keepdim=True, out=alpha)

    # Do a safe division here
    torch.lt(alpha, eps, out=is_zero)
    alpha.masked_fill_(is_zero, 1)
    torch.div(residual_inner_prod, alpha, out=alpha)
    alpha.masked_fill_(is_zero, 0)

    # We'll cancel out any updates by setting alpha=0 for any vector that has already converged
    alpha.masked_fill_(has_converged, 0)

    # Update residual
    # residual_{k} = residual_{k-1} - alpha_{k} mat p_vec_{k-1}
    torch.addcmul(residual, -alpha, mvms, out=residual)

    # Update precond_residual
    # precon_residual{k} = M^-1 residual_{k}
    precond_residual = residual.clone()

    _jit_linear_cg_updates(
        result,
        alpha,
        residual_inner_prod,
        eps,
        beta,
        residual,
        precond_residual,
        mul_storage,
        is_zero,
        curr_conjugate_vec,
    )


def linear_cg(
    matmul_closure,
    rhs,
    n_tridiag=0,
    tolerance=None,
    eps=1e-10,
    stop_updating_after=1e-10,
    max_iter=None,
    max_tridiag_iter=None,
    initial_guess=None,
    preconditioner=None,
    settings=LinearCGSettings(),
):
    """
    Implements the linear conjugate gradients method for (approximately) solving systems of the form

        lhs result = rhs

    for positive definite and symmetric matrices.

    Args:
      - matmul_closure - a function which performs a left matrix multiplication with lhs_mat
      - rhs - the right-hand side of the equation
      - n_tridiag - returns a tridiagonalization of the first n_tridiag columns of rhs
      - tolerance - stop the solve when the (average) norm of the residual(s) is less than this
      - eps - noise to add to prevent division by zero
      - stop_updating_after - will stop updating a vector after this residual norm is reached
      - max_iter - the maximum number of CG iterations
      - max_tridiag_iter - the maximum size of the tridiagonalization matrix
      - initial_guess - an initial guess at the solution `result`
      - precondition_closure - a functions which left-preconditions a supplied vector

    Returns:
      result - a solution to the system (if n_tridiag is 0)
      result, tridiags - a solution to the system, and corresponding tridiagonal matrices (if n_tridiag > 0)
    """
    # Unsqueeze, if necesasry
    is_vector = rhs.ndimension() == 1
    if is_vector:
        rhs = rhs.unsqueeze(-1)

    # Some default arguments
    if max_iter is None:
        max_iter = settings.max_cg_iterations
    if max_tridiag_iter is None:
        max_tridiag_iter = settings.max_lanczos_quadrature_iterations
    if initial_guess is None:
        initial_guess = torch.zeros_like(rhs)
    else:
        # Unsqueeze, if necesasry
        is_vector = initial_guess.ndimension() == 1
        if is_vector:
            initial_guess = initial_guess.unsqueeze(-1)
    if tolerance is None:
        tolerance = settings.cg_tolerance
    if preconditioner is None:
        preconditioner = _default_preconditioner
        precond = False
    else:
        precond = True

    # If we are running m CG iterations, we obviously can't get more than m Lanczos coefficients
    if max_tridiag_iter > max_iter:
        raise RuntimeError("Getting a tridiagonalization larger than the number of CG iterations run is not possible!")

    # Check matmul_closure object
    if torch.is_tensor(matmul_closure):
        matmul_closure = matmul_closure.matmul
    elif not callable(matmul_closure):
        raise RuntimeError("matmul_closure must be a tensor, or a callable object!")

    # Get some constants
    num_rows = rhs.size(-2)
    n_iter = min(max_iter, num_rows) if settings.terminate_cg_by_size else max_iter
    n_tridiag_iter = min(max_tridiag_iter, num_rows)
    eps = torch.tensor(eps, dtype=rhs.dtype, device=rhs.device)

    # Get the norm of the rhs - used for convergence checks
    # Here we're going to make almost-zero norms actually be 1 (so we don't get divide-by-zero issues)
    # But we'll store which norms were actually close to zero
    rhs_norm = rhs.norm(2, dim=-2, keepdim=True)
    rhs_is_zero = rhs_norm.lt(eps)
    rhs_norm = rhs_norm.masked_fill_(rhs_is_zero, 1)

    # Let's normalize. We'll un-normalize afterwards
    rhs = rhs.div(rhs_norm)
    initial_guess = initial_guess.div(rhs_norm)

    # residual: residual_{0} = b_vec - lhs x_{0}
    residual = rhs - matmul_closure(initial_guess)
    batch_shape = residual.shape[:-2]

    # result <- x_{0}
    result = initial_guess.expand_as(residual).contiguous()

    # Maybe log
    if settings.verbose_linalg:
        # settings.verbose_linalg.logger.debug(
        print(f"Running CG on a {rhs.shape} RHS for {n_iter} iterations (tol={tolerance}). Output: {result.shape}.")

    # Check for NaNs
    if not torch.equal(residual, residual):
        raise RuntimeError("NaNs encountered when trying to perform matrix-vector multiplication")

    # Sometime we're lucky and the preconditioner solves the system right away
    # Check for convergence
    residual_norm = residual.norm(2, dim=-2, keepdim=True)
    has_converged = torch.lt(residual_norm, stop_updating_after)

    if has_converged.all() and not n_tridiag:
        n_iter = 0  # Skip the iteration!

    # Otherwise, let's define precond_residual and curr_conjugate_vec
    else:
        # precon_residual{0} = M^-1 residual_{0}
        precond_residual = preconditioner(residual)
        curr_conjugate_vec = precond_residual
        residual_inner_prod = precond_residual.mul(residual).sum(-2, keepdim=True)

        # Define storage matrices
        mul_storage = torch.empty_like(residual)
        alpha = torch.empty(*batch_shape, 1, rhs.size(-1), dtype=residual.dtype, device=residual.device)
        beta = torch.empty_like(alpha)
        is_zero = torch.empty(*batch_shape, 1, rhs.size(-1), dtype=torch.bool, device=residual.device)

    # Define tridiagonal matrices, if applicable
    if n_tridiag:
        t_mat = torch.zeros(
            n_tridiag_iter, n_tridiag_iter, *batch_shape, n_tridiag, dtype=alpha.dtype, device=alpha.device
        )
        alpha_tridiag_is_zero = torch.empty(*batch_shape, n_tridiag, dtype=torch.bool, device=t_mat.device)
        alpha_reciprocal = torch.empty(*batch_shape, n_tridiag, dtype=t_mat.dtype, device=t_mat.device)
        prev_alpha_reciprocal = torch.empty_like(alpha_reciprocal)
        prev_beta = torch.empty_like(alpha_reciprocal)

    update_tridiag = True
    last_tridiag_iter = 0

    # It's conceivable we reach the tolerance on the last iteration, so can't just check iteration number.
    tolerance_reached = False

    # Start the iteration
    for k in range(n_iter):
        # Get next alpha
        # alpha_{k} = (residual_{k-1}^T precon_residual{k-1}) / (p_vec_{k-1}^T mat p_vec_{k-1})
        mvms = matmul_closure(curr_conjugate_vec)
        if precond:
            torch.mul(curr_conjugate_vec, mvms, out=mul_storage)
            torch.sum(mul_storage, -2, keepdim=True, out=alpha)

            # Do a safe division here
            torch.lt(alpha, eps, out=is_zero)
            alpha.masked_fill_(is_zero, 1)
            torch.div(residual_inner_prod, alpha, out=alpha)
            alpha.masked_fill_(is_zero, 0)

            # We'll cancel out any updates by setting alpha=0 for any vector that has already converged
            alpha.masked_fill_(has_converged, 0)

            # Update residual
            # residual_{k} = residual_{k-1} - alpha_{k} mat p_vec_{k-1}
            residual = torch.addcmul(residual, alpha, mvms, value=-1, out=residual)

            # Update precond_residual
            # precon_residual{k} = M^-1 residual_{k}
            precond_residual = preconditioner(residual)

            _jit_linear_cg_updates(
                result,
                alpha,
                residual_inner_prod,
                eps,
                beta,
                residual,
                precond_residual,
                mul_storage,
                is_zero,
                curr_conjugate_vec,
            )
        else:
            _jit_linear_cg_updates_no_precond(
                mvms,
                result,
                has_converged,
                alpha,
                residual_inner_prod,
                eps,
                beta,
                residual,
                precond_residual,
                mul_storage,
                is_zero,
                curr_conjugate_vec,
            )

        torch.norm(residual, 2, dim=-2, keepdim=True, out=residual_norm)
        residual_norm.masked_fill_(rhs_is_zero, 0)
        torch.lt(residual_norm, stop_updating_after, out=has_converged)

        if (
            k >= min(10, max_iter - 1)
            and bool(residual_norm.mean() < tolerance)
            and not (n_tridiag and k < min(n_tridiag_iter, max_iter - 1))
        ):
            tolerance_reached = True
            break

        # Update tridiagonal matrices, if applicable
        if n_tridiag and k < n_tridiag_iter and update_tridiag:
            alpha_tridiag = alpha.squeeze(-2).narrow(-1, 0, n_tridiag)
            beta_tridiag = beta.squeeze(-2).narrow(-1, 0, n_tridiag)
            torch.eq(alpha_tridiag, 0, out=alpha_tridiag_is_zero)
            alpha_tridiag.masked_fill_(alpha_tridiag_is_zero, 1)
            torch.reciprocal(alpha_tridiag, out=alpha_reciprocal)
            alpha_tridiag.masked_fill_(alpha_tridiag_is_zero, 0)

            if k == 0:
                t_mat[k, k].copy_(alpha_reciprocal)
            else:
                torch.addcmul(alpha_reciprocal, prev_beta, prev_alpha_reciprocal, out=t_mat[k, k])
                torch.mul(prev_beta.sqrt_(), prev_alpha_reciprocal, out=t_mat[k, k - 1])
                t_mat[k - 1, k].copy_(t_mat[k, k - 1])

                if t_mat[k - 1, k].max() < 1e-6:
                    update_tridiag = False

            last_tridiag_iter = k

            prev_alpha_reciprocal.copy_(alpha_reciprocal)
            prev_beta.copy_(beta_tridiag)

    # Un-normalize
    result = result.mul(rhs_norm)

    if not tolerance_reached and n_iter > 0:
        warnings.warn(
            "CG terminated in {} iterations with average residual norm {}"
            " which is larger than the tolerance of {} specified by"
            " linear_operator.settings.cg_tolerance."
            " If performance is affected, consider raising the maximum number of CG iterations by running code in"
            " a linear_operator.settings.max_cg_iterations(value) context.".format(
                k + 1, residual_norm.mean(), tolerance
            ),
            UserWarning,
        )

    if is_vector:
        result = result.squeeze(-1)

    if n_tridiag:
        t_mat = t_mat[: last_tridiag_iter + 1, : last_tridiag_iter + 1]
        return result, t_mat.permute(-1, *range(2, 2 + len(batch_shape)), 0, 1).contiguous()
    else:
        return result