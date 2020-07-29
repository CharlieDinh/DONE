import torch
import numpy as np
from torch import nn

def l2_norm(model):
    weights = model
    if isinstance(model, nn.Module):
        weights = list(model.parameters())
    norm = 0.
    for weight in weights:
        norm += weight.data.norm() ** 2
    # assert norm >= 0.
    return np.sqrt(norm)


def dot_product(weights1, weights2):
    if isinstance(weights1, nn.Module):
        weights1 = list(weights1.parameters())
    if isinstance(weights2, nn.Module):
        weights2 = list(weights2.parameters())
    dot = 0.
    for weight1, weight2 in zip(weights1, weights2):
        dot += weight1.data.mul(weight2.data).sum()
    return dot


def add_params(to, fr, ratio=1., in_place=True):
    result = []
    for t, f in zip(to, fr):
        if in_place:
            t.data.add_(ratio * f)
        else:
            result.append(t.data + ratio * f)
    if not in_place:
        return result


def conjugate_gradient(loss, model, iters=10):
    if isinstance(model, nn.Module):
        model = list(model.parameters())
    grads = torch.autograd.grad(loss, model, create_graph=True, retain_graph=True)

    p = [torch.zeros_like(grad) for grad in grads]
    g = grads
    d = [-grad.clone().detach() for grad in grads]

    for i in range(iters):
        Hv = torch.autograd.grad(grads, model, d,
                                 only_inputs=True, create_graph=True, retain_graph=True)
        old_g_norm = l2_norm(g)
        alpha = (old_g_norm ** 2) / (dot_product(d, Hv))
        add_params(p, d, alpha)
        add_params(g, Hv, alpha)
        beta = (l2_norm(g) ** 2) / (old_g_norm ** 2)
        d = add_params([- data.data for data in g], d, beta, in_place=False)

    return p

def armijo_search(model: nn.Module, direction, X, y, loss_func, beta=0.1, initial_lr=1., tau=0.707):
    """
    direction: the descent direction in optimization. e.g. for gradient descent, direction
               is the nagative of the gradient evaluated at the current weights.
    loss_func: a callable loss function
    """
    initial_f = loss_func(model(X), y)
    grads = torch.autograd.grad(initial_f, list(model.parameters()), create_graph=True,
                                retain_graph=True, only_inputs=True)
    dot = dot_product(grads, direction)
    lr = initial_lr

    add_params(model.parameters(), direction, lr)
    while loss_func(model(X), y).data > initial_f + lr * beta * dot:
        add_params(model.parameters(), direction, (tau - 1) * lr)
        lr = tau * lr

    # Recover the old weights
    add_params(model.parameters(), direction, -tau)
    return lr

def hessian_vec_prod(loss, params, dt):
    """
    loss is dependent on params
    dt is the vector to multiply with the Hessian
    """
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    hv = torch.autograd.grad(grads, params, dt, only_inputs=True, create_graph=True, retain_graph=True)
    return hv


# From PyHessian
def top_eigenvalue(loss, model, max_iter=100, tol=1e-5):
    """
    Calculate the eigenvalue with highest absolute value in the Hessian
    Returns: maximum eigenvalue and corresponding eigenvector
    """
    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    v = [torch.rand(param.size()) for param in params]
    v = normalize(v)
    eigval = None
    for i in range(1, max_iter+1):
        model.zero_grad()
        Hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        tmp_eigval = sum([torch.sum(Hvi * vi) for Hvi, vi in zip(Hv, v)])
        v = normalize(Hv)
        if eigval is None:
            eigval = tmp_eigval
        else:
            if abs(tmp_eigval - eigval) / (abs(eigval) + 1e-6) < tol:
                break
            else:
                eigval = tmp_eigval
    return eigval, v


def max_min_eigvals(loss, model):
    """
    Find the maximum and minimum eigenvalues
    loss: must be convex (Hessian is positive semi-definite)
    """
    model.zero_grad()
    loss = loss_fn(model(X_train), y_train)
    for param in model.parameters():
        loss += (-0.0001/2) * torch.sum(param * param)

    max_eigval, max_eigvec = top_eigenvalue(loss, model)

    model.zero_grad()
    # loss = loss_fn(model(X_train), y_train)
    for param in model.parameters():
        loss -= (max_eigval/2) * torch.sum(param * param)
    min_eigval, min_eigvec = top_eigenvalue(loss, model)
    min_eigval += max_eigval
    assert(max_eigval > min_eigval)
    return max_eigval, min_eigval


def hessian(grads, model):
    """
    Order: W first, then b
    """
    size = grads[0].shape[1] + 1  # +1 is for the bias
    hess = torch.zeros(size, size)#.to(device)
    i = 0
    for grad in grads[0].flatten():
        W_second = torch.autograd.grad(grad, model.parameters(), retain_graph=True)
        hess[i] = torch.cat([W_second[0], W_second[1].view(1, 1)], 1)
        i += 1
    b_second = torch.autograd.grad(grads[1], model.parameters(), retain_graph=True)
    hess[i] = torch.cat([b_second[0], b_second[1].view(1, 1)], 1)
    i += 1
    return hess

def newton_direction(grads, model):
    """
    Order: W first, then b
    """
    grads_as_vector = torch.cat([-grads[0].data.clone().detach(), -grads[1].clone().view(1, 1)], 1)
    hess = hessian(grads, model)#.to(device)
    inverse_hess = torch.inverse(hess)#.to(device)
    direction = torch.matmul(grads_as_vector, inverse_hess)
    weights_direction = direction[0, 0:-1].view(grads[0].shape)#.to(device)
    bias_direction = direction[0, -1].view(grads[1].shape)#.to(device)
    return [weights_direction, bias_direction]

def richardson_iter(grads, model, tol=1e-6):
    def hessian_max_min_eig(grads, model):
        hess = hessian(grads, model)#.to(device)
        eigvalues = torch.eig(hess)[0]
        real_eigvalues = [real for real, img in eigvalues if torch.abs(img) < 1e-6]
        return max(real_eigvalues), min(real_eigvalues)
    newton_direction = [torch.zeros_like(param.data) for param in model.parameters()]
    eigmax, eigmin = hessian_max_min_eig(grads, model)
    alpha = 2 / (eigmax + eigmin)
    num_iters = int(torch.ceil(((eigmax + eigmin) / (2 * eigmin)) * np.log(1/tol)))
    # print("alpha:", alpha)
    for it in range(1, num_iters + 1):
        model.zero_grad()
        Hv = torch.autograd.grad(grads,
                                 model.parameters(),
                                 grad_outputs=newton_direction,
                                 create_graph=False,
                                 retain_graph=True)
        for dt, hv, grad in zip(newton_direction, Hv, grads):
            dt.data = dt.data - alpha * hv - alpha * grad.data
    return newton_direction

def descent_direction(loss, model, mode="sgd"):
    """
    mode: eigther "sgd", "newton", or "richardson"
    """
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    mode = mode.lower()
    if mode not in ["sgd", "newton", "richardson"]:
        raise Exception("Undefined mode of direction. Must be either sgd or newton")
    if mode == "sgd":
        # return [-grad.data.to(device) for grad in grads]
        return [-grad.data for grad in grads]
    elif mode == "newton":
        return newton_direction(grads, model)
    else:
        return richardson_iter(grads, model, tol=1e-6)