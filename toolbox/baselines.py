import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.optimize import quadratic_assignment

def baseline(loader):
    all_b = []
    all_u = []
    all_acc = []
    all_p = []
    for batch in loader:
        (data1, data2, target) = batch
        g1 = data1['input'][:,0,:,:].cpu().detach().numpy()
        g2 = data2['input'][:,0,:,:].cpu().detach().numpy()
        planted = target.cpu().detach().numpy()
        n = len(planted[0])
        bs = planted.shape[0]
        for i in range(bs):
            all_b.append((g1[i]*g2[i]).sum()/2)
            if planted[i].ndim == 2:
                pl = np.argmax(planted[i],1)
            all_p.append((g1[i]*g2[i][pl,:][:, pl]).sum()/2)
            Pp = perm2mat(pl)
            res_qap = quadratic_assignment(g1[i],-g2[i],method='faq',options={'P0':Pp})
            all_u.append((g1[i]*g2[i][res_qap['col_ind'],:][:, res_qap['col_ind']]).sum()/2)
            all_acc.append(np.sum(pl==res_qap['col_ind'])/n)
    return np.array(all_b), np.array(all_u), np.array(all_acc), np.array(all_p)

# inspired from the matlab code
# https://github.com/jovo/FastApproximateQAP/blob/master/code/SGM/relaxed_normAPPB_FW_seeds.m



def perm2mat(p):
    n = np.max(p.shape)
    P = np.zeros((n,n))
    for i in range(n):
        P[i, p[i]] = 1
    return P

def fro_norm(P, A, B):
    return np.linalg.norm(np.dot(A, P) - np.dot(P, B), ord='fro') ** 2

def indef_rel(P, A, B):
    return -np.trace(np.transpose(A@P)@(P@B))

def relaxed_normAPPB_FW_seeds(A, B, max_iter=1000, seeds=0, verbose=False):
    AtA = np.dot(A.T, A)
    BBt = np.dot(B, B.T)
    p = A.shape[0]
    
    def f1(P):
        return np.linalg.norm(np.dot(A, P) - np.dot(P, B), ord='fro') ** 2
    
    tol = 5e-2
    tol2 = 1e-4
    
    P = np.ones((p, p)) / (p - seeds)
    P[:seeds, :seeds] = np.eye(seeds)
    
    f = f1(P)
    var = 1
    s = 0

    while not (np.abs(f) < tol) and (var > tol2) and (s<max_iter):
        fold = f
        
        grad = 2*(np.dot(AtA, P) - np.dot(np.dot(A.T, P), B) - np.dot(np.dot(A, P), B.T) + np.dot(P, BBt))
        
        grad[:seeds, :] = 0
        grad[:, :seeds] = 0
        
        #G = np.round(grad)
        
        row_ind, col_ind = linear_sum_assignment(grad[seeds:, seeds:])
        
        Ps = perm2mat(col_ind)
        Ps[:seeds, :seeds] = np.eye(seeds) 
        
        C = np.dot(A, P - Ps) + np.dot(Ps - P, B)
        D = np.dot(A, Ps) - np.dot(Ps, B)
        
        aq = np.trace(np.dot(C, C.T))
        bq = np.trace(np.dot(C, D.T) + np.dot(D, C.T))
        aopt = -bq / (2 * aq)
        
        Ps4 = aopt * P + (1 - aopt) * Ps
        
        f = f1(Ps4)
        P = Ps4
        
        var = np.abs(f - fold)
        s += 1
    
    _, col_ind = linear_sum_assignment(-P.T)
    
    if verbose:
        return P.T, col_ind, s
    else:
        return P.T, col_ind, None

def all_qap_scipy(loader, max_iter = 1000, maxiter_faq=30, seeds = 0, verbose=False):
    all_qap = []
    all_d = []
    all_planted = []
    all_acc = []
    all_accd = []
    all_fd = []
    all_fproj = []
    all_fqap = []
    all_fplanted = []
    all_conv_nit = []
    all_nit =[]
    for batch in loader:
        (data1, data2, target) = batch
        g1 = data1['input'][:,0,:,:].cpu().detach().numpy()
        g2 = data2['input'][:,0,:,:].cpu().detach().numpy()
        planted = target.cpu().detach().numpy()
        
        n = len(planted[0])
        bs = planted.shape[0]
        
        for i in range(bs):
            if planted[i].ndim == 2:
                pl = np.argmax(planted[i],0)
            P, col, s = relaxed_normAPPB_FW_seeds(g1[i],g2[i], max_iter=max_iter, seeds=seeds, verbose=verbose)
            if verbose:
                all_conv_nit.append(s)
            Pp = perm2mat(col)
            all_fd.append(fro_norm(P.T, g1[i], g2[i]))
            all_fproj.append(fro_norm(Pp.T, g1[i],g2[i]))
            res_qap = quadratic_assignment(g2[i],-g1[i],method='faq',options={"P0": P, "maxiter": maxiter_faq})
            P_qap = perm2mat(res_qap['col_ind'])
            all_fqap.append(fro_norm(P_qap.T, g1[i], g2[i]))
            P_planted = perm2mat(pl)
            all_fplanted.append(fro_norm(P_planted.T, g1[i],g2[i]))
            
            all_planted.append((g2[i]*g1[i][pl,:][:, pl]).sum()/2)
            all_qap.append((g2[i]*g1[i][res_qap['col_ind'],:][:, res_qap['col_ind']]).sum()/2)
            all_d.append((g2[i]*g1[i][col,:][:, col]).sum()/2)
            all_acc.append(np.sum(pl==res_qap['col_ind'])/n)
            all_accd.append(np.sum(pl==col)/n)
            if verbose:
                all_nit.append(res_qap['nit'])
    if verbose:
        return np.array(all_planted), np.array(all_qap), np.array(all_d), np.array(all_acc), np.array(all_accd), np.array(all_fd), np.array(all_fproj), np.array(all_fqap), np.array(all_fplanted), np.array(all_conv_nit), np.array(all_nit)
    else:
        return np.array(all_planted), np.array(all_qap), np.array(all_d), np.array(all_acc), np.array(all_accd), np.array(all_fd), np.array(all_fproj), np.array(all_fqap), np.array(all_fplanted)