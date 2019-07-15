from math import log, sqrt
import numpy as np
from numpy import exp, fft, array, pi, zeros, linspace
import matplotlib.pyplot as plt
import time
from scipy import interpolate
from takeparam import *

#//------------------------------------------------------------------------------
#Parameters
#//------------------------------------------------------------------------------
param = ['r', 'divid', 'T', 'kappa', 'sigma_V', 'rho',
         'K', 'H', 'V0', 'theta', 'rho', 'L', 'h', 'Nt', 'Nv']
start_val_param = takeparam('data.json', param)
finish_val_param = {}
r = start_val_param.get('r')
# premia 3.045403
divid = start_val_param.get('divid')
# premia 5.127110
# up-and-out call option parameters
T = start_val_param.get('T')

# Heston model parameters
kappa = start_val_param.get('kappa')  # heston parameter, mean reversion
sigma_V = start_val_param.get('sigma_V')  # heston parameter, volatility of variance.
rho = start_val_param.get('rho')  # heston parameter #correlation

K = start_val_param.get('K')  # strike
H = start_val_param.get('H')  # barrier

V0=start_val_param.get('V0')
theta=start_val_param.get('theta') #long run
rho=start_val_param.get('rho')

r_divid = r-divid
# method parameters

L = start_val_param.get('L')
h = start_val_param.get('h')
Nt= start_val_param.get('Nt')
Nv = start_val_param.get('Nv')

# Variance tree, by Zanette, Briani and Apolloni (Z-B-A)

def compute_f(r, omega):
    """Internal function of Z-B-A. For tree computation."""
    return 2*sqrt(r)/omega


def compute_v(R, omega):
    """Internal function of Z-B-A. For tree computation."""
    if R > 0:
        return (R**2) * (omega**2)/4.0
    else:
        return 0.0


def build_volatility_tree(T, v0, kappa, theta, omega, N):
    """Robust tree computation procedure, by Z-B-A."""
    div_by_zero_counter = 0
    f = zeros((N+1, N+1))
    f[0, 0] = compute_f(v0, omega)
    dt = float(T)/float(N)
    sqrt_dt = sqrt(dt)
    V = zeros((N+1, N+1))
    V[0, 0] = compute_v(f[0, 0], omega)
    f[1, 0] = f[0, 0]-sqrt_dt
    f[1, 1] = f[0, 0]+sqrt_dt
    V[1, 0] = compute_v(f[1, 0], omega)
    V[1, 1] = compute_v(f[1, 1], omega)

    for i in range(1, N):
        for j in range(i+1):
            f[i+1, j] = f[i, j] - sqrt_dt
            f[i+1, j+1] = f[i, j] + sqrt_dt
            V[i+1, j] = compute_v(f[i+1, j], omega)
            V[i+1, j+1] = compute_v(f[i+1, j+1], omega)

    f_down = zeros((N+1, N+1), dtype=int)
    f_up = zeros((N+1, N+1), dtype=int)
    pu_f = zeros((N+1, N+1))
    pd_f = zeros((N+1, N+1))
    for i in range(0, N):
        for j in range(i+1):
            # /*Compute mu_f*/
            v_curr = V[i][j]
            mu_r = kappa*(theta-v_curr)
            z = 0
            while V[i, j] + mu_r*dt < V[i+1, j-z] and j-z >= 0:
                z += 1
            f_down[i, j] = -z
            Rd = V[i+1, j-z]  # the next low vertice we can reach
            z = 0
            while V[i, j] + mu_r*dt > V[i+1, j+z] and j+z <= i:
                z += 1
            Ru = V[i+1, j+z]  # the next high vertice we can reach
            f_up[i, j] = z
            if Ru == Rd:
                div_by_zero_counter += 1
            pu_f[i, j] = (V[i, j]+mu_r*dt-Rd)/(Ru-Rd)

            if Ru-1.e-9 > V[i+1, i+1] or j+f_up[i][j] > i+1:
                pu_f[i][j] = 1.0
                f_up[i][j] = i+1-j
                f_down[i][j] = i-j

            if Rd+1.e-9 < V[i+1, 0] or j+f_down[i, j] < 0:
                pu_f[i, j] = 0.0
                f_up[i, j] = 1 - j
                f_down[i, j] = 0 - j
            pd_f[i, j] = 1.0 - pu_f[i][j]
    return [V, pu_f, pd_f, f_up, f_down]

# Option pricing in Heston model

def call_price(approx = 0):
    now = time.perf_counter()
        
    # making volatilily tree
    markov_chain = build_volatility_tree(T, V0, kappa, theta, sigma_V, Nv)
    V = markov_chain[0]
    pu_f = markov_chain[1]
    pd_f = markov_chain[2]
    f_up = markov_chain[3]
    f_down = markov_chain[4]
    
    y_max = L * log(2.0)
    
    M = 2**5  # number of points in price grid, starting value
    border = 2 * y_max / h
    while(M < border):
        M = 2 * M

    y_min = - 0.5*M*h
    y_max = - y_min
    
    Y = linspace(y_min, y_max, num = M, endpoint = False, dtype = np.float64)

    # tree info
    finish_val_param.update(dict.fromkeys(["h"], h))
    finish_val_param.update(dict.fromkeys(["Y[0]"], Y[0]))
    finish_val_param.update(dict.fromkeys(["Y[M-1]"], Y[M-1]))
    finish_val_param.update(dict.fromkeys(["min variance:"], V[Nv, 0]))
    finish_val_param.update(dict.fromkeys(["max variance:"], V[Nv, Nv]))
    
    XI = np.linspace(-pi/h, pi/h, num = M, endpoint = False)
    S = H * np.exp(Y + rho/sigma_V * V[0,0])
    
    m1 = np.ones(M) # n=1, ..., M.
    m1[1::2]=-1 # -1^n, n=1, ..., M.

    def fft_m(x):
        """fft routine, modified to match e^{-ix\\xi}"""
        return m1*fft.fft(x*m1)

    def ifft_m(x):
        """ifft routine, modified to match e^{ix\\xi}"""
        return m1*fft.ifft(m1*x).real
    
    def G(x):
        """payoff_function for a given option type (up-and-out call there)"""
        if (K < x) and (x < H):
            return  x - K
        else:
            return 0

    delta_t = T/(Nv * Nt)
    rho_hat = sqrt(1 - rho**2)
    q = 1.0/delta_t + r
    factor = (q*delta_t)**(-1)

    treshold = 1e-2
    default_drift = r_divid - (rho/sigma_V) * kappa * theta
    discount_factor = exp(default_drift*delta_t)
    
    f_next = zeros((Nv+1, M))
    for k in range(Nv):
        f_next[k] = [G(H*np.exp(y + (rho/sigma_V)*V[Nv, k])) for y in Y]

    f_prev = zeros((Nv, M))
    
    def solve_problem_wh(n, k, n_next, k_ud, f_next_ud):
        # set up variance-dependent parameters for a given step
        sigma_ud = rho_hat * sqrt(V[n_next, k_ud])
        gamma_ud = r_divid - 0.5 * V[n_next, k_ud] - (rho/sigma_V) * kappa * (theta - V[n_next, k_d])
        
        if(approx == 0):
            # beta_plus and beta_minus
            beta_minus_ud = - (gamma_ud + sqrt(gamma_ud**2 + 2*sigma_ud**2 * q))/sigma_ud**2
            beta_plus_ud = - (gamma_ud - sqrt(gamma_ud**2 + 2*sigma_ud**2 * q))/sigma_ud**2

            # factor functions
            phi_plus_ud = np.zeros(M, dtype=complex)
            phi_plus_ud[:M//2+1] = beta_plus_ud/(beta_plus_ud - 1j*XI[:M//2+1])
            phi_plus_ud[M//2+1:] = np.conjugate(phi_plus_ud[M//2-1:0:-1])

            phi_minus_ud = np.zeros(M, dtype=complex)
            phi_minus_ud[:M//2+1] = -beta_minus_ud/(-beta_minus_ud + 1j*XI[:M//2+1])
            phi_minus_ud[M//2+1:] = np.conjugate(phi_minus_ud[M//2-1:0:-1])
        elif(approx == 1):
            def psi(xi):
                return (sigma_ud**2/2) * xi **2 - 1j*gamma_ud*xi

            def mex_minus(om_plus = 10):
                integrand_minus = np.zeros(M, dtype=complex)
                integrand_minus[:M//2+1] = np.log(1 + psi(XI[:M//2+1] + 1j*om_plus)/q) / (XI[:M//2+1] + 1j*om_plus)**2
                integrand_minus[M//2+1:] = np.conjugate(integrand_minus[M//2-1:0:-1])
                Fm = ifft_m(integrand_minus) * 1/h # normalization 1/M is already inside
                Fm = Fm.real
                Fm[:M//2] = 0
                Fm[M//2:] = Fm[M//2:] * np.exp(-Y[M//2:]*om_plus)
                Fm_0 = Fm[M//2]
                Fm[M//2] = 0.5*Fm_0 

                F_hat = fft_m(Fm) * h
                first_term = - 1j * Fm_0 * XI
                second_term = - F_hat * (XI ** 2)
                return np.exp(first_term + second_term)

            def mex_plus(om_minus = -10):
                integrand_plus = np.zeros(M, dtype=complex)
                integrand_plus[:M//2+1] = np.log(1 + psi(XI[:M//2+1] + 1j*om_minus)/q) / (XI[:M//2+1] + 1j*om_minus)**2
                integrand_plus[M//2+1:] = np.conjugate(integrand_plus[M//2-1:0:-1])
                Fp = ifft_m(integrand_plus) * 1/h
                Fp = Fp.real
                Fp[M//2+1:] = 0
                Fp[:M//2+1] = Fp[:M//2+1] * np.exp(-Y[:M//2+1]*om_minus)

                Fp_0 = Fp[M//2]
                Fp[M//2] = 0.5*Fp_0

                Fp_hat = fft_m(Fp) * h

                first_term = 1j * Fp_0 * XI
                second_term = - Fp_hat * (XI ** 2)
                return np.exp(first_term + second_term)
            
            phi_plus_ud = np.zeros(M, dtype=complex)
            phi_plus_ud = mex_plus()

            phi_minus_ud = np.zeros(M, dtype=complex)
            phi_minus_ud = mex_minus()
 
        # factorization calculation
        for i in range(Nt):
            # up-and-out call part
            step1ud = ifft_m(phi_minus_ud * fft_m(f_next_ud))
            step1ud = step1ud * np.where(Y + (rho/sigma_V)*V[n_next,k_ud] < 0, 1, 0)  # suboptimal, but stable search
            f_prev_bar_part = factor * ifft_m(phi_plus_ud * fft_m(step1ud))
            f_prev_bar_part = f_prev_bar_part * np.where(Y + (rho/sigma_V)*V[n,k] < 0, 1, 0)
            f_next_ud = f_prev_bar_part
        return f_next_ud

    for n in range(Nv-1, -1, -1):
        for k in range(n+1):
            k_u = k + f_up[n, k]
            k_d = k + f_down[n, k]
            f_next_u = f_next[k_u]
            f_next_d = f_next[k_d]
            
            if V[n, k] >= treshold:
                # factorization calculation
                f_prev_d = solve_problem_wh(n, k, n+1, k_d, f_next_d)
                f_prev_u = solve_problem_wh(n, k, n+1, k_u, f_next_u)
            elif V[n, k] < treshold:
                f_prev_u = discount_factor * f_next_u
                f_prev_d = discount_factor * f_next_d
            f_prev[k] = f_prev_u * pu_f[n, k] + f_prev_d * pd_f[n, k]
          #  filtering output
        f_next = f_prev
    explicit_values = f_next[0][:]
    then = time.perf_counter()
    
    def interp_price(s0):
        """quadratic interpolation"""
        x = (S[S<s0][-1], S[S>=s0][0], S[S>=s0][1])
        y = (explicit_values[S<s0][-1], explicit_values[S>=s0][0], explicit_values[S>=s0][1])
        g = interpolate.interp1d(x, y, kind='quadratic')
        return float(g(s0))
    
    prices_range = np.array(range(80, 140, 10))
#     S_plot = [price for price in prices_range]
#     prices_WH =  np.array([interp_price(price) for price in prices_range])
#     plt.plot(S, explicit_values)
#     plt.show()
    for price in prices_range:
        finish_val_param.update(dict.fromkeys(['{:3d} '.format(price)], '{:.5f}'.format(interp_price(price))))
    finish_val_param.update(dict.fromkeys(["time"], (str(then-now)[:4] + " sec")))
    
call_price()
result_dict = {"params": start_val_param, "result": finish_val_param}
with open('result.json', 'w')as outfile:
    json.dump(result_dict, outfile)