import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
import cmath
from scipy.special import factorial

#compare with show_symmetries.py, lines 64–76
def show_heatmap(nc_scores, vmin=None, vmax=None, title='Quasisymmetry discovery'):
    """Visualize the non-commutativity heatmap"""
    norb = np.shape(nc_scores)[0]
    plt.figure(figsize=(8, 6))
    plt.imshow(nc_scores, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='viridis')
    plt.colorbar(label='Non-commutativity norm (log scale)')
    plt.title(title)

    # Add text annotations for exact values
    for i in range(norb):
        for j in range(norb):
            if i <= j:  # Only show upper triangle
                plt.text(j, i, f'{nc_scores[i,j]:.2e}',
                        ha='center', va='center', color='white' if nc_scores[i,j] > 1e-2 else 'black')

    plt.tight_layout()
    plt.show()

def integers_to_phases_polynomial(N):
    """
    Computes the coefficients [a_0, a_1, ..., a_N] for the polynomial
    P(x) = a_0 + a_1*x + a_2*x^2 + ... + a_N*x^N 
    that maps integers n=0..N to exp(i * n * 2 * pi / (N+1) ) (unit semicircle).
    """
    omega = np.exp(1j * 2 * np.pi / (N+1))
    final_poly = [0j] * (N + 1)
    
    # FIX: Initialize as 1.0 (or 1 + 0j), not the imaginary unit (1j)
    falling_fact = [1 + 0j] 
    
    for k in range(N + 1):
        scalar = ((omega - 1)**k) / factorial(k)
        for i, c in enumerate(falling_fact):
            final_poly[i] += c * scalar
        
        if k < N:
            next_fact = [0j] * (len(falling_fact) + 1)
            for i, c in enumerate(falling_fact):
                next_fact[i] -= c * k       
                next_fact[i+1] += c         
            falling_fact = next_fact
            
    return final_poly
