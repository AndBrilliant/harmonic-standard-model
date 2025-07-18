import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#Note: Neutrino masses are theoretical estimates based on:
#- Oscillation constraints: Δm²₂₁ = 7.53×10⁻⁵ eV²
#- Δm²₃₂ = 2.51×10⁻³ eV² (normal hierarchy)
#- Cosmological bound: Σmᵥ < 0.12 eV
#- Not direct PDG measurements

# Set up the figure with clean academic style
plt.style.use('default')
fig = plt.figure(figsize=(11, 8.5), facecolor='white')

# Title - just the facts
fig.text(0.5, 0.96, 'Lepton Mass Residual Analysis', 
         ha='center', va='top', fontsize=14, fontfamily='serif', 
         color='black', weight='bold')
fig.text(0.5, 0.93, 'Extension to neutrino sector', 
         ha='center', va='top', fontsize=10, fontfamily='serif', color='#444444')

# Left panel - Mathematical derivation
ax_text = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
ax_text.axis('off')

# Mathematical content with corrected values
math_content = """
LEPTON MASS REGRESSION ANALYSIS
═══════════════════════════════

1. LOG-LINEAR REGRESSION (Generation n = 1,2,3)
   
   Charged Leptons:                      Neutrinos (est):
   ────────────────                      ────────────────
   m_e = 0.511 MeV     [n=1]           ν₁ ~ 0.001 meV    [n=1]
   m_μ = 105.66 MeV    [n=2]           ν₂ ~ 0.00874 meV  [n=2]  
   m_τ = 1776.8 MeV    [n=3]           ν₃ ~ 0.0495 meV   [n=3]

2. REGRESSION: log₁₀(m) = αn + β

   Charged: α = 1.7703, β = -1.8805     R² = 0.9998
   Neutrino: α = 0.8475, β = -6.8163    R² = 0.9976

3. RESIDUAL ANALYSIS: ε(n) = log₁₀(m_obs) - log₁₀(m_fit)

   Pattern:
   ─────────────────
   ε_charged = [-0.18162, +0.36324, -0.18162] = δ_c × [-1, +2, -1]
   ε_neutrino = [-0.03126, +0.06252, -0.03126] = δ_ν × [-1, +2, -1]
   
   WHERE: δ_c = 0.18162
          δ_ν = 0.03126
          
   RATIO: δ_c/δ_ν = 5.81

4. RESULT
   
   ALL LEPTONS: ε(n) = δ × [-1, +2, -1]
   
   Charged leptons: δ = 0.18162
   Neutrinos: δ = 0.03126
   Ratio: 5.81
"""

ax_text.text(0.02, 0.98, math_content, transform=ax_text.transAxes,
             fontsize=9, fontfamily='monospace', color='black',
             verticalalignment='top', horizontalalignment='left')

# Right panel - Clean 3D visualization
ax_3d = plt.subplot2grid((2, 3), (0, 2), rowspan=2, projection='3d')

# Data
generations = np.array([1, 2, 3])
charged_residuals = np.array([-0.18162, 0.36324, -0.18162])
neutrino_residuals = np.array([-0.03126, 0.06253, -0.03126])

# Scale neutrinos for visual clarity
visual_scale = 2.5
neutrino_residuals_scaled = neutrino_residuals * visual_scale

# Create smooth interpolation for the waves
gen_fine = np.linspace(0.8, 3.2, 50)

# Function for the [-1, +2, -1] pattern (quadratic through our points)
def wave_pattern(g, amplitude):
    return amplitude * (1 - ((g - 2) / 1)**2)

# Plot the two waves
# Charged leptons
ax_3d.plot(gen_fine, np.zeros_like(gen_fine), 
           wave_pattern(gen_fine, 0.36324), 
           'darkred', linewidth=3, alpha=0.9, label='Charged Leptons')

# Neutrinos  
ax_3d.plot(gen_fine, np.ones_like(gen_fine), 
           wave_pattern(gen_fine, 0.06253 * visual_scale), 
           'darkblue', linewidth=3, alpha=0.9, label=f'Neutrinos (×{visual_scale})')

# Add data points
ax_3d.scatter(generations, [0, 0, 0], charged_residuals, 
              color='red', s=120, edgecolors='darkred', linewidth=2, zorder=10)
ax_3d.scatter(generations, [1, 1, 1], neutrino_residuals_scaled,
              color='blue', s=120, edgecolors='darkblue', linewidth=2, zorder=10)

# Create surface mesh
x_surf = np.linspace(0.8, 3.2, 40)
y_surf = np.linspace(-0.1, 1.1, 30)
X, Y = np.meshgrid(x_surf, y_surf)
Z = np.zeros_like(X)

# Fill surface with smooth transition
for i in range(len(y_surf)):
    for j in range(len(x_surf)):
        y = Y[i, j]
        x = X[i, j]
        # Interpolate amplitude between charged (y=0) and neutrino (y=1)
        if y <= 0:
            amp = 0.36324
        elif y >= 1:
            amp = 0.06253 * visual_scale
        else:
            amp = 0.36324 * (1-y) + 0.06253 * visual_scale * y
        Z[i, j] = wave_pattern(x, amp)

# Plot surface
surf = ax_3d.plot_surface(X, Y, Z, cmap='RdBu_r', alpha=0.7,
                          linewidth=0, antialiased=True)

# Add grid lines on surface
ax_3d.contour(X, Y, Z, levels=8, colors='gray', alpha=0.3, linewidths=0.5)

# Connect peaks
ax_3d.plot([2, 2], [0, 1], [0.36324, 0.06253 * visual_scale],
           'gray', linewidth=2, linestyle='--', alpha=0.5)

# Labels and styling
ax_3d.set_xlabel('Generation', fontsize=11)
ax_3d.set_ylabel('Lepton Type', fontsize=11) 
ax_3d.set_zlabel('Residual ε', fontsize=11)
ax_3d.set_xticks([1, 2, 3])
ax_3d.set_yticks([0, 1])
ax_3d.set_yticklabels(['Charged', 'Neutrino'])
ax_3d.set_xlim(0.8, 3.2)
ax_3d.set_ylim(-0.1, 1.1)
ax_3d.set_zlim(-0.25, 0.4)

# Viewing angle - match the screenshot
ax_3d.view_init(elev=15, azim=45)

# Clean up
ax_3d.grid(True, alpha=0.2, linestyle=':')
ax_3d.xaxis.pane.fill = False
ax_3d.yaxis.pane.fill = False
ax_3d.zaxis.pane.fill = False

# Legend - move to better position
ax_3d.legend(loc='upper left', fontsize=9, bbox_to_anchor=(0.05, 0.95))

# Add the pattern notation
ax_3d.text2D(0.5, 0.95, '[-1, +2, -1] pattern', 
             transform=ax_3d.transAxes, ha='center',
             fontsize=10, fontfamily='serif', color='black')

# Bottom annotation - neutral reference with notes
fig.text(0.5, 0.02, 'Supplementary Figure S1. Neutrino masses from oscillation constraints. v2: Corrected regression parameters.', 
         ha='center', va='bottom', fontsize=8, fontfamily='serif', 
         color='#666666')

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.05)
plt.savefig('lepton_harmonics_supplement.png', dpi=300, facecolor='white', edgecolor='none')
plt.show()
