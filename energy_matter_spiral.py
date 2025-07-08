import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Set up full page figure
plt.style.use('default')  # White background
fig = plt.figure(figsize=(16, 16), facecolor='white')
ax = fig.add_subplot(111, projection='polar', facecolor='white')

# Title with the key insight
fig.suptitle('Universal Energy-Matter Spiral: From γ (-3²) to Matter (3×2)', 
             fontsize=24, fontweight='bold', y=0.98)

# Particle data - ENERGY BASED (in eV)
particles = [
    # Photons (various energies)
    ('Radio', 1e-9, 'photon'),
    ('μWave', 1e-6, 'photon'),
    ('IR', 1e-3, 'photon'),
    ('Visible', 2, 'photon'),
    ('UV', 1e2, 'photon'),
    ('X-ray', 1e4, 'photon'),
    ('γ-ray', 1e6, 'photon'),  # Start of gamma range
    ('γ-MeV', 1e7, 'photon'),   # MeV gammas
    ('γ-GeV', 1e9, 'photon'),   # GeV gammas
    ('γ-TeV', 1e12, 'photon'),  # TeV gammas (cosmic rays)
    
    # Neutrinos (rest energy)
    ('νe', 1e-5, 'neutrino'),  # ~10 μeV estimate
    ('νμ', 1.3e-2, 'neutrino'),  # ~13 meV estimate
    ('ντ', 5e-2, 'neutrino'),  # ~50 meV estimate
    
    # Leptons (rest energy)
    ('e', 5.11e5, 'lepton'),
    ('μ', 1.057e8, 'lepton'),
    ('τ', 1.777e9, 'lepton'),
    
    # Quarks (rest energy)
    ('u', 2.16e6, 'quark'),
    ('d', 4.67e6, 'quark'),
    ('s', 9.34e7, 'quark'),
    ('c', 1.27e9, 'quark'),
    ('b', 4.18e9, 'quark'),
    ('t', 1.728e11, 'quark'),
    
    # Bosons
    ('W', 8.04e10, 'boson'),
    ('Z', 9.12e10, 'boson'),
    ('H', 1.25e11, 'boson'),
]

# Sort by energy
particles.sort(key=lambda x: x[1])

# Color scheme
colors = {
    'photon': '#ffff00',
    'neutrino': '#ff00ff',
    'lepton': '#0080ff',
    'quark': '#ff0000',
    'boson': '#00ff00'
}

# Calculate spiral coordinates
tightness = 0.5  # As requested
energies = [p[1] for p in particles]
log_energies = np.log10(energies)

# Normalize to use full viewing area
e_min, e_max = min(log_energies), max(log_energies)
e_range = e_max - e_min

# Create spiral
theta_vals = []
r_vals = []
particle_theta = []
particle_r = []

for i, (name, energy, ptype) in enumerate(particles):
    # Spiral equation: r = a * e^(b*theta)
    # We want uniform spacing along spiral
    theta = i * tightness
    r = (np.log10(energy) - e_min) / e_range * 0.9 + 0.05  # Normalize to 0.05-0.95
    
    particle_theta.append(theta)
    particle_r.append(r)
    
    # Plot particle
    ax.scatter(theta, r, s=200, c=colors[ptype], edgecolor='black', 
               linewidth=2, zorder=10, alpha=0.9)
    
    # Add label with smart positioning
    # Offset labels radially to avoid overlaps
    if i > 0 and abs(particle_theta[i] - particle_theta[i-1]) < 0.3:
        # Close to previous particle, offset more
        label_offset = 0.06 + (i % 3) * 0.02
    elif ptype == 'photon':
        label_offset = 0.02
    else:
        label_offset = 0.03
        
    # Rotate text for better fit on spiral
    text_angle = np.degrees(theta) - 90 if np.degrees(theta) > 180 else np.degrees(theta) + 90
    
    ax.text(theta, r + label_offset, name, ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='black',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8),
            rotation=text_angle if len(particles) > 15 else 0)

# Draw connecting spiral line
# Interpolate for smooth curve
from scipy.interpolate import interp1d
theta_smooth = np.linspace(min(particle_theta), max(particle_theta), 1000)
f = interp1d(particle_theta, particle_r, kind='cubic')
r_smooth = f(theta_smooth)

ax.plot(theta_smooth, r_smooth, 'k-', linewidth=2, alpha=0.5)

# Highlight the critical transition at 10^6 eV (pair production threshold)
matter_threshold_idx = next(i for i, p in enumerate(particles) if p[0] == 'γ-ray')
if matter_threshold_idx >= 0:
    theta_threshold = particle_theta[matter_threshold_idx]
    r_threshold = particle_r[matter_threshold_idx]
    
    # Draw radial line at threshold
    ax.plot([theta_threshold, theta_threshold], [0, r_threshold], 'r--', 
            linewidth=3, alpha=0.7, label='E = 2mc² threshold')
    ax.text(theta_threshold, r_threshold/2, 'MATTER\nTHRESHOLD', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='red', bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

# Add the key insight annotation
ax.text(0.5, -0.15, 
        'Photon range: 10⁻⁹ to 10⁶ eV\n' + 
        '-9 = -3² (radio) → +6 = 3×2 (matter threshold)\n' +
        'The 3/2 ratio defines the energy-matter boundary!',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=14, bbox=dict(boxstyle="round,pad=0.5", 
        facecolor='yellow', alpha=0.3))

# Add legend - moved to upper right to avoid blocking
legend_elements = [
    mpatches.Patch(color=colors['photon'], label='Photons (pure energy)'),
    mpatches.Patch(color=colors['neutrino'], label='Neutrinos'),
    mpatches.Patch(color=colors['lepton'], label='Charged Leptons'),
    mpatches.Patch(color=colors['quark'], label='Quarks'),
    mpatches.Patch(color=colors['boson'], label='Bosons'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
          framealpha=0.9, edgecolor='black')

# Customize plot
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

# Remove radial labels but keep grid
ax.set_yticklabels([])
ax.set_xticklabels([])

# Add radial labels manually
radial_positions = [0.2, 0.4, 0.6, 0.8]
radial_labels = ['-6', '0', '+6', '+12']  # log10(eV)
for pos, label in zip(radial_positions, radial_labels):
    ax.text(0, pos, f'10^{label} eV', ha='center', va='center',
            fontsize=10, color='black', bbox=dict(boxstyle="round,pad=0.2",
            facecolor='lightgray', alpha=0.8))

# Add spiral equation
ax.text(0.98, 0.02, 'r = log₁₀(E/eV)\nθ = particle index × 0.5',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=12, bbox=dict(boxstyle="round,pad=0.3",
        facecolor='lightgray', alpha=0.8))

# Save the figure as both PNG and PDF
plt.savefig('energy_matter_spiral.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('energy_matter_spiral.pdf', format='pdf', bbox_inches='tight', facecolor='white')
print("Saved as: energy_matter_spiral.png and energy_matter_spiral.pdf")

plt.tight_layout()
plt.show()

# Additional analysis plot showing the energy progression
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig2.suptitle('Energy Progression Analysis', fontsize=18, fontweight='bold')

# Top: Log energy vs particle index
particle_names = [p[0] for p in particles]
log_e = [np.log10(p[1]) for p in particles]
particle_types = [p[2] for p in particles]

for i, (name, log_energy, ptype) in enumerate(zip(particle_names, log_e, particle_types)):
    ax1.scatter(i, log_energy, s=150, c=colors[ptype], edgecolor='white', linewidth=2)
    if i % 2 == 0:  # Label every other point to avoid crowding
        ax1.text(i, log_energy + 0.5, name, ha='center', va='bottom', fontsize=9, rotation=45)

ax1.plot(range(len(particles)), log_e, 'w--', alpha=0.5, linewidth=1)
ax1.axhline(y=6, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax1.text(len(particles)/2, 6.5, 'Matter Creation Threshold (10⁶ eV)', 
         ha='center', va='bottom', fontsize=12, color='red', fontweight='bold')

ax1.set_xlabel('Particle Index (sorted by energy)', fontsize=14)
ax1.set_ylabel('log₁₀(Energy/eV)', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_title('Energy Spectrum: From Photons to Planck Scale', fontsize=16)

# Bottom: Energy differences (to show scaling)
energy_ratios = [log_e[i+1] - log_e[i] for i in range(len(log_e)-1)]
ax2.bar(range(len(energy_ratios)), energy_ratios, color='cyan', edgecolor='white', alpha=0.7)
ax2.set_xlabel('Transition Index', fontsize=14)
ax2.set_ylabel('Δ(log₁₀ E)', fontsize=14)
ax2.set_title('Energy Spacing: Evidence of Harmonic Structure?', fontsize=16)
ax2.grid(True, alpha=0.3, axis='y')

# Look for patterns in ratios
avg_photon_spacing = np.mean(energy_ratios[:6])
avg_matter_spacing = np.mean(energy_ratios[7:])
ax2.axhline(y=avg_photon_spacing, color='yellow', linestyle='--', alpha=0.7, 
            label=f'Avg photon spacing: {avg_photon_spacing:.2f}')
ax2.axhline(y=avg_matter_spacing, color='red', linestyle='--', alpha=0.7,
            label=f'Avg matter spacing: {avg_matter_spacing:.2f}')
ax2.legend()

plt.tight_layout()
plt.show()

print("\nKEY INSIGHTS:")
print("="*50)
print(f"1. Photon energy range: 10^-9 to 10^6 eV")
print(f"   -9 = -3² and +6 = 3×2")
print(f"2. Total range: {e_max - e_min:.1f} orders of magnitude")
print(f"3. Matter threshold at exactly 3×2 in exponent")
print(f"4. The 3/2 ratio appears at the energy-matter boundary!")
