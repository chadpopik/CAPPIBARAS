# Cross-spectra and Average Profile Predictions for Inference of Baryonic Astrophysics off high-Resolution Astronomical Surveys (CAPPIBARAS)


First run ModuleCheck.ipynb, which will explain and run every one of the individual model modules and ensure they work, then ForwardModelCheck.ipynb, which compiles the entire pipeline from profile to signal. 

To come:
- SHMR that accounts for centrals vs satellites
- DESI SMF from catalog

# Profiles

One-halo stuff:

$P_{GNFW}(x; z, m) = P_\text{scale}(z, m) \ P_0 \Big[\frac{x}{x_c}\Big]^\gamma\bigg[1+\Big[\frac{x}{x_c}\Big]^\alpha \bigg]^\beta $ ,

$A^{\{P_0, x_c, \gamma, \alpha, \beta \}}(z, m)=A_0^{\{x_c, P_0, \gamma, \alpha, \beta \}} \Big[\frac{m}{10^{14} M_\odot}\Big]^{\alpha_m^{\{P_0, x_c, \gamma, \alpha, \beta \}}}[1+z]^{\alpha_z^{\{P_0, x_c,\gamma, \alpha, \beta \}}}$ ,

$P_\text{th}(x; z, m) = P_\text{th, 200c}(z, m) \ P_0(z, m) \Big[\frac{x}{x_c(z, m)}\Big]^\gamma\bigg[1+\Big[\frac{x}{x_c(z, m)}\Big]^\alpha \bigg]^{\beta(z, m)} $ ,

$P_\text{th, 200c} = \frac{\Omega_b}{\Omega_m} \frac{G m 200 \rho_c(z)}{2R_{200c}(m, z)}$ ,

$\rho_\text{gas}(x; z, m) = \rho_\text{gas, c}(z) \ P_0(z, m) \Big[\frac{x}{x_c}\Big]^\gamma\bigg[1+\Big[\frac{x}{x_c}\Big]^{\alpha(z, m)} \bigg]^\frac{\beta(z, m)-\gamma}{\alpha(z, m)} $ ,

$\rho_\text{gas, c}(z) = \frac{\Omega_b}{\Omega_m} \rho_c(z)$ ,

# Two-halo

$P(r; z, m) = P^{1h}(r; z, m) + A_{2h} P^{2h}_\text{lin}(r; z, m)$

$P(k; z, m) = \mathcal{F} \big( P(r;z, m) \big)$
$P^{2h}_\text{lin}(k; z, m) = P_\text{lin}(k; z) b_h(z, m) \int_{m_\text{min}}^{m_\text{min}} \frac{dn}{dm}(z, m') b_h(z, m') P(k;z, m')dm'$
$P^{2h}_\text{lin}(r; z, m) = \mathcal{F}^{-1} \big( W(k) P^{2h}(k;z, m) \big)$

# Weighting


# Projection

Profiles are projected onto the sky by integrating over the line of sight:
$\bar{P}^{2D}(R; z) = \int_{-\infty}^{\infty} \bar{P}^{3D}(r=\sqrt{l^2+d_A^2(z) R^2}) dl$ ,
where $\bar{P}^{3D}(r)$ is the mas/redshift-averaged three-dimensional radial profile as a function of radial distance $r$ in Mpc, $l$ is the line of sight radial distance in Mpc, $R$ is the angular size distance on the sky in radians, and $d_A(z)$ is the angular diameter distance at redshift $z$. We make the following simplifications:
- Although the redshift $z$ changes as we change the line of sight $l$, the changes from $l^2$ term dwarfs corresponding changes in $d_A^2(z+dz(l)) R^2$, and therefore we can keep a constant $z$ value over the line of sight integration.
- The profile averaged over redshift $\langle \bar{P}^{2D}(R; z) \rangle_z$ is equivalent to the profile at the median redshift $\bar{P}^{2D}(R; \bar{z})$.
- Assuming the profile is symmetric along the line of sight, our integral over the range ($-\infty, \infty$) is equal to twice the integral over the range ($0, \infty$).
- The profile converges at some minimum and maximum line of sight $l_\text{min}, l_\text{min}$ which we can use as bounds instead of $0, \infty$.
  
This simplifies our profile to:
$\bar{P}^{2D}(R) = 2 \int_{l_\text{min}}^{l_\text{max}} \bar{P}^{3D}(r=\sqrt{l^2+d_A^2(\bar{z}) R^2}) dl$ ,
where tests have shown sufficient values of $l_\text{min}=10^{-3}, l_\text{max}=10$ Mpc.

The projected profile is then transform using a Radial Hankel Transform (RHT), after interpolating to proper $R$ values:
$\bar{P}^{2D}(\ell_\text{RHT}) = \mathcal{F}_\text{RHT} \big(\bar{P}^{2D}(R \rightarrow R_\text{RHT} = \frac{1}{\ell_\text{RHT}}) \big)$ ,
where the $\ell_\text{RHT}$ values go from $\ell_\text{RHT, min} \approx \frac{1}{r_\text{max}}, \ell_\text{RHT, max} \approx \frac{1}{r_\text{min}}$, where $r$ is the 3D radial distance given in the equations above.

This is then convolved with the beam and response (which is 1 for the kSZ), both of which have been interpolated to the proper $\ell$ values:
$\bar{P}^{2D}_\text{beam}(\ell_\text{RHT}) = \bar{P}^{2D}(\ell_\text{RHT}) B(\ell \rightarrow \ell_\text{RHT}) R(\ell \rightarrow \ell_\text{RHT})$ .

It is then converted back:
$\bar{P}^{2D}_\text{beam}(R_\text{RHT}) = \mathcal{F}^{-1}_\text{RHT} \big( \bar{P}^{2D}_\text{beam}(\ell_\text{RHT}) \big) $ ,

The signal is summed over values
$\bar{\Sigma}^P(R) = 2 \pi \Delta R \sum_{R'=\Delta R}^{R'=R} \big( R' \bar{P}^{2D}_\text{beam}(R_\text{RHT} \rightarrow R') \big)$ ,

And the CAP is performed
$\bar{\Sigma}_\text{CAP}^P(R) = \bar{\Sigma}^P(R) - \big[\bar{\Sigma}^P(f_\text{disc}R) - \bar{\Sigma}^P(R)\big]$ ,

$\Delta T_\text{tSZ} (R) = \frac{\sigma_T}{m_e c^2} \ \frac{2+2X_H}{3+5X_H}  \ \bar{\Sigma}_\text{CAP}^{P_\text{th}}(R)$

$\Delta T_\text{kSZ} (R) = T_\text{CMB} \ \frac{v_\text{RMS} \sigma_T}{m_p} \ \frac{1+X_H}{2} \ \bar{\Sigma}_\text{CAP}^{\rho_\text{gas}}(R)$



