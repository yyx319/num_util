'''
Function to write cloudy input files
'''




# Write to sim.in file to prepare Cloudy run
with open(input_file, "w") as f:
    f.write("title \"this is the input stream for a planetary nebula\"\n")
    f.write("# shape of incident radiation field as ordered pairs giving the energy and intensity\n")
    f.write(f"interpolate {interpolate_pairs[:line_lengths[0]]}\n")
    for line, length in enumerate(line_lengths[1:]):
        f.write(f"continue {interpolate_pairs[np.sum(line_lengths[:line+1]):np.sum(line_lengths[:line+1])+length]}\n")
    f.write("#\n")
    f.write("# log of starting radius in cm\n")
    f.write(f"radius {np.log10(Ri_cm):.2f}\n")
    f.write("#\n")
    f.write("# log of hydrogen density - cm^-3 -\n")
    f.write(f"hden {np.log10(n_H):.2f}\n")
    f.write("#\n")
    f.write("# this is a sphere with large covering factor, sphere is expanding (photons do not interact with line-absorbing gas on far side)\n")
    f.write("sphere\n")
    f.write("covering factor 1.0\n")
    f.write("#\n")
    f.write("# dimensionless ratio of densities of ionizing photons to hydrogen\n")
    f.write(f"ionization parameter {log10_U_S:.2f}\n")
    f.write("#\n")
    f.write("# abundances taken from Gutkin 2016\n")
    f.write(f"abundances {logrel_abundances_str}\n")
    f.write("#\n")
    f.write("# depletion factors taken from Gutkin 2016 as well\n")
    f.write("metals deplete \"ISM_gutkin.dep\"\n")
    f.write("#\n")
    f.write("# calculation will stop when electron temperature falls below this value\n")
    f.write("stop temperature 4000\n")
    f.write("#\n")
    f.write("# add grains, with abundances scaled to match those assumed for carbon and silicon above, PAHs are important at 10^5 AA only\n")
    f.write("grains Orion\n")
    f.write("#\n")
    f.write("# save output spectrum\n")
    f.write(f"save continuum units Angstroms \"sed_out_{star_ptc_initial_mass:.2e}_{star_ptc_metallicity:.2e}_{age_of_ssp:.2e}.txt\"\n")
