import pandas as pd
import numpy as np

file_path = "../../data/radiative_transfer/output/"
file_name = file_path + "data.h5"

hdf_keys = np.array([])

# Finding hdf keys
with pd.HDFStore(file_name, 'r') as hdf:
     hdf_keys = np.append(hdf_keys, hdf.keys())


for i in hdf_keys:
     table = pd.read_hdf(file_name, i) # Face-on view
     wvl = table['wvl'].to_numpy() # rest-frame wavelength [micron]
     flux = table['flux'].to_numpy() # flux [W/m^2]
     r = table['r'].to_numpy() # half-light radius [kpc]
     n = table['n'].to_numpy() # Sersic index
     df = pd.DataFrame({"wvl":wvl, "flux":flux, "r":r, "n":n})
     df.to_csv(f"{file_path}/raw_data/{i[1:]}.csv", index=False)

