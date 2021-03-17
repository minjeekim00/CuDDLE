import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
from barcode import Barcode

barcode_realfake = Barcode(real, fake)
barcode_realreal = Barcode(real, real)
barcode_fakefake = Barcode(fake, fake)

rf_f, rf_d = barcode_realfake.get_barcode()
rr_f, rr_d = barcode_realreal.get_barcode()
ff_f, ff_d = barcode_fakefake.get_barcode()

print("Mutual Fidelity   : {:.3f} | Mutual Diversity   : {:.3f}".format(rf_f, rf_d))
print("Relative Fidelity : {:.3f} | Relative Diversity : {:.3f}".format(rf_f/rr_f, rf_d/(np.sqrt(rr_d) * np.sqrt(ff_d))))
