import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import Barcode

barcode_realfake = Barcode(real, fake)
barcode_realreal = Barcode(real, real)

rf_f, rf_d = barcode_realfake.get_barcode()
rr_f, rr_d = barcode_realreal.get_barcode()

print("Mutual Fidelity   : {:.3f} | Mutual Diversity   : {:.3f}".format(rf_f, rf_d))
print("Relative Fidelity : {:.3f} | Relative Diversity : {:.3f}".format(rf_f/rf_f, rf_d/rr_d))