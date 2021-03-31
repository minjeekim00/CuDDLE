import numpy as np

from barcode import Barcode

superior = np.load('./brain_superior_embs.npz')['distance'].squeeze()
inferior = np.load('./brain_inferior_embs.npz')['distance'].squeeze()

barcode = Barcode(superior, inferior)
barcode_dict = barcode.get_barcode()
