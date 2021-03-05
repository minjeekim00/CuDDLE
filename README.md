# Official Code Implementation for Barcode
This is a repository for the paper "Barcode Method for Generative Model Evaluation driven by Topological Data Analysis".

# Basic Concepts

- (Mutual) Fidelity : How well the generative model generates fake images.
- (Mutual) Diversity : How diverse the generated images are.

However, these two are relative values, not absolute values. Thereforre, we suggest users to calculate **Relative Fidelity** and **Relative Diversity**.

- Relative Fidelity : (Fidelity between real and generated images) / (Fidelity between real and real images)
- Relative Diversity : (Diveristy between real and generated images) / (Diversity between real and real images)

These two can be considered as normalized fidelity and normalized diversity compared to real data.

# Usage

Take a look in

    barcode_example.py

**barcode_example.py** is composed as:

```
barcode_realfake = Barcode(real, fake)
barcode_realreal = Barcode(real, real)

rf_f, rf_d = barcode_realfake.get_barcode()
rr_f, rr_d = barcode_realreal.get_barcode()

print("Mutual Fidelity   : {:.3f} | Mutual Diversity   : {:.3f}".format(rf_f, rf_d))
print("Relative Fidelity : {:.3f} | Relative Diversity : {:.3f}".format(rf_f/rf_f, rf_d/rr_d))
```
