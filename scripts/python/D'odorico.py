# Inspired by D'odorico et al. (2007) "On soil moisture–vegetation feedbacks and their possible effects on the dynamics of dryland ecosystems"

import numpy as np

# Change in vegitation biomass
def dVdt(V, Vcc, α):
    return α * V * (Vcc - V)

def Vcc(θ_bar, θ_bar_w, ζ):
    if θ_bar <= θ_bar_w:
        return 0
    else:
        Δ = θ_bar - θ_bar_w
        return Δ / (ζ + Δ)

def θ_bar(V, θ_bar_0, β):
    return θ_bar_0 + β * V
