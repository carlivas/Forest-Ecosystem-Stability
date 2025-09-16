# 🌳 Forest Ecosystem Stability
This repo includes the code i wrote in the process of writing my Master's Thesis "Tipping Points and Early Warning Signals in Complex Ecosystem Models". 

This project was about whether the presumption that *tipping points* (TPs) (understood as a single saddle-node bifurcation) and the associated effect of *critical slowing down* occur in physical systems governed by agent-based dynamics, holds. This involved me writing an agent-based forest ecosystem model, modelling trees as agents which can
- grow
- compete
- reproduce

The reproducting is facilitated through a positive feedback mechanism, was intentionally designed to have a "global" and a "local" mode, the "global" mode being the globally averaged tree density and the "local" model being a local estimation of the tree density using a kernel density estimation approach.

The work took a full year to write and resulted in the insight that tipping points do in fact happen in these kinds of systems, and the locality of the positive feedback was found to be of special importance.
