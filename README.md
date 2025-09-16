# 🌳 Forest Ecosystem Stability
This repository contains the code and analysis for my master's thesis, which investigates critical transitions in forest ecosystems using agent-based models. The project aims to understand how ecosystem tipping points change with increasing complexity of positive feedbacks inherent in local-scale ecosystem models.

## 📜 Abstract
Several large ecosystems face irreversible, abrupt transitions to different states, such as a rainforest transitioning to a savannah. These shifts are driven by positive feedback loops that become dominant at tipping points (TPs). 

"Critical slowing down" (CSD) can be used to predict TPs by monitoring increased variability in observables, acting as early-warning signals (EWS). My research explores how spatial feedback mechanisms affect forest resilience and investigates if classic tipping point theory applies to more complex ecological systems.

## 💻 Methodology
- Model Construction: Two individual-based forest ecosystem models were constructed to examine the effects of local vs. non-local positive feedbacks. The individual-based approach ensures that the saddle-node bifurcation is not "baked in" to the model via governing differential equations.
- Numerical Simulation: The project numerically found different stable vegetation configurations within the models and determined their bifurcations as control parameters, such as mean annual rainfall, varied.
- Data Analysis: The simulations were used to constrain under which circumstances abrupt, system-wide ecosystem collapse could be expected and to assess whether a collapse could be predicted from real-world observations. I analyzed whether CSD could be detected in a spatially explicit model with local interactions.

## 🔬 Results
The individual-based models were found to exhibit tipping dynamics, including multistability and a form of bifurcation. Localized feedback was found to decrease the window of multistability. While CSD was not explicitly modeled, the main idea of predicting tipping points through a loss of resilience seems to be preserved. However, measuring this loss of resilience via a decrease in variance was found to be ineffective.
