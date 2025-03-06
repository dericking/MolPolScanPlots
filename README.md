# MolPolScan Plotter and Table Maker
Plotting results from halla_molpol_sim scans. 

## Execution
python3 MolPolScans.py 

### Execution Flags/Arguments
| Flag | Description |
|------|-------------|
| --file | file to read [CSV with additional metadata line] |
| --energy | beam energy [in GeV] |
| --magnet | specify scan magnet [1-5] |
| --setpoint | (optional) adds set point to plot and table outputs |

## CSV File Format

Line 1: Simulation set metadata

Line 2: CSV header

Line 3: Simulation summary data

### Line 1
Metadata sampled from a halla_molpol_sim macro. The form itself is arbitary and the python plotting script looks for the keys

Sample ==> "DESC=4PASS_ERIC_DP,Q1=-0.45,Q2=-0.45,Q3=0.4,Q4=0.44,Q5=1.345,Q6=4,thcommin=45deg,thcommax=135deg,phimin=-45deg,phimax=40deg,beamE=8.59GeV,targetPolPct=0.08015"

| DESC | Description |
|------|-------------|
| Q1 | Quad 1 [Pole Tip in Tesla] |
| Q2 | Quad 2 [Pole Tip in Tesla] |
| Q3 | Quad 3 [Pole Tip in Tesla] |
| Q4 | Quad 4 [Pole Tip in Tesla] |
| Q5 | Dipole [Tesla] |
| Q6 | Helmholtz Coil [Tesla] |
| thcommin | Minimum Theta (Center of Mass) [degrees] |
| thcommax | Minimum Theta (Center of Mass) [degrees] |
| phimin | Minimum Theta (Center of Mass) [degrees] |
| phimax | Minimum Theta (Center of Mass) [degrees] |
| beamE | Beam Energy in GeV |
| targetPolPct | Target Spin Polarization |

### Line 2 & 3

CSV Header of results from analyzed simulation data should contain all of the following items

| Header Value  | Description |
|--------------|-------------|
| MagPoleTip   | Pole tip for magnet being scanned       |
| MagCurrent   | Column for associated current with that pole tip       |
| CoinUncAzz   | Uncorrected analyzing power for coincidences       |
| CoinCorAzz   | Corrected analyzing power for coincidences       |
| CoinCorErr   | Standard error on the mean corrected coincidence analyzing power      |
| CoinNormRt   | Percent of events that resulted in a coincidence       |
| LeftUncAzz   | Uncorrected analyzing power for electron hits on the left spectrometer arm       |
| LeftCorAzz   | Uncorrected analyzing power for electron hits on the left spectrometer arm       |
| LeftCorErr   | Standard error on the mean corrected left-arm analyzing power       |
| LeftNormRt   | Percent of events that resulted in a electron hit on the left spectrometer arm      |
| RghtUncAzz   | Uncorrected analyzing power for electron hits on the right spectrometer arm      |
| RghtCorAzz   | Corrected analyzing power for electron hits on the right spectrometer arm      |
| RghtCorErr   | Standard error on the mean corrected right-arm analyzing power      |
| RghtNormRt   | Percent of events that resulted in a electron hit on the right spectrometer arm      |
| SimuEvents   | Total number of simulated events      |
