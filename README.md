# A regret policy for the DVRPTW

This repository includes a regret policy for the dynamic vehicle routing problem with time windows (DVRPTW), which has been developed for the EURO meets NeurIPS Vehicle Routing Challenge.
The proposed policy is to be found in baselines/strategies/_strategies.py.
For further information on how the code can be called we refer to [this](https://github.com/ortec/euro-neurips-vrp-2022-quickstart) repo. 
The explanation of the algorithm can be found in UPB.pdf.

The code can be called as follows:

``python solver.py --instance instances/ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35.txt --verbose``