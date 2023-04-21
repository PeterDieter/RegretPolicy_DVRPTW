# A regret policy for the DVRPTW

This repository includes a regret policy for the dynamic vehicle routing problem with time windows (DVRPTW), which has been developed for the EURO meets NeurIPS Vehicle Routing Challenge. The approach ranked first place on the weekly leaderboard for approximately half the time the competition lasted and thereby earned 380€ in prize money. Furthermore, it ranked 6th place in the final evaluation (for the dynamic case) among approximately 50 teams and has been presented at the NeurIPS 2022 Virtual Workshop.
The proposed policy is to be found in baselines/strategies/_strategies.py.
For further information on how the code can be called we refer to [this](https://github.com/ortec/euro-neurips-vrp-2022-quickstart) repo. 
The explanation of the algorithm can be found in [UPB.pdf](https://github.com/ortec/euro-neurips-vrp-2022-quickstart/blob/main/papers/UPB.pdf).

Also make sure that you first create an executable of the HGS solver:

```
cd baselines/hgs_vrptw
make clean
make all
```

The code can be called as follows:

``python solver.py --instance pathToInstance --verbose``
