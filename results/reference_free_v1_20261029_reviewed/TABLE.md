# Reference-free saved-evidence detection

source: executed

| case | recovery | complete | hashes | reference-free | prior oracle agrees | max abs | max relative* | wall s |
|---|---|---|---|---|---|---:|---:|---:|
| case0 | resume | True | True | reject | True | 8 | 8000000000000 | 0.026139 |
| case0 | restore_sampler | True | True | accept | True | 7.5610304295370412e-08 | 2.1859106281341663e-05 | 0.015659 |
| case0 | restore_optimizer | True | True | reject | True | 8 | 8000000000000 | 0.019859 |
| case0 | restore_full | True | True | accept | True | 7.5610304295370412e-08 | 2.1859106281341663e-05 | 0.014452 |
| case0 | older | True | True | accept | True | 7.5610304295370412e-08 | 2.1859106281341663e-05 | 0.017732 |
| case0 | scratch | True | True | accept | True | 7.5610304295370412e-08 | 2.1859106281341663e-05 | 0.018021 |
| case1 | resume | True | True | reject | True | 0.72322484439293888 | 227.82069745494792 | 0.017845 |
| case1 | restore_sampler | True | True | reject | True | 0.72322484439293888 | 227.82069745494792 | 0.019700 |
| case1 | restore_optimizer | True | True | accept | True | 7.5610304295370412e-08 | 2.1859106281341663e-05 | 0.018071 |
| case1 | restore_full | True | True | accept | True | 7.5610304295370412e-08 | 2.1859106281341663e-05 | 0.015378 |
| case1 | older | True | True | accept | True | 7.5610304295370412e-08 | 2.1859106281341663e-05 | 0.015020 |
| case1 | scratch | True | True | accept | True | 7.5610304295370412e-08 | 2.1859106281341663e-05 | 0.018404 |

*Coordinate relative error uses denominator max(abs(expected), 1e-12); diagnostic only.
Maximum includes original 4→8 and observed overlap checks, not only final tensors.
All acceptance is conditional on trusted checkpoint 4 and the specified input/update recipe.
