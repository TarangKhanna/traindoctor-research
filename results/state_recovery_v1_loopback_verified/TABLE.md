# State recovery execution

source: executed

| state | action | completes | semantic valid | oracle agrees | redone steps | cost score |
|---|---|---|---|---|---:|---:|
| case0 | resume | True | False | True | 2 | 0.0000 |
| case0 | restore_sampler | True | True | True | 2 | 0.8333 |
| case0 | restore_optimizer | True | False | True | 2 | 0.0000 |
| case0 | restore_full | True | True | True | 2 | 0.8333 |
| case0 | older | True | True | True | 6 | 0.5000 |
| case0 | scratch | True | True | True | 10 | 0.1667 |
| case1 | resume | True | False | True | 2 | 0.0000 |
| case1 | restore_sampler | True | False | True | 2 | 0.0000 |
| case1 | restore_optimizer | True | True | True | 2 | 0.8333 |
| case1 | restore_full | True | True | True | 2 | 0.8333 |
| case1 | older | True | True | True | 6 | 0.5000 |
| case1 | scratch | True | True | True | 10 | 0.1667 |
