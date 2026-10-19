# Reproducing the mutation pilot from a clean clone

The generator and pilot read `jobs/` and `traindoctor_env/` from a checkout of the library. `TD_REPO` is the path to a checkout at the pinned commit (the one in the top-level `requirements.txt`, 0b0a36b).

Offline tests (no Modal, no API calls; 31 passed in 0.3 s on Python 3.12):

```
git clone https://github.com/TarangKhanna/traindoctor-research && git clone https://github.com/TarangKhanna/traindoctor-env
(cd traindoctor-env && git checkout 0b0a36b6b2a34a620e959c665a59f8a92b7413c2)
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r traindoctor-research/requirements.txt
export TD_REPO=$PWD/traindoctor-env
cd traindoctor-research/experiments/mutation_pilot && python -m pytest -q
```

Validation replays additionally need a Modal account (client importable, or `TD_MODAL_SITE` pointing at its site-packages) and cost about $0.15 of CPU:

```
python -m pilot.validate run    # 8 tasks x noop/gold/decoy/alt trusted replays
```
