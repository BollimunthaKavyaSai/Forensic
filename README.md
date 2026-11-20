# Cross-Platform Digital Forensic Triage Validated on Acquired Evidence

Code, trained models and per-fold results for the paper
*"Cross-Platform Digital Forensic Triage Validated on Acquired Evidence: An Open-Source Pipeline with
Leakage-Controlled Graph and Tabular Learning"*.

## Layout

| Path | Contents |
|---|---|
| `experiments/cfreds_validate.py` | Scores the extraction stage against the NIST CFReDS Data Leakage Case answer key (Section 5.1) |
| `experiments/atlas_parse.py`, `run_atlas.py` | ATLASv2 Security/Sysmon logs → process provenance graphs → cross-vector / LOSO evaluation of LR, RF, XGBoost, MLP, GCN, GraphSAGE, GAT (Section 5.2) |
| `experiments/run_kronodroid.py` | KronoDroid random, family-grouped and temporal protocols, incl. FT-Transformer (Section 5.3) |
| `experiments/android_apk_e2e.py` | Deployable static APK model and end-to-end test on F-Droid APKs (Section 5.3) |
| `experiments/data_prep.py`, `run_experiments.py`, `analyze_results.py` | Audit of the earlier benchmarks (Kaggle activity log, ADROIT) (Section 5.4) |
| `experiments/analyze_r4.py` | Summary tables, significance tests (Holm-adjusted) and figures |
| `experiments/prototype_ext/` | `.evtx` parser, APK feature extractor, trained triage models (`models/`) used by the app |
| `experiments/tsk_shim.py` | Pure-Python pytsk3/pyewf replacement (dissect.evidence + dissect.ntfs) |
| `experiments/results/` | Per-fold CSVs, significance tests, CFReDS outputs, figures |
| `app/` | Django REST back end + React front end of the prototype, with the corrections described in Section 5.5 |

## Data (not redistributed)

* NIST CFReDS Data Leakage Case: https://cfreds-archive.nist.gov/data_leakage_case/
* ATLASv2: https://bitbucket.org/sts-lab/atlasv2 — labels: https://bitbucket.org/sts-lab/reapr-ground-truth
* KronoDroid: https://github.com/aleguma/kronodroid
* F-Droid APKs: https://f-droid.org/repo/
* Audited benchmarks: https://www.kaggle.com/datasets/jimohyusuf/cybercrime-forensic-dataset and
  https://www.kaggle.com/datasets/saurabhshahane/android-malware-dataset

Download them and adjust the paths at the top of each script.

## Running

```bash
pip install -r requirements.txt
cd experiments
python atlas_parse.py            # builds results/atlas_graphs.pkl
SEEDS=0,1 python run_atlas.py cross_vector
python run_kronodroid.py family all
python analyze_r4.py
```

## App configuration

Database credentials are read from environment variables: `POSTGRES_USER`, `POSTGRES_PASSWORD`,
`DJANGO_SECRET_KEY`. Values in `app/config/*.yaml` and setup scripts are local development defaults only.
