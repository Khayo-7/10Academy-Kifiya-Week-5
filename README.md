# 10Academy-Kifiya-Week-5
```
├───.github
│   └───workflows
├───dashboard
│   ├───data
│   ├───logs
├───deployment
│   ├───app
│   ├───logs
├───logs
├───notebooks
├───resources
│   ├───configs
│   ├───data
│   ├───encoders
│   ├───models
│   │   └───checkpoints
│   └───scalers
├───screenshots
│   ├───dashboard
│   └───deployment
├───scripts
│   ├───data_utils
│   ├───modeling
│   ├───utils
├───src
└───tests
```

```sh
mkdir -p .github/workflows dashboard/{data,logs} deployment/{app,logs} logs notebooks resources/{configs,data,encoders,models/checkpoints,scalers} screenshots/{dashboard,deployment} scripts/{data_utils,modeling,utils} src tests
```

```sh
touch requirement.txt .github/workflows/ci.yml deployment/app/__init__.py notebooks/initial_EDA.ipynb scripts/{data_utils,modeling,utils}/__init__.py scripts/__init__.py src/__init__.py tests/__init__.py
```