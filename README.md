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
touch .github/workflows/__init__.py dashboard/{data,logs}/__init__.py deployment/{app,logs}/__init__.py notebooks/__init__.py resources/{configs,data,encoders,models/checkpoints,scalers}/__init__.py screenshots/{dashboard,deployment}/__init__.py scripts/{data_utils,modeling,utils}/__init__.py scripts/__init__.py src/__init__.py tests/__init__.py
```