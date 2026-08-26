# Camera–Radar Multimodal Engine

This directory is the repository-owned source for the living-room fall-risk
algorithm service.  It runs as an independent process because RTMPose3D,
BioSTGCN, OpenSDK and the GPU runtime do not belong in the App backend
container.

The production data path is:

```text
Camera BioSTGCN + IWR6843 B0 calibrated TCN
  -> Camera/Radar timestamp alignment
  -> spatial Alignment Gate
  -> Radar Eligibility Gate
  -> Camera-led Fusion v2
  -> App backend adapter
  -> Android App
```

The engine preserves the existing Camera model, Radar checkpoint, 50 ms
alignment gate, Fixed 0.6/0.4 baseline and Camera-led Fusion v2 rules.
Track-first / Risk-late-binding remains a shadow pipeline.

## Repository boundary

Committed here:

- runtime contracts, adapters and fusion services;
- Camera/Radar alignment and eligibility logic;
- API routes, tests and offline tools;
- the small living-room calibration JSON;
- dependency and environment templates.

Not committed:

- public datasets or recorded Camera/Radar sessions;
- runtime JSONL/log output;
- local secrets and SDK binaries;
- model weights other than separately approved deployment artifacts.

## Local launch

Use the existing algorithm Python environment and run from the repository
`backend` directory so the `app.modules...` package path is stable:

```powershell
Set-Location "<repo>\backend"
& "<algorithm-python>" -m uvicorn `
  app.modules.fall.multimodal_engine.main:app `
  --host 127.0.0.1 --port 8001
```

The repository launcher provides the same operation without depending on the
caller's working directory:

```powershell
Set-Location "<repo>"
.\scripts\start-multimodal-engine.ps1 `
  -PythonPath "<algorithm-python>" `
  -Port 8001
```

Copy `.env.example` to `.env` in this directory for machine-local algorithm
settings.  The App backend continues to consume the service through
`APP_FALL_RISK_BASE_URL=http://host.docker.internal:8001`.

The IWR6843 worker remains independently runnable from
`fall/radar_module` on port 8010.
