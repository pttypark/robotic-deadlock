# RWARE 3D AGV Viewer

This viewer replays an exported FCFS cross-simulation trace in a browser-based
Three.js scene.

## Generate a Trace

```powershell
.\JS26\Scripts\python.exe scripts\export_fcfs_3d_trace.py
```

The default output is:

```text
viewer3d/data/fcfs_trace.json
```

## Run Locally

Serve the `viewer3d` directory with any static HTTP server, then open the page:

```powershell
cd viewer3d
python -m http.server 8765 --bind 127.0.0.1
```

```text
http://127.0.0.1:8765/
```

The page imports Three.js from the public unpkg CDN.
