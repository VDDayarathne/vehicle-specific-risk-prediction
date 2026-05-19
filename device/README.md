# Edge Device Prototype

This folder contains a minimal edge runtime for the KaduGuard research prototype.

What it does:
- reads OBD-II telemetry when the `python-obd` package and adapter hardware are available
- falls back to a deterministic simulator for demos and development
- posts samples to `POST /api/telemetry`
- stores failed uploads in a local JSONL queue and retries them later

## Run locally

```bash
python -m device.edge_agent --backend http://localhost:8000 --vehicle car --simulator
```

To simulate a bus or lorry:

```bash
python -m device.edge_agent --backend http://localhost:8000 --vehicle bus --simulator
```

## Optional live OBD-II mode

If you have a compatible adapter and install `python-obd`, omit `--simulator`:

```bash
python -m device.edge_agent --backend http://localhost:8000 --vehicle car
```

## Docker

Build and run the simulator container:

```bash
docker build -f device/Dockerfile -t kaduguard-edge .
docker run --rm kaduguard-edge
```

## Files

- `edge_agent.py` - posts telemetry and handles offline buffering
- `obd_adapter.py` - live OBD reader with simulator fallback
- `queue/` - offline JSONL retry queue
