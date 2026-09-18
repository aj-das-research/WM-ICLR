# Isolated real-video data audit runtime

Python 3.11; install the exact dependencies from `requirements.lock.txt` in this directory's `.venv`. System ffmpeg/ffprobe decode original MP4 footage. This environment is separate from both model training and simulation.

From the project root:

```bash
python3.11 -m venv environments/real_video/.venv
environments/real_video/.venv/bin/python -m pip install -r environments/real_video/requirements.lock.txt
environments/real_video/.venv/bin/python scripts/real_video/fetch_openh_sample.py
```

The script downloads only the named Open-H subset metadata and first episode MP4/Parquet. It records an immutable upstream revision, verifies downloaded byte sizes/LFS SHA256 values, decodes the real footage, checks nominal timing and schema, and writes a contact sheet plus JSON audit. No access token, model checkpoint, training, or external publication is used.
