# Eyegaze vectors (local data)

Place per-video CSVs here as `{video_id}.csv` (same layout as before: `timestamp`, `gaze_0_*`, `gaze_1_*`, etc.). The app serves them from this directory; they are **not** part of the git repository.

If you already had this folder before it was removed from git, your files stay on disk after `git pull` only if you **back them up before pulling**, then restore:

```bash
# Before git pull (on a clone that still had tracked CSVs)
cp -a data/eyegaze_vectors "data/eyegaze_vectors.bak"
git pull
rm -rf data/eyegaze_vectors && mv "data/eyegaze_vectors.bak" data/eyegaze_vectors
```

Or pull first, then copy the folder back from backup or shared storage.
