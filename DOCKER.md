# Docker container for Jupyter + notebooks

1. **Build the image**
   ```
   docker build -t bigproject-ai .
   ```

2. **Run a container**
   ```
   docker run -it --rm \
     -p 8888:8888 \
     -p 7767:7767 \
     -v "${PWD}/save_original":/save_original \
     bigproject-ai
   ```

   - The four files you asked for (`api.key.txt`, `db_info.txt`, `final.ipynb`, `run.ipynb`) are baked into `/app`.
   - `/save_original` is declared as a volume so you can mount the host `save_original` directory and keep files out of the image layers.
   - `CMD` launches `jupyter notebook` bound to `0.0.0.0:8888` with an empty token, so browse to `http://localhost:8888` from your host.
   - Port 7767 is exposed for when you start `uvicorn final:app ...` inside the container (use another shell with `docker exec -it <container> bash` if needed).
