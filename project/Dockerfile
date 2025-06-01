FROM dustynv/l4t-pytorch:r36.4.0
WORKDIR /app

# Skopiuj kod źródłowy do kontenera
COPY .. /app
#RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://pypi.jetson-ai-lab.dev/jp6/cu126/

RUN pip install --no-cache-dir -r requirements.txt

# Domyślna komenda
#CMD ["python3", "-m", "tools.train"]
CMD ["python3", "-m", "tools.infer", "--evaluate", "True", "--infer_samples", "False"]
