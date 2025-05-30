FROM pytorch/pytorch:latest
WORKDIR /app

# Skopiuj kod źródłowy do kontenera
COPY .. /app

RUN pip install --no-cache-dir -r requirements.txt

# Domyślna komenda
CMD ["python", "-m", "tools.train"]
