# CNN Classifier (PyTorch + FastAPI)

## Setup
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# GAN
# open VS Code terminal in project root
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# deps
python -m pip install --upgrade pip
pip install -r requirements.txt
# If you don't have requirements.txt:
# pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
# pip install fastapi "uvicorn[standard]" numpy==1.26.4 matplotlib==3.8.4 pillow tqdm

python -m scripts.train_gan_mnist
# outputs: weights\gan_mnist_gen.pt  (required by the API)

uvicorn app.main:app --reload
# open http://127.0.0.1:8000/docs

curl http://127.0.0.1:8000/health

docker build -t gan-api .
docker run -p 8000:80 gan-api
# then open http://127.0.0.1:8000/docs
