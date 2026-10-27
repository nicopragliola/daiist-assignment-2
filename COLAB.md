# Training on Google Colab

Runtime > Change runtime type > T4 GPU. Colab brings its own PyTorch, so do not
run `uv sync` there. The dataset download is about 2 GB; the whole run takes
roughly 10 to 15 minutes on a T4.

1. In Colab, open the key icon (Secrets) on the left, add a secret named
   `ROBOFLOW_API_KEY` with your key, and allow this notebook to read it.

2. Run these cells:

```python
!git clone -b assignment-2 https://github.com/nicopragliola/daiist-assignment-2.git
%cd daiist-assignment-2
!pip install -q "ultralytics>=8.4,<8.5" "roboflow>=1.7"
```

```python
import os
from google.colab import userdata
os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")
```

```python
!python main.py train
```

3. Download what the run produced:

```python
!nvidia-smi --query-gpu=name --format=csv,noheader > results/trained_on.txt
!zip -qr training_outputs.zip weights results
from google.colab import files
files.download("training_outputs.zip")
```

4. Unzip it on your computer. On GitHub, switch to the `assignment-2` branch,
   then Add file > Upload files, and drag in the `weights` and `results`
   folders. Commit directly to `assignment-2`.

Changing only the costs in `assumptions.json` does not need Colab or a new
training run: `uv run python decision.py` redoes the costing from the saved
scores.
