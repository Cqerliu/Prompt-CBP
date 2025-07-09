# Prompt-CBP: a novel prompt learning-based model for predicting cross-species promoters
## Project Description
**Prompt-CBP** is a novel deep learning framework based on prompt learning for predicting promoters across multiple species. This repository provides:

- Implementation of the Prompt-CBP model architecture
- Preprocessed promoter datasets for 5 model organisms
- Training and prediction pipelines

## Dataset Information
### Supported Species
| Species | Type | Data Source |
|---------|------|-------------|
| *Homo sapiens* | Eukaryote | [EPDnew](https://epd.expasy.org/epd/EPDnew_database.php) |
| *Mus musculus* | Eukaryote | [EPDnew](https://epd.expasy.org/epd/EPDnew_database.php) |
| *Drosophila melanogaster* | Eukaryote | [EPDnew](https://epd.expasy.org/epd/EPDnew_database.php) |
| *Escherichia coli* | Prokaryote | [RegulonDB](https://regulondb.ccg.unam.mx/) |
| *Bacillus subtilis* | Prokaryote | [DBTBS](https://dbtbs.hgc.jp/) |

Preprocessed datasets are available in the `DataSet/data` directory.

## Repository Structure
```bash
Prompt-CBP/
├── DNABERT-2-117M/ # Pre-trained DNABERT-2 models
├── DataSet/ 
│ ├── MyDataset.py  # Data preprocessing and tokenization
│ └── data/ # Preprocessed promoter datasets
├── main/ # Main training scripts
│ ├── train.py # Model training script
│ └──predict.py # Prediction script
└── model/ # Model framework implementations
  └── BertCBP.py # Prompt-CBP model architecture
```


## Usage Instructions

### Data Preparation
Preprocessed datasets for all species are available in `DataSet/data/`. To use custom data:
1. Format sequences as csv files with promoter/non-promoter labels
2. Maintain directory structure consistent with provided datasets
3. Update paths in `MyDataset.py` accordingly

### DNABERT-2 Pre-trained Model Setup
Before training or prediction, download the DNABERT-2 base model.
Download the pre-trained DNABERT-2 model from the [official repository](https://github.com/MAGICS-LAB/DNABERT_2) and place it in the `DNABERT-2-117M/` directory.

### Model Training
Run the training script: You only need to run the train.py script to train the Prompt CBP model. We provide four different training strategies in this script. We also provide detailed annotation explanations in the script.

Four training strategies are supported:
1. Used for training baseline models.
2. Prompt strategy.
3. Output Fine-tuning strategy.
4. Prompt+Output Fine tuning strategy.

**Basic training command:**
```bash
cd main/
python train.py
```

### Promoter Prediction
Run predictions using a trained model:
```bash
python predict.py
```

### Dependencies
- Python 3.8+
- Core packages:
```bash
torch==2.6.0
transformers==4.52.4
peft==0.15.2
scikit-learn==1.3.2
numpy==1.26.4
pandas==2.3.0
```