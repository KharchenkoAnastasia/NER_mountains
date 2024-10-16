from pathlib import Path

import pandas as pd
#from sklearn.model_selection import train_test_split



ROOT_DIR = Path(__file__).parent.parent
DATASET = ROOT_DIR / "data" / "mountain_ner_dataset.csv"

def main() -> None:
    # Load data
    df = pd.read_csv(DATASET, sep=';')


if __name__ == "__main__":
    main()