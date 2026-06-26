# GCAT-FCMAC

Lightweight Graph Neural Network with Fuzzy Cerebellar Model for Real-Time Intrusion Localization in Industrial Power IoT.

Published in *IEEE Transactions on Industrial Informatics*.

## Note

The source code files in this repository have been **encrypted**. If you need access to the decryption key, please contact the corresponding author.

## Requirements

```bash
pip install -r requirements.txt
```

- Python >= 3.9
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.4.0
- scikit-learn >= 1.3.0
- NumPy >= 1.24.0

## Project Structure

```
├── README.md
├── requirements.txt
├── model.py
├── dataset.py
├── train.py
├── evaluate.py
├── config.py
├── data.zip
└── best_model_V3.*
```
## Simple Model file
- model.py

## Full file with model, train, evaluate and config
- code.zip

## Data Availability

- **Simulated datasets**: Included in `data.zip`.
- **Pre-trained model**: Available as `best_model_V3.7z.*` (split archive).
- **Real-world dataset**: Collected from a 110 kV substation with 20 sensor devices. Due to confidentiality agreements, the raw dataset is not publicly available. Anonymized versions can be requested for academic research by contacting the corresponding author.

## Citation

```bibtex
@article{gcat_fcmac_tii,
  title={Toward Real-Time Intrusion Localization in Industrial Power IoT: A Lightweight Graph Neural Network with Fuzzy Cerebellar Model},
  journal={IEEE Transactions on Industrial Informatics},
  year={2026}
}
```

## License

This project is licensed under the MIT License.
