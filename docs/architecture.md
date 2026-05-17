# Tehnicka arhitektura projekta

## Pregled pipeline-a

```
[PTB-XL dataset]            [MIT-BIH NSTDB]
       |                           |
       v                           v
01_prepare_data.py           noise/nstdb.py
(segmentacija, normalizacija) (ucitavanje šuma, resampliranje)
       |
       v
[data/processed/*.npz]
(train.npz, val.npz, test_clean.npz)
       |
       +------------------+
       |                  |
       v                  v
02_train.py          02_train_v3.py
(V1 - cisti signali)  (V3 - augmentacija šumom)
       |                  |
       v                  v
[best_model.pt]      [best_model_v3.pt]
       |                  |
       +--------+---------+
                |
     +----------+----------+
     |                     |
     v                     v
03_evaluate_clean.py   04_generate_noisy.py
(evaluacija na         (30 šumnih test skupova
 cistom skupu)          5 vrste x 6 SNR nivoa)
                              |
                              v
                        05_evaluate_noisy.py
                        (evaluacija robusnosti
                         na svim kombinacijama)
                              |
                              v
               06_generate_thesis_figures.py
               07_generate_thesis_tables.py
               08_export_thesis_numbers.py
               09_sync_thesis_assets.py
```

## Opis modula (`src/ekg_mi/`)

### `data/`

| Modul              | Namjena                                                             |
|-------------------|---------------------------------------------------------------------|
| `loader.py`        | Ucitavanje PTB-XL WFDB zapisa i metadata (pandas + wfdb)           |
| `dataset.py`       | PyTorch Dataset omotac (EKGDataset), stratifikovano dijeljenje      |
| `mock_dataset.py`  | Sinteticki EKG za testove (bez realnih podataka)                    |
| `preprocessing.py` | Bandpass filter (0.5-40 Hz, Butterworth) i z-score normalizacija    |

### `models/`

| Modul              | Namjena                                                             |
|-------------------|---------------------------------------------------------------------|
| `baseline_cnn.py`  | 1D CNN arhitektura - 4 konvolucijska bloka + MLP klasifikator       |

### `noise/`

| Modul              | Namjena                                                             |
|-------------------|---------------------------------------------------------------------|
| `synthetic.py`     | Generisanje gaussovog šuma i šuma elektricne mreze (50 Hz)         |
| `nstdb.py`         | Ucitavanje realnog šuma iz MIT-BIH NSTDB, resampliranje 360->100 Hz|
| `injection.py`     | SNR-kontrolisano dodavanje šuma (tacnost ±0.5 dB)                  |
| `augmentation.py`  | Batch-nivo augmentacija za trening V3 (vjerovatnoca 0.5)           |

### `training/`

| Modul              | Namjena                                                             |
|-------------------|---------------------------------------------------------------------|
| `trainer.py`       | Osnovna petlja treninga s ranim zaustavljanjem                      |

### `evaluation/`

| Modul              | Namjena                                                             |
|-------------------|---------------------------------------------------------------------|
| `evaluator.py`     | Inferencija i binarna klasifikacija s pragom odlucivanja            |
| `metrics.py`       | 7 metrika: F1, AUC-ROC, AUC-PR, tacnost, preciznost, odziv, specificnost |

### `visualization/`

| Modul              | Namjena                                                             |
|-------------------|---------------------------------------------------------------------|
| `style.py`         | Tezna paleta boja i matplotlib stil (seaborn tema)                  |
| `confusion.py`     | Heatmapa konfuzione matrice                                         |
| `signals.py`       | Plotanje EKG signala (vremenski domen)                              |
| `training_plots.py`| Krivulje gubitka i metrika po epohi                                |

### `utils/`

| Modul              | Namjena                                                             |
|-------------------|---------------------------------------------------------------------|
| `seed.py`          | Fiksacija sjemena (PyTorch, NumPy, Python random, CUDA)             |
| `io.py`            | Pomocne funkcije za ucitavanje i cuvanje NPZ/JSON fajlova           |

---

## Opis skripti (`scripts/`)

| Skript                              | Ulaz                             | Izlaz                                    |
|------------------------------------|----------------------------------|------------------------------------------|
| `00_smoke_test.py`                 | mock podaci                      | provjera radi li pipeline                |
| `01_prepare_data.py`               | `data/raw/`                      | `data/processed/*.npz`                   |
| `02_train.py`                      | `data/processed/`                | `outputs/models/best_model.pt`           |
| `02_train_v3.py`                   | `data/processed/`                | `outputs/models/best_model_v3.pt`        |
| `03_evaluate_clean.py`             | `best_model.pt`, `test_clean.npz`| `outputs/reports/eval_clean.json`        |
| `03_evaluate_clean_v3.py`          | `best_model_v3.pt`               | `outputs/reports/eval_clean_v3.json`     |
| `04_generate_noisy.py`             | `test_clean.npz`                 | `data/noisy/*.npz` (30 fajlova)          |
| `05_evaluate_noisy.py`             | `best_model.pt`, `data/noisy/`   | `outputs/reports/eval_noisy.json`        |
| `05_evaluate_noisy_v3.py`          | `best_model_v3.pt`               | `outputs/reports/eval_noisy_v3.json`     |
| `06_generate_thesis_figures.py`    | eval JSON fajlovi                | `outputs/figures/thesis/*.pdf`           |
| `06_tune_threshold.py`             | `best_model.pt`                  | `outputs/reports/threshold_tuning.json`  |
| `06_tune_threshold_v3.py`          | `best_model_v3.pt`               | `outputs/reports/threshold_tuning_v3.json`|
| `07_generate_thesis_tables.py`     | eval JSON fajlovi                | `outputs/tables/thesis/*.tex`            |
| `07b_generate_comparison_tables.py`| oba eval JSON seta               | V1 vs V3 poredba (tabele)                |
| `08_export_thesis_numbers.py`      | eval JSON fajlovi                | `thesis/numbers.json`                    |
| `08_generate_comparison_figures.py`| oba eval JSON seta               | V1 vs V3 grafovi                         |
| `09_sync_thesis_assets.py`         | `outputs/`                       | `thesis/` (lokalni folder teze)          |
| `compare_v1_v3.py`                 | oba eval JSON seta               | Sveobuhvatna poredba V1 vs V3            |

---

## Arhitektura CNN modela (`BaselineCNN`)

```
Ulaz: [batch, 12, 1000]
  - 12 EKG odvoda (sve elektode PTB-XL)
  - 1000 vremenskih koraka (10 sekundi @ 100 Hz)

Blok 1: Conv1d(12,  32,  kernel=16, padding=7) -> BatchNorm1d -> ReLU -> MaxPool1d(2)
  Izlaz: [batch, 32,  500]

Blok 2: Conv1d(32,  64,  kernel=5,  padding=2) -> BatchNorm1d -> ReLU -> MaxPool1d(2)
  Izlaz: [batch, 64,  250]

Blok 3: Conv1d(64,  128, kernel=5,  padding=2) -> BatchNorm1d -> ReLU -> MaxPool1d(2)
  Izlaz: [batch, 128, 125]

Blok 4: Conv1d(128, 256, kernel=5,  padding=2) -> BatchNorm1d -> ReLU -> MaxPool1d(2)
  Izlaz: [batch, 256, 62]

AdaptiveAvgPool1d(1) -> [batch, 256]
Flatten()            -> [batch, 256]
Dropout(0.5)
Linear(256, 64)      -> ReLU
Linear(64, 1)        -> logit

Gubitak: BCEWithLogitsLoss
Prag klasifikacije: sigmoid(logit) >= 0.5 => MI (pozitivna klasa)
```

Ukupan broj trenabilnih parametara: ~275 000.

---

## Konfiguracija (`configs/config.yaml`)

| Parametar                        | Vrijednost        | Opis                                     |
|---------------------------------|-------------------|------------------------------------------|
| `seed`                           | 42                | Sjeme za reproducibilnost                |
| `paths.raw`                      | `data/raw`        | Sirovi podaci (PTB-XL, NSTDB)            |
| `paths.processed`                | `data/processed`  | Obradeni segmenti (NPZ)                  |
| `paths.noisy`                    | `data/noisy`      | Šumni test skupovi                       |
| `paths.models`                   | `outputs/models`  | Sacuvani modeli (.pt)                    |
| `paths.figures`                  | `outputs/figures` | Grafovi (.pdf, .png)                     |
| `paths.reports`                  | `outputs/reports` | Metrike (.json)                          |
| `paths.tables`                   | `outputs/tables`  | LaTeX tabele (.tex)                      |
| `preprocessing.sampling_rate`    | 100 Hz            | Ciljni sampling rate (PTB-XL je na 100 Hz)|
| `preprocessing.segment_length`   | 1000              | Dužina segmenta u uzorcima (10 sekundi)  |
| `preprocessing.bandpass_low`     | 0.5 Hz            | Donja granica bandpass filtera           |
| `preprocessing.bandpass_high`    | 40.0 Hz           | Gornja granica bandpass filtera          |
| `model.in_channels`              | 12                | Broj EKG odvoda                          |
| `model.num_filters`              | [32, 64, 128]     | Reference za konfiguraciju (model koristi 256 u bl.4)|
| `model.kernel_size`              | 7                 | Referentna vrijednost kernela            |
| `model.dropout`                  | 0.3               | Referentna stopa (klasifikator koristi 0.5)|
| `model.num_classes`              | 1                 | Binarna klasifikacija                    |
| `training.batch_size`            | 64                | Velicina batch-a                         |
| `training.lr`                    | 1e-3              | Pocetna stopa ucenja (Adam)              |
| `training.max_epochs`            | 50                | Maksimalan broj epoha                    |
| `training.early_stopping_patience`| 7               | Epohe bez poboljšanja do zaustavljanja   |
| `training.val_split`             | 0.15              | Udio validacionog skupa                  |
| `training.test_split`            | 0.15              | Udio test skupa                          |
| `noise.types`                    | 5 vrsta           | gaussian, baseline_wander, powerline, muscle_artifact, electrode_motion |
| `noise.snr_db`                   | [-6,0,6,12,18,24] | Šest nivoa odnosa signal-šum             |
