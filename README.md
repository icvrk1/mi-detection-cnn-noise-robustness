# Detekcija infarkta miokarda korištenjem CNN modela u uslovima prisustva šuma

Ovaj projekat implementira sistem za automatsku detekciju infarkta miokarda (MI) iz 12-kanalnih EKG signala korištenjem jednodimenzionalne konvolucijske neuronske mreže (1D CNN). Sistem je razvijen nad PTB-XL skupom podataka i fokusiran je na binarnu klasifikaciju zapisa na klase **MI** i **NORM**, uz strogu podjelu po pacijentima.

Poseban fokus projekta je analiza **robusnosti modela u prisustvu šuma**. Pored osnovnog modela treniranog na čistim signalima, implementiran je i model treniran uz **augmentaciju šumom**, kako bi se ispitalo da li takav pristup može poboljšati ponašanje modela u realističnijim uslovima rada.

Konačni rezultati u radu prikazani su kroz **10 nezavisnih treninga** za svaki model, uz izvještavanje srednje vrijednosti, standardne devijacije i statističke značajnosti razlika između modela.

---

## Ključni rezultati

### V1 model - bez augmentacije šumom
| Metrika | Vrijednost |
|--------|------------|
| F1 | 0.8868 ± 0.0039 |
| AUC-ROC | 0.9723 ± 0.0013 |
| Tačnost | 0.9145 ± 0.0038 |

### V3 model - sa augmentacijom šumom
| Metrika | Vrijednost |
|--------|------------|
| F1 | 0.8831 ± 0.0048 |
| AUC-ROC | 0.9724 ± 0.0014 |
| Tačnost | 0.9124 ± 0.0047 |

### Glavni zaključci
- Na **čistom testnom skupu** modeli V1 i V3 pokazuju **statistički ekvivalentne performanse**.
- Prednost modela V3 uočava se prvenstveno u uslovima **srednjeg nivoa šuma**.
- Augmentacija šumom daje **statistički značajna poboljšanja robusnosti** za pojedine tipove šuma, posebno za:
  - kolebanje osnovne linije,
  - mišićni artefakt,
  - artefakt pomicanja elektrode,
  pri SNR vrijednostima oko **0 dB i 6 dB**.

## Tehnologije

| Biblioteka    | Verzija  | Namjena                          |
|--------------|----------|----------------------------------|
| Python        | >= 3.10  | Programski jezik                 |
| PyTorch       | 2.5.1    | Neuronska mreza i trening        |
| wfdb          | 4.1.2    | Ucitavanje PTB-XL EKG zapisa     |
| neurokit2     | 0.2.10   | Ucitavanje NSTDB šum baze        |
| NumPy         | 2.1.2    | Numericke operacije              |
| SciPy         | 1.14.1   | Digitalni filteri                |
| scikit-learn  | 1.5.2    | Metrike klasifikacije            |
| pandas        | 2.2.3    | Rad s metapodacima PTB-XL        |
| matplotlib    | 3.9.2    | Vizualizacija                    |
| seaborn       | 0.13.2   | Stilizacija grafova              |
| plotly        | 5.24.1   | Interaktivni grafovi             |
| Streamlit     | 1.39.0   | Interaktivni dashboard           |
| PyYAML        | 6.0.2    | Konfiguracija                    |
| pytest        | 8.3.3    | Testovi                          |

## Struktura projekta

```
configs/              Centralna YAML konfiguracija (hiperparametri, putanje)
data/
  raw/                PTB-XL WFDB zapisi i NSTDB šum baza (preuzmite posebno)
  processed/          Segmentirani i normalizovani NumPy nizovi
  noisy/              Test skupovi s injektovanim šumom (30 kombinacija)
src/ekg_mi/           Instalabilni Python paket
  data/               Ucitavanje skupova podataka i DataLoader
  models/             Definicija 1D CNN arhitekture
  noise/              Modul za injekciju šuma (sinteticki + NSTDB)
  training/           Trening i rano zaustavljanje
  evaluation/         Metrike i evaluacija po tipu šuma
  visualization/      Pomocne funkcije za grafove
  utils/              Fiksacija seeda, ulaz/izlaz
scripts/              Samostalni pipeline skripti
notebooks/            Eksplorativna analiza
dashboard/            Streamlit interaktivni demo
tests/                pytest unit testovi
docs/                 Tehnicka dokumentacija
```

## Instalacija

Preduvjet: Python 3.10 ili noviji. Preporucuje se kreiranje virtualnog okruzenja.

```bash
# Kreirati i aktivirati virtualno okruzenje (Windows)
python -m venv .venv
.venv\Scripts\activate

# Instalirati zavisnosti
pip install -r requirements.txt

# Instalirati paket u razvojnom nacinu
pip install -e .
```

Za tacno reproducibilno okruzenje koristiti:

```bash
pip install -r requirements_lock.txt
pip install -e .
```

## Podaci

Projekat koristi dva skupa podataka koji se ne commituju u repozitorij. Moraju se preuzeti i
raspakirati rucno u odgovarajuce direktorije unutar `data/raw/` prije pokretanja pipeline-a.

### PTB-XL

Preuzeti s PhysioNet-a i raspakirati u:

```
data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
```

Veza: https://physionet.org/content/ptb-xl/

### MIT-BIH NSTDB (šum baza)

Preuzeti s PhysioNet-a i raspakirati u:

```
data/raw/nstdb/
```

Veza: https://physionet.org/content/nstdb/

Nakon raspakiranja, pokrenuti `01_prepare_data.py` koji ce segmentirati i normalizovati PTB-XL
signale i sacuvati ih u `data/processed/`.

## Pokretanje

Skripte se pokrecu redom. Svaki script cita `configs/config.yaml` za putanje i hiperparametre.

| Skript                              | Namjena                                                |
|------------------------------------|-------------------------------------------------------|
| `01_prepare_data.py`               | Ekstrakcija, segmentacija i normalizacija PTB-XL       |
| `02_train.py`                      | Trening primarnog modela V1                           |
| `02_train_v3.py`                   | Trening modela V3 (augmentacija šumom)                |
| `03_evaluate_clean.py`             | Evaluacija V1 na cistom test skupu                    |
| `03_evaluate_clean_v3.py`          | Evaluacija V3 na cistom test skupu                    |
| `04_generate_noisy.py`             | Generisanje 30 šumnih test skupova (5 vrste x 6 SNR)  |
| `05_evaluate_noisy.py`             | Evaluacija V1 na šumnim skupovima                     |
| `05_evaluate_noisy_v3.py`          | Evaluacija V3 na šumnim skupovima                     |
| `06_generate_thesis_figures.py`    | Generisanje svih grafova za tezu                      |
| `07_generate_thesis_tables.py`     | Generisanje LaTeX tabela za tezu                      |
| `07b_generate_comparison_tables.py`| V1 vs V3 poredba (tabele)                             |
| `08_export_thesis_numbers.py`      | Izvoz kljucnih metrika za tekst teze                  |
| `08_generate_comparison_figures.py`| V1 vs V3 poredenje(grafovi)                            |
| `09_sync_thesis_assets.py`         | Kopiranje grafova i tabela u folder teze              |

Poređenje V1 i V3:

```bash
python scripts/compare_v1_v3.py
```

### Multi-seed analiza

Za procjenu varijanse i statisticku poredbu V1 i V3 iz vise nezavisnih treninga:

```bash
python scripts/10_run_multi_seed.py --num-seeds 10 --start-seed 42
python scripts/11_aggregate_runs.py
python scripts/12_aggregate_training_curves.py
```

Objedinjeni rezultati (`outputs/reports/aggregate_v1.json`, `aggregate_v3.json`,
`aggregate_comparison.json`) prikazuju se na stranici "Multi-seed analiza" u dashboardu.


## Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard sadrži sedam stranica:

1. **Pregled signala** - Pregled PTB-XL EKG signala iz test skupa
2. **Laboratorija šuma** - Interaktivna vizualizacija injekcije šuma
3. **Pregled modela** - Arhitektura CNN, broj parametara, V1 vs V3
4. **Rezultati treninga** - Krivulje ucenja, konfuziona matrica, metrike
5. **Robusnost** - Heatmape F1/AUC-ROC po tipu šuma i SNR nivou
6. **Live predikcija** - Interaktivna klasifikacija EKG signala
7. **Multi-seed analiza** - Objedinjeni rezultati iz vise nezavisnih treninga (srednja vrijednost ± standardna devijacija, p-vrijednosti)


Pojedinačni modeli i veći izlazni fileovi nisu dio repozitorija, nego se generišu lokalno.

Glavni rezultati rada zasnovani su na multi-seed analizi i objedinjeni su u fileovima:
- `outputs/reports/aggregate_v1.json`
- `outputs/reports/aggregate_v3.json`
- `outputs/reports/aggregate_comparison.json`

Za svaki pojedinačni run generišu se zasebni izlazi, uključujući najbolji model, finalni model, historiju treniranja i evaluacijske izvještaje. Pojedinačni runovi služe za procjenu varijanse i statističko poređenje, dok su glavni zaključci rada izvedeni iz objedinjene analize svih runova.

