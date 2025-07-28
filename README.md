# Tesi su autoencoder
Questa tesi tratta di un progetto che prevede l'autenticazione di un autoencoder in grado di codificare tessuti tumorali tramite l'utilizzo di pathway e metapathway.

I file presenti sono i dati e i file python usati per l'implementazione, addestramento ed inferenza del modello.
Gli script possono generare file di stato del modello '.pt', dello scaler per la normalizzazione dei dati '.pkl' e file json per varie metriche.

## Inizializzazione
Tutte le dipendenze sono specificate nel file `deps/requirements.txt` e possono essere installate tramite lo script `deps/install.sh`

## Workflow
Scegliere la tipologia di torch.device modificando `utils.DEFAULT_DEVICE`
Scegliere se avere indici casuali per ogni grid search oppure fissarli in `sweep.py`
Eseguire `sweep.py` per effettuare un training per ogni combinazione di iper-parametri (grid-search) specificati in `sweep.PARAM_GRID`

## Output
Ogni training viene salvato all'interno della directory `sweeps/sweep_####` per poter essere valutato tramite `eval.py`
Nel file `sweeps/results.jsonl` sono presenti tutte le metriche dei diversi training per poter valutare la migliore combinazione di iper-parametri.
Nel file `sweeps/split_indices.json` sono presenti gli indici generati casualmente per suddividere il dataset in training, validation e test set. Ogni training e lo script di valutazione usano lo stesso set di indici.
