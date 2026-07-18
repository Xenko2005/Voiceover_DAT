# Real Datasets for Evaluation

## Recommended Quick Dataset: FLEURS

Use FLEURS first. It is easy to download from Hugging Face and has real speech in English, Japanese, and Vietnamese. The sentences are n-way parallel, so we can use:

- English or Japanese audio as source.
- Vietnamese transcription of the parallel sentence as reference translation.

This is good for:

- Temporal Alignment Error (TAE).
- Real-Time Factor (RTF).
- ChrF against Vietnamese reference.
- Comparing baseline vs Semantic Chunk Buffering + Adaptive Length Constraint.

Prepare English -> Vietnamese:

```powershell
python experiments\prepare_fleurs_dataset.py `
  --source-config en_us `
  --target-config vi_vn `
  --split test `
  --num-items 20 `
  --group-size 5 `
  --output-dir datasets\fleurs_eval
```

Prepare Japanese -> Vietnamese:

```powershell
python experiments\prepare_fleurs_dataset.py `
  --source-config ja_jp `
  --target-config vi_vn `
  --split test `
  --num-items 20 `
  --group-size 5 `
  --output-dir datasets\fleurs_eval
```

The script writes:

- `audio/*.wav`: real grouped source audio.
- `module1_json/*.json`: Module 1-style chunks with timestamps.
- `manifest.csv`: dataset metadata and Vietnamese references.
- `module1_json_files.txt`: paths for batch scripts.

## More Direct EN-VI Speech Translation Dataset: PhoST

PhoST is more directly aligned with the project topic because it is an English-Vietnamese speech translation dataset with triplets:

- English source audio.
- English transcript.
- Vietnamese subtitle/reference.

Use PhoST for final/report-scale experiments if you can afford the download/storage size.

Project page:

```text
https://github.com/VinAIResearch/PhoST
```

Your local PhoST text data path:

```text
D:\datasets\PhoST_subset\text_data
```

Convert PhoST `text_data` into Module 1-style JSON:

```powershell
python experiments\prepare_phost_module1_json.py `
  --text-root "D:\datasets\PhoST_subset\text_data" `
  --split test `
  --output-dir datasets\phost_test_module1 `
  --max-files 10 `
  --max-segments-per-file 20
```

If you later download PhoST audio, add `--audio-root`:

```powershell
python experiments\prepare_phost_module1_json.py `
  --text-root "D:\datasets\PhoST_subset\text_data" `
  --audio-root "D:\datasets\PhoST_subset\audio_data" `
  --split test `
  --output-dir datasets\phost_test_module1 `
  --max-files 10 `
  --max-segments-per-file 20
```

Run baseline on the converted dataset:

```powershell
python experiments\run_baseline_dataset.py `
  --input-dir datasets\phost_test_module1 `
  --output-root experiment_outputs\baseline_phost_test `
  --model qwen3.5:2b `
  --limit 10 `
  --max-chunks 20
```

Run the proposed system on the same converted dataset:

```powershell
python experiments\train_phost_module_optimizers.py `
  --text-root "D:\datasets\PhoST_subset\text_data" `
  --output module_3_adaptive_translation\outputs\phost_optimizer_config_fulltrain.json `
  --train-limit-files 3028 `
  --dev-limit-files 31 `
  --train-sample-rows 327370 `
  --dev-sample-rows 1935

python experiments\run_proposed_dataset.py `
  --input-dir datasets\phost_test_module1 `
  --output-root experiment_outputs\proposed_phost_test `
  --model qwen3.5:2b `
  --limit 10 `
  --max-segments 20 `
  --batch-size 5 `
  --optimizer-config module_3_adaptive_translation\outputs\phost_optimizer_config_fulltrain.json
```

The optimizer config trains/tunes lightweight pipeline modules:

- `length_model`: ridge regression predictor for Vietnamese unit length.
- `recommended_runtime`: tuned `tts_ceiling`, `margin`, `lambda_dur`, and `mu_flu`.
- `semantic_chunking`: tuned Module 2 thresholds from simulated ASR fragmentation.

Aggregate dataset-level metrics:

```powershell
python experiments\compare_dataset_runs.py `
  --baseline-root experiment_outputs\baseline_phost_test `
  --proposed-root experiment_outputs\proposed_phost_test `
  --output-dir evaluation\phost_test_outputs
```

Note: with only `text_data`, this evaluates chunking, translation length control, TTS duration, and alignment using PhoST oracle timestamps. It does not evaluate ASR error. To evaluate the full ASR pipeline, download the PhoST audio files referenced by each `.yaml` file, for example `162.wav`.

## Not Recommended for EN -> VI

CoVoST 2 is excellent, but its official English-source target languages do not include Vietnamese. It is not the best fit for English-to-Vietnamese evaluation.

