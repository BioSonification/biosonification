# Example Compositions

This directory contains example MIDI files for the BioSonification gallery.

## Structure

```
examples/
├── midi/              # MIDI files for examples
├── audio/             # Auto-generated WAV files (converted from MIDI)
└── README.md
```

## Adding New Examples

### Step 1: Place MIDI Files
Place your MIDI files in the `midi/` subdirectory with names matching the `midi_filename` in `web/examples_data.py`.

Example filenames:
- `ecoli_adaptive.mid`
- `yeast_adaptive.mid`
- `drosophila_adaptive.mid`
- etc.

### Step 2: Classify with CRNN Model
Run the classification script to analyze consonance for each example:

```bash
cd /Users/aloha_kuino/Desktop/biosonification_rate_music
source venv/bin/activate
python -m web.classify_examples
```

The script will:
1. ✅ Find all MIDI files in `examples/midi/`
2. 🎵 Classify each one using the CRNN consonance model
3. 📝 Display results with predictions and confidence scores
4. 💾 Save results to `examples/consonance_ratings.json`

### Step 3: Update examples_data.py
Copy the consonance data from the script output and paste it into `web/examples_data.py`:

```python
{
    "id": "ecoli",
    "organism": "E. coli",
    ...
    "consonance": {
        "prediction": "consonant",
        "confidence": 0.87,
        "consonant_score": 0.87,
        "dissonant_score": 0.13
    }
}
```

### Step 4: Test
1. Restart the Flask application
2. Navigate to the Examples page
3. Your examples should now display with consonance ratings! 🎉

## File Naming

The filename stem (without `.mid` extension) is used to match MIDI files with examples:
- `ecoli_adaptive.mid` → filename stem: `ecoli_adaptive`
- `drosophila_long.mid` → filename stem: `drosophila_long`

## Consonance Ratings

Each example will show:
- **Prediction**: CONSONANT or DISSONANT
- **Confidence**: 0-100% likelihood of the prediction
- **Visual Badge**: Color-coded display in the gallery

### Color Coding
- 🟢 **Green**: Consonant composition
- 🔴 **Red**: Dissonant composition

## Requirements

- MIDI files must be valid and readable by `librosa`
- Recommended duration: 5-30 seconds
- All MIDI files in `midi/` directory will be classified

## Troubleshooting

**No MIDI files found:**
- Ensure files are placed in `web/static/examples/midi/`
- Check file extensions are `.mid` (lowercase)

**Classification failed:**
- Verify CRNN model exists at `/diploma/crnn_music_classifier/models/best_crnn.pth`
- Check MIDI file is not corrupted
- Try with a different MIDI file

**Examples don't display in gallery:**
- Verify `consonance` field is properly added to `examples_data.py`
- Check browser console for JavaScript errors
- Restart Flask application

## Example JSON Structure

```json
{
  "id": "ecoli",
  "organism": "E. coli",
  "scientific_name": "Escherichia coli",
  "description": "Bacterial genome, 1800 bp fragment",
  "midi_filename": "ecoli_adaptive.mid",
  "bars": 9,
  "duration_seconds": 18,
  "genome_size": "4.6 MB",
  "icon": "🦠",
  "consonance": {
    "prediction": "consonant",
    "confidence": 0.8723,
    "consonant_score": 0.8723,
    "dissonant_score": 0.1277
  }
}
```
