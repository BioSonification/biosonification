"""Classify consonance for example MIDI files and update examples_data.py"""

import json
import sys
from pathlib import Path

# Add diploma to path for crnn_music_classifier imports
# Use a known absolute path instead of __file__ which can be unreliable
BIOSONIFICATION_ROOT = Path("/Users/aloha_kuino/Desktop/biosonification_rate_music").resolve()
DIPLOMA_ROOT = Path("/Users/aloha_kuino/Desktop/diploma").resolve()

if str(DIPLOMA_ROOT) not in sys.path:
    sys.path.insert(0, str(DIPLOMA_ROOT))

PROJECT_ROOT = BIOSONIFICATION_ROOT

from web.consonance_classifier import get_classifier

EXAMPLES_MIDI_DIR = PROJECT_ROOT / "web" / "static" / "examples" / "midi"
EXAMPLES_DATA_FILE = PROJECT_ROOT / "web" / "examples_data.py"


def classify_examples():
    """Classify all MIDI files in examples directory and update examples_data.py"""

    # Ensure directory exists
    EXAMPLES_MIDI_DIR.mkdir(parents=True, exist_ok=True)

    classifier = get_classifier()
    if not classifier.is_ready():
        print(f"❌ Классификатор не готов: {classifier.get_error()}")
        return False

    # Find all MIDI files in examples directory
    midi_files = sorted(EXAMPLES_MIDI_DIR.glob("*.mid"))

    if not midi_files:
        print(f"⚠️  MIDI файлы не найдены в {EXAMPLES_MIDI_DIR}")
        print("   Поместите MIDI файлы в эту папку и запустите скрипт снова")
        return False

    print(f"🎵 Найдено {len(midi_files)} MIDI файлов")
    print()

    # Classify each file
    results = {}
    for midi_path in midi_files:
        print(f"Классифицирую: {midi_path.name}...", end=" ")
        result = classifier.classify(str(midi_path))

        if result["success"]:
            results[midi_path.stem] = {
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "scores": result["scores"],
            }
            confidence_pct = round(result["confidence"] * 100)
            print(f"✅ {result['prediction'].upper()} ({confidence_pct}%)")
        else:
            print(f"❌ Ошибка: {result['error']}")
            return False

    print()
    print("=" * 60)
    print("📊 Результаты классификации:")
    print("=" * 60)
    for filename, data in results.items():
        print(f"\n{filename}:")
        print(f"  Предсказание: {data['prediction'].upper()}")
        print(f"  Уверенность:  {data['confidence']*100:.1f}%")
        print(f"  Консонансность: {data['scores']['consonant']*100:.1f}%")
        print(f"  Диссонансность: {data['scores']['dissonant']*100:.1f}%")

    # Save results to JSON for reference
    json_output = EXAMPLES_MIDI_DIR.parent / "consonance_ratings.json"
    with open(json_output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Результаты сохранены в {json_output}")

    return results


def generate_example_code(results):
    """Generate Python code snippet for examples_data.py"""
    print("\n" + "=" * 60)
    print("📝 Добавьте консонансность к примерам в examples_data.py:")
    print("=" * 60)
    print("\nДобавьте эти поля к каждому примеру:")
    print("""
    "consonance": {
        "prediction": "consonant" or "dissonant",
        "confidence": 0.87,
        "consonant_score": 0.87,
        "dissonant_score": 0.13
    }
    """)

    print("\nПримеры для копирования:")
    for filename, data in results.items():
        print(f'\n    # {filename}')
        print(f'    "consonance": {{')
        print(f'        "prediction": "{data["prediction"]}",')
        print(f'        "confidence": {data["confidence"]:.4f},')
        print(f'        "consonant_score": {data["scores"]["consonant"]:.4f},')
        print(f'        "dissonant_score": {data["scores"]["dissonant"]:.4f}')
        print(f'    }}')


if __name__ == "__main__":
    results = classify_examples()
    if results:
        generate_example_code(results)
