"""Unit tests for SnakeClassifier inference engine and uncertainty gating."""

from pathlib import Path
from PIL import Image
import pytest

from src.predict import SnakeClassifier, load_and_preprocess_pil


@pytest.fixture(scope="module")
def classifier() -> SnakeClassifier:
    """Fixture providing initialized SnakeClassifier using best.pt."""
    ckpt_path = Path("outputs/checkpoints/best.pt")
    if not ckpt_path.is_file():
        pytest.skip("outputs/checkpoints/best.pt not present, skipping inference tests.")
    return SnakeClassifier(checkpoint_path=ckpt_path)


def test_predict_dict_keys_and_probabilities(classifier: SnakeClassifier) -> None:
    """Verify inference output dictionary structure and probability normalization."""
    img = Image.new("RGB", (224, 224), color=(180, 50, 30))
    res = classifier.predict(img)

    required_keys = {"label", "raw_label", "confidence", "uncertain", "uncertainty_threshold", "probabilities"}
    assert required_keys.issubset(res.keys())

    probs = res["probabilities"]
    assert "venomous" in probs
    assert "non_venomous" in probs
    assert probs["venomous"] + probs["non_venomous"] == pytest.approx(1.0, rel=1e-4)

    assert res["confidence"] == pytest.approx(max(probs["venomous"], probs["non_venomous"]), rel=1e-4)
    assert res["label"] in ["venomous", "non_venomous", "uncertain"]


def test_uncertainty_threshold_logic(classifier: SnakeClassifier) -> None:
    """Verify uncertainty gating flags predictions when confidence is below threshold."""
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))

    # Strict threshold: confidence will be below 0.999 -> label must be 'uncertain'
    res_uncertain = classifier.predict(img, uncertainty_threshold=0.999)
    assert res_uncertain["uncertain"] is True
    assert res_uncertain["label"] == "uncertain"
    assert res_uncertain["raw_label"] in ["venomous", "non_venomous"]

    # Lenient threshold: confidence >= 0.500 -> label must be definitive
    res_definitive = classifier.predict(img, uncertainty_threshold=0.500)
    assert res_definitive["uncertain"] is False
    assert res_definitive["label"] in ["venomous", "non_venomous"]
    assert res_definitive["label"] == res_definitive["raw_label"]


def test_non_rgb_image_modes(classifier: SnakeClassifier) -> None:
    """Verify classifier correctly processes RGBA, Grayscale (L), and Palette (P) image formats."""
    # RGBA (4 channels)
    rgba_img = Image.new("RGBA", (224, 224), color=(200, 100, 50, 255))
    res_rgba = classifier.predict(rgba_img)
    assert res_rgba["label"] in ["venomous", "non_venomous", "uncertain"]

    # Grayscale ('L')
    gray_img = Image.new("L", (224, 224), color=128)
    res_gray = classifier.predict(gray_img)
    assert res_gray["label"] in ["venomous", "non_venomous", "uncertain"]

    # Palette ('P')
    pal_img = Image.new("P", (224, 224), color=1)
    res_pal = classifier.predict(pal_img)
    assert res_pal["label"] in ["venomous", "non_venomous", "uncertain"]


def test_predict_batch(classifier: SnakeClassifier) -> None:
    """Verify batched inference processes lists of images correctly."""
    imgs = [
        Image.new("RGB", (224, 224), color=(255, 0, 0)),
        Image.new("RGB", (224, 224), color=(0, 255, 0)),
        Image.new("RGB", (224, 224), color=(0, 0, 255)),
    ]
    results = classifier.predict_batch(imgs, batch_size=2)
    assert len(results) == 3
    for r in results:
        assert r["label"] in ["venomous", "non_venomous", "uncertain"]
        assert r["probabilities"]["venomous"] + r["probabilities"]["non_venomous"] == pytest.approx(1.0, rel=1e-4)


def test_invalid_image_source_raises() -> None:
    """Verify load_and_preprocess_pil raises appropriate errors on missing or invalid files."""
    with pytest.raises(FileNotFoundError):
        load_and_preprocess_pil("non_existent_image_file.jpg")

    with pytest.raises(TypeError):
        load_and_preprocess_pil(12345)  # type: ignore
