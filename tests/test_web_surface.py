from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


TEMPLATE_DIR = Path(__file__).parents[1] / "src" / "perquire" / "web" / "templates"


def render_index() -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml"]),
    )
    return env.get_template("index.html").render()


def test_investigation_surface_exposes_interaction_semantics() -> None:
    html = render_index()

    assert 'aria-label="Toggle navigation"' in html
    assert 'aria-controls="navbarNav"' in html
    assert 'id="manual-tab"' in html and 'aria-controls="manual"' in html
    assert 'id="file-tab"' in html and 'aria-controls="file"' in html
    assert 'id="manual" role="tabpanel" aria-labelledby="manual-tab"' in html
    assert 'id="file" role="tabpanel" aria-labelledby="file-tab"' in html
    assert 'id="errorAlert"' in html and 'role="alert"' in html
    assert 'aria-live="assertive"' in html


def test_validation_and_product_copy_use_the_page_surface() -> None:
    html = render_index()

    assert "alert(" not in html
    assert "showError('Please enter an embedding vector')" in html
    assert "showError('Please select a file')" in html
    assert "revolutionary AI system" not in html
    assert "Investigates unknown embeddings through iterative questions" in html
