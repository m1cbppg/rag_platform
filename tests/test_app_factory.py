import os
import subprocess
import sys


def test_importing_app_does_not_connect_to_external_infrastructure() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from src.rag_platform.main import create_app; "
                "app = create_app(); "
                "print(app.title)"
            ),
        ],
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
