
.PHONY: install run-api clean

install:
	pip install -e .

run-api:
	uvicorn app.main:app --reload --app-dir src --host 0.0.0.0 --port 8000

run-cli:
	python3 scripts/classify_folder.py data/raw

clean:
	rm -rf build dist *.egg-info src/*.egg-info
