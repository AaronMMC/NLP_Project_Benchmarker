.PHONY: all install test benchmark benchmark-fast clean

all: install benchmark

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

benchmark:
	python -m src.run_benchmark --config config.yaml

benchmark-fast:
	python -m src.run_benchmark --config config.yaml --fast

clean:
	rm -rf results/metrics/* results/figures/* results/errors/* logs/*
