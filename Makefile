PYTHON ?= python
.PHONY: test doctor smoke-maze smoke-sudoku summarize
test:
	$(PYTHON) -m pytest
doctor:
	$(PYTHON) -m transport_reasoning doctor
smoke-maze:
	$(PYTHON) -m transport_reasoning prepare-data --config configs/cpu_smoke_maze.json
	$(PYTHON) -m transport_reasoning train --config configs/cpu_smoke_maze.json
smoke-sudoku:
	$(PYTHON) -m transport_reasoning prepare-data --config configs/cpu_smoke_sudoku.json
	$(PYTHON) -m transport_reasoning train --config configs/cpu_smoke_sudoku.json
summarize:
	$(PYTHON) -m transport_reasoning summarize
