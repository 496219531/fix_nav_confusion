PYTHON ?= python3

.PHONY: help install regression report report-rerun

help:
	@echo "Available targets:"
	@echo "  make install       Install Python dependencies"
	@echo "  make regression    Run key case regression checks"
	@echo "  make report        Read current outputs and print normalized summary"
	@echo "  make report-rerun  Rerun full repair, then print normalized summary"

install:
	$(PYTHON) -m pip install -r requirements.txt

regression:
	$(PYTHON) scripts/check_regressions.py

report:
	$(PYTHON) scripts/run_full_and_report.py

report-rerun:
	$(PYTHON) scripts/run_full_and_report.py --rerun
