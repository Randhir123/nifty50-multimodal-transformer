# Coding standards

General:

* Python only
* use type hints
* write docstrings for public functions
* keep functions small and testable
* prefer explicit data schemas

Modeling:

* keep models coursework-scale
* do not introduce unnecessary complexity
* add TODOs only when there is a clear next milestone
* design interfaces so text, image, tabular, and KG branches can be trained separately and fused later

Data:

* assume CSV-based local datasets unless instructed otherwise
* never silently fill critical missing values without comments
* preserve date ordering
* avoid leakage from future data

Repo hygiene:

* keep README updated after each milestone
* prefer PR-sized logical changes
* avoid placeholder code that claims to implement missing functionality
