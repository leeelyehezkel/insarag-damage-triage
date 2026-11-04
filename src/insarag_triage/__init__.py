"""insarag_triage: ordinal A–F damage triage of single-building crops after earthquakes.

Modules
-------
core             data loading, letterbox transform, backbones, CORN ordinal loss, training (`run`), metrics
analysis         extended metrics, cluster-bootstrap confidence intervals, error galleries, learning curves
pool             cropping buildings from a detection dataset into an unlabelled pool, quality filter
active_learning  query-by-committee batch selection, Roboflow upload/import helpers, progress tracking
leakage          image fingerprints to find the same picture across training and test sets
data             download the public Roboflow dataset versions and assemble them by crop id
inference        load the released model and grade new crops
"""
__version__ = "0.1.0"
GRADES = ["A", "B", "C", "D", "E", "F"]
