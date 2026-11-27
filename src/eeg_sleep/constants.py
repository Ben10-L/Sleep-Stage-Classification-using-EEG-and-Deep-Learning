"""Stage conventions used consistently across the project."""
STAGES = ("W", "N1", "N2", "N3", "REM")
STAGE_TO_INT = {label: index for index, label in enumerate(STAGES)}
INT_TO_STAGE = {v: k for k, v in STAGE_TO_INT.items()}
# The Sleep-EDF annotations include R and both 3 and 4; merge 3/4 to N3.
ANNOTATION_TO_STAGE = {
    "Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
    "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM",
}
EXCLUDED_ANNOTATIONS = {"Sleep stage ?", "Movement time"}
EEG_PICKS = ("EEG Fpz-Cz", "EEG Pz-Oz")
DEFAULT_FS = 100.0
DEFAULT_EPOCH_SECONDS = 30.0
