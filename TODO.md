# TODO
- [x] Update `templates/identify.html` to add optional in-app camera capture (getUserMedia) and a review/confirm step.
- [x] Keep backend route `/identify` and multipart field name `bird_image` unchanged.
- [x] Add client-side JS for:
  - [x] Starting/stopping camera
  - [x] Capturing a frame to a Blob/File
  - [x] Showing preview + “Confirm & Analyze” submit
  - [x] Handling permission/unsupported errors
- [ ] Verify server-side preview and AI results still render correctly after submit.
- [ ] Run the Flask app and test both upload and camera paths.

- [ ] Add confusion-matrix evaluation pipeline for NIPE predictions
  - [x] Add `metrics.py` (confusion matrix + top misclassifications)
  - [x] Add `confusion_matrix_eval.py` to generate JSON outputs using dataset folder ground truth
  - [ ] Add a Flask route/template to view the generated results



