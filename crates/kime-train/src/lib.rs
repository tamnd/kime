//! Training, distillation and calibration fitting. Outside the inference graph on purpose, so nothing it pulls in reaches the server binary. See spec/12-training.md.

#![forbid(unsafe_code)]
