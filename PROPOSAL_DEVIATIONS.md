# Deviation from Original Proposal

## Original proposal (as approved)
- **Features (4)**: Packet Size (bytes), Protocol Type (TCP/UDP/ICMP), Source Port Range,
  Flags Present (SYN/ACK/FIN)
- **Data**: 150 network connection sessions from sandbox or mock network logs
- **Model**: Gaussian Naive Bayes, Scikit-Learn, 3 classes (C1/C2/C3)
- **Evaluation**: 80/20 train/test split, Accuracy, Precision, Recall, F1

## What changed, and why
Two additional features were introduced beyond the approved proposal:
- `duration_ms` -- total session/flow duration
- `packet_count` -- total packets exchanged in the session/flow

**Why**: Using only the original 4 features, the model reached 90% accuracy, but Suspicious
(C2) recall was only 0.62 -- scanning traffic was frequently confused with malicious traffic,
because packet size/protocol/port/flags alone don't reliably separate a quick reconnaissance
probe from a real exploit attempt. `duration_ms` and `packet_count` are standard NetFlow/IPFIX-
style flow fields (the kind of features real IDS research datasets like CICIDS2017 use) that
directly target this ambiguity: a scan is characteristically brief and low-volume, while an
active exploit or flood typically is not.

## What stayed the same
Everything else in the approved proposal is unchanged: Gaussian Naive Bayes via Scikit-Learn,
the C1/C2/C3 taxonomy, 150 mock-generated sessions as the base dataset (`generate_data.py`),
an 80/20 train/test split, and Accuracy/Precision/Recall/F1 as the evaluation metrics.
Stratified k-fold cross-validation and a confusion matrix were added as supplementary rigor
on top of the required 80/20 split, not a replacement for it.

## Scope note
Subsequent real-world validation work -- extracting features from captured pcaps and public
IoT malware datasets (IoT-23) to test the model against real traffic instead of only synthetic
test data -- goes beyond what the proposal asked for. That work is documented separately in
`REAL_WORLD_VALIDATION_FINDINGS.md` as exploratory/extension material. It produced genuine
findings (a flow-key bug, a Naive Bayes modeling limitation, and a feature-validity/domain-
shift issue on IoT traffic), but should be presented in the report as extension work, not as
evidence for or against the core 6-feature classifier itself.
