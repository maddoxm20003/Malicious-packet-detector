# Real-World Validation Findings

Notes from validating the trained Naive Bayes classifier against real captured traffic,
rather than only the synthetic test split. Written to be adapted directly into the
report's results/limitations/discussion section.

**Scope note**: this entire document covers exploration beyond the original approved
proposal (see `PROPOSAL_DEVIATIONS.md`) and should be presented in the report as extension
work, not as evidence for or against the core 6-feature classifier itself.

## Background

All accuracy figures reported up to this point (90% -> 100% -> 96.7%) came from the same
synthetic generator (`generate_data.py`), evaluated against a held-out split of that same
synthetic data. That measures whether the model can learn the generator's own assumptions
-- it does not measure whether the model works on real network traffic. To test that, flows
were extracted from a real captured pcap (`LocalWifiMLTest.pcap`), hand-labeled as safe (C1),
converted into the same feature schema via `extract_pcap_features.py`, and scored with the
already-trained model via `evaluate_real_data.py`.

## Finding 1: A flow-key bug inflated the apparent error rate

The first real-data run scored **53.8% accuracy** (14/26 correct) on traffic that was 100%
genuinely safe. Investigating the misclassified rows showed every one of them had an
unusually short duration (<43ms) and very few packets (<=4) -- statistically identical to
the synthetic "scanning" class.

Root cause: the flow key used to group packets into a "session"
(`src_ip, dst_ip, sport, dport, proto`) is directional. A single ordinary request/response
exchange produces packets in both directions, which were being counted as *two separate
flows* instead of one conversation -- each half naturally short and low-packet-count. Nine
of the 26 extracted "flows" in the test pcap were confirmed to be exactly this: reverse-
direction halves of the same connection (e.g. a browser request to `104.18.12.46:80` and
its reply were recorded as two unrelated flows).

**Fix:** canonicalized the flow key (sorted endpoints) so both directions of a conversation
merge into one flow, matching how real NetFlow-style tooling (e.g. CICFlowMeter) defines a
bidirectional flow. This reduced the 26 fragmented flows to the correct **17 real
conversations** and raised real-world accuracy to **58.8%** (10/17 correct).

This is a legitimate engineering bug (in the feature-extraction pipeline, not the model)
and is worth reporting separately from the modeling limitation below -- it's the kind of
defect that only surfaces when the model is tested against real, unfiltered network data.

## Finding 2: Naive Bayes cannot represent a bimodal "safe" class

After the flow-key fix, every remaining misclassification was still a *genuinely safe*
connection that happened to be quick (duration ~10-49ms, 1-7 packets) -- e.g. a fast TLS
handshake or a small API call. The training data's Safe (C1) profile only modeled one mode
of safe traffic: long, chatty sessions (`duration_ms ~ N(900, 500)`,
`packet_count ~ N(70, 40)`). Real safe traffic is not unimodal -- it includes both long
sessions and short, quick ones.

**Attempted fix:** `generate_data.py` was changed so 30% of C1 sessions are drawn from a
second, faster profile (`duration_ms ~ N(40, 20)`, `packet_count ~ N(6, 3)`) meant to
represent this quick-but-safe mode, and the model was retrained.

**Result: no improvement (still 58.8% on the real set).** Inspecting the retrained model's
fitted parameters explains why:

| Class | duration_ms (fitted mean, std) | packet_count (fitted mean, std) |
|---|---|---|
| C1: Safe | 662.9, 495.5 | 55.9, 43.7 |
| C2: Suspicious | 31.8, 17.3 | 3.7, 2.7 |
| C3: Malicious | 200.0, 104.8 | 306.8, 175.1 |

Gaussian Naive Bayes fits **exactly one Gaussian per feature per class** -- it cannot
represent two separate peaks within a class. Mixing a second "quick" mode into C1's true
generating distribution didn't teach the model two peaks; it just widened the single
fitted bell curve (a bigger standard deviation) around roughly the same mean. The result:
a genuinely safe 40ms connection is still judged far more likely to be a scan than safe --
measured directly:

```
P(duration = 40ms | C1) = 0.000365
P(duration = 40ms | C2) = 0.020610
```

**C2 (scanning) is ~56x more likely than C1 (safe) at that duration value under the
model's fitted distributions, even though the sample is genuinely safe.** This is not a
data or feature-engineering problem -- it is a structural limitation of the Naive Bayes
algorithm's per-class independent-Gaussian assumption, and it can't be fixed by adjusting
the synthetic generator alone.

## Takeaways for the report

- Synthetic-only evaluation was systematically overoptimistic: 96.7% (synthetic) vs. 58.8%
  (real, post-bugfix) on this validation set.
- One real gap (Finding 1) was a genuine pipeline bug, fixable and fixed.
- The remaining gap (Finding 2) is a real limitation of Gaussian Naive Bayes itself:
  it assumes each class is unimodal per feature, which real "safe" traffic is not.
  Fixing this properly would require either a model that can represent multi-modal
  per-class distributions (e.g. a Gaussian Mixture-based classifier, or binning
  duration/packet_count into categorical buckets so Naive Bayes' per-category
  probabilities can capture multiple modes instead of one continuous Gaussian), or
  splitting "Safe" into two labeled sub-classes during training.
- Small sample size caveat: this validation set is one 17-flow pcap capture from one
  network at one point in time -- useful for demonstrating the methodology and finding
  real bugs/limitations, but not a statistically powered estimate of real-world accuracy
  on its own.
