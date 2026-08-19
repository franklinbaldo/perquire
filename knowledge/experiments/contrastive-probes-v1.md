---
type: Experiment Protocol
title: Contrastive semantic probes v1
description: Conditional experiment testing whether relative scores across deliberately contrasting semantic probes improve search under the frozen benchmark v1.
status: conditional
experiment_id: contrastive-probes-v1
claim_id: adaptive-feedback-v1
depends_on: semantic-inversion-benchmark-v1
---

# Contrastive semantic probes v1

This experiment is activated only if benchmark v1 identifies adaptive-feedback signal worth improving or a failure mode that contrastive evidence can specifically address.

A target vector does not answer natural-language questions. A scored probe is therefore treated only as a proximity observation. Each contrastive set contains alternative semantic descriptions whose relative scores form evidence for the next proposal step.

The experiment must use the frozen benchmark-v1 cases and budget accounting. Success requires improvement over the best simpler benchmark method on preregistered metrics; an aesthetically plausible trajectory is not sufficient.
