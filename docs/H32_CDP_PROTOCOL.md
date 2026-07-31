# H32-CDP Clean-Decision-Preserved RLIF Protocol

Status: user-authorized and locked before implementation or H32-CDP results  
Date: 2026-07-20

## 1. Motivation and single permitted change

H32-RLIF is complete and remains rejected as the publication model: it improves all
three registered corruption conditions but reduces clean validation Accuracy at all
three seeds. The user explicitly authorizes one final revision because the robustness,
parameter, and structural benefits may outweigh the small clean loss.

H32-CDP retains the exact H32 architecture, frozen seed-matched B1 parent, optimizer,
seeds, corruption cycle, residual scale, parameter counts, maximum epochs, checkpoint
selection, and all original losses. It makes exactly one scientific change:

> Add clean prediction distillation from the frozen B1 parent to prevent corruption
> training from unnecessarily moving the reliable clean decision boundary.

No module, gate, prototype, adapter, parameter, external data, class weight, or test
access is added. This explicit user authorization supersedes the earlier H32-v2 stop
rule only for this single CDP hypothesis; it does not authorize further variants.

## 2. Locked objective

For clean logits `z`, frozen B1 clean logits `z_B1`, and the same corrupted-view
terms as H32, the loss is:

`L_H32 + 2.0 * KL(softmax(stopgrad(z_B1)) || softmax(z))`.

Temperature is exactly 1.0. The coefficient 2.0 is fixed before implementation and
may not be tuned. H32's original loss remains:

`CE(clean) + 0.5 CE(corrupt) + 0.1 KL(stopgrad(p_clean) || p_corrupt) + 0.01 residual_L2`.

The clean teacher is evaluated inside the existing frozen parent path. Validation is
not used in training, and no test split may be loaded.

## 3. Fixed development protocol

- canonical CrisisMMD Task 2: 5,119 train / 1,097 validation / 1,098 test;
- `openai/clip-vit-large-patch14-336`;
- seed-matched immutable B1 parents;
- seeds 3141, 1729, 2718, in that order;
- AdamW, LR `1e-4`, weight decay `1e-4`;
- physical batch size 1, gradient accumulation 4;
- at most 8 epochs, patience 4;
- first maximum clean validation Accuracy selects the checkpoint;
- fixed missing-image, missing-text, different-class image-mismatch cycle;
- no class weights, label smoothing, CLIP unfreezing, or hyperparameter search.

## 4. Verification gate before training

Required: syntax checks; focused and complete project tests; exact H32 step-zero B1
identity for all four conditions; nonzero CDP/RLIF gradients and zero frozen gradients;
finite positive clean-distillation loss after a controlled nonzero residual; unchanged
517,509 trainable parameters; deterministic different-class mismatch audit; and no
test access. No formal run is authorized before these checks and hash locking.

## 5. Seed-3141 continuation gate

Seed 3141 must satisfy all of the following:

1. clean Accuracy delta versus B1 >= 0;
2. clean Macro F1 delta >= -0.002;
3. clean Weighted F1 delta >= 0;
4. mean corrupted Accuracy delta versus B1 >= 0.05;
5. each corruption Accuracy delta >= 0.01;
6. residual standard deviation >= 0.01;
7. clean-distillation loss is finite and reported;
8. independent audit passes without test access.

Failure closes H32-CDP immediately without replication or redesign.

## 6. Three-seed admission gate

If seed 3141 passes, run both remaining seeds unchanged. Admission requires:

1. mean paired clean Accuracy delta >= 0;
2. at least two of three clean Accuracy deltas >= 0;
3. mean paired clean Macro F1 delta >= -0.001;
4. mean paired clean Weighted F1 delta >= 0;
5. overall mean corruption Accuracy gain >= 0.05;
6. every corruption's mean Accuracy delta >= 0.01;
7. no class loses more than 0.02 mean clean F1;
8. trainable parameters <= 30% of B3 and total parameters < B3;
9. all three audits pass and every seed is reported.

Only after passing all conditions may a separate immutable test lock be proposed.
Final test ensemble Accuracy must be at least 0.91. Test results may not select or
modify the method.

## 7. Final stop rule

There is no CDP-v2, alternative coefficient, altered residual scale, new module,
fourth seed, or H33 under the current authority. Failure returns the paper to B1 as
the formal Accuracy/lightweight anchor and H32 as robustness/negative evidence.
