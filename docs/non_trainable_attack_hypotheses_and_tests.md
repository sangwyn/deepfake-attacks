# Non-trainable attack hypotheses and tests

This document contains only hypotheses derived from the reviewed papers and experiments that can confirm or falsify them. “Non-trainable” permits per-image optimization, frozen pretrained models, and statistics from a disjoint calibration set, but excludes fitting surrogate detectors, attack generators, learned filters, or other reusable model parameters.

## H1 — Spectrum-randomized gradients transfer better than spatial gradients alone

**Hypothesis.** Randomly perturbing the spectrum during gradient computation exposes attack directions that are less tied to one detector’s frequency response. A separately normalized fusion of spatial and spectrum-randomized gradients will therefore transfer better than either branch alone.

**Test.** Generate three attacks under the same perturbation and gradient-evaluation budget:

1. spatial gradient only;
2. spectrum-randomized gradient only, using `IDCT(DCT(x + xi) * M)` with seeded noise and masks;
3. equal-weight fusion after normalizing each branch independently.

**Controls.** Use identical images, seeds, iterations, source models, and output constraints. Include an identity spectrum transform to verify recovery of the spatial-only variant.

**Measure.** Cross-model targeted ASR, worst-direction ASR, SSIM, LPIPS, post-save perturbation norms, and gradient cosine similarity between the two branches.

**Support.** The fused attack improves held-out transfer, especially the weaker transfer direction, without a material quality loss.

**Falsification.** Gains occur only on the source detector or disappear when compute and distortion are matched.

## H2 — Transferability is frequency-band-specific, not simply low-frequency

**Hypothesis.** Different detector families share vulnerabilities in particular radial or directional frequency bands; “put perturbations in low frequencies” is too coarse.

**Test.** Partition the DCT plane into low-, mid-, and high-frequency radial bands, followed by orientation-sensitive sub-bands if the radial test is positive. Optimize one attack per band with equal perturbation energy and an unrestricted-frequency control.

**Controls.** Equalize RGB distortion, DCT energy, iterations, and gradient evaluations. Keep the band masks fixed before looking at target results.

**Measure.** A complete source→target transfer matrix, ASR by band, SSIM/LPIPS, robustness after resize and JPEG encoding, and perturbation energy actually retained in each band after saving.

**Support.** One or more bands consistently improve transfer across seeds or detector directions, with confidence intervals excluding negligible differences.

**Falsification.** Band choice has no reproducible effect, or the apparent advantage is explained by unequal distortion or codec survival.

## H3 — Spectrum simulation and block transformations are complementary

**Hypothesis.** Spectrum simulation regularizes frequency dependence, while block shuffle/rotation regularizes local spatial and patch-token dependence. Combining them should bridge detector architectures better than either transformation alone.

**Test.** Compare four matched variants:

1. no spectrum or block transformation;
2. spectrum simulation only;
3. block shuffle/rotation only;
4. spectrum simulation plus block shuffle/rotation.

**Controls.** Use the same optimization objective, perturbation budget, stochastic seed stream, and total gradient evaluations. Verify that identity transformation parameters recover the untransformed variant.

**Measure.** Cross-architecture ASR, worst-target ASR, paired success differences, quality metrics, and the variance of gradients across transformed views.

**Support.** The combined variant beats the stronger individual transformation on held-out targets and reduces target-to-target variance.

**Falsification.** The combination merely averages incompatible gradients, increases distortion, or fails to improve the harder target.

## H4 — Functional and forensic-cue diversity matters more than architecture labels

**Hypothesis.** An ensemble transfers well when its members are accurate and rely on complementary forensic cues, not merely because their backbones have different names.

**Test.** Construct equal-size frozen source sets representing:

- similar architectures and similar input cues;
- different architectures but similar RGB cues;
- different cues, such as RGB, residual/high-pass, and frequency representations.

Use leave-one-detector-out evaluation so the target never contributes to attack generation or selection.

**Controls.** Match ensemble cardinality, source clean accuracy, loss scale, iterations, and compute. Repeat the comparison after removing the weakest clean source.

**Measure.** Held-out ASR, worst-target ASR, pairwise gradient agreement, source calibration, and clean accuracy of every source.

**Support.** Cue-diverse ensembles outperform architecture-diverse RGB ensembles, and adding a clean-weak or functionally mismatched source reduces transfer.

**Falsification.** Architecture labels predict transfer as well as cue diversity, or ensemble composition has no effect after clean accuracy is controlled.

## H5 — Consensus/orthogonal weighting helps only with valid source gradients

**Hypothesis.** Consensus directions capture shared vulnerabilities, while orthogonal directions preserve attack strength on individual models. Their combination helps only when all source gradients are meaningful and at least one detector remains held out.

**Test.** Compare:

1. equal mean gradient;
2. consensus component only;
3. orthogonal components only;
4. fixed consensus/orthogonal weighting;
5. the same variants after adding a lower-accuracy or functionally mismatched source.

**Controls.** Match perturbation budget, integrated-gradient approximation, momentum, and compute. Determine all weights without target feedback.

**Measure.** Leave-one-detector-out ASR, per-target losses, eigenvalue stability of the gradient Gram matrix, component norms, and the effect of each source on the final direction.

**Support.** Combined weighting improves median held-out ASR without materially harming any target, while the weak-source experiment predicts when it should fail.

**Falsification.** Equal averaging is as good, orthogonal components only recover source success, or no held-out detector is available.

## H6 — Optimized natural degradations attack shared forensic statistics

**Hypothesis.** Exposure, blur, and noise become transferable attacks when their strengths are optimized to reduce real/fake statistical differences; arbitrary post-processing does not provide the same effect.

**Test.** Compare:

1. fixed operator strengths sampled from a predeclared grid;
2. per-image optimization of one operator at a time;
3. jointly optimized exposure, blur, and noise;
4. a sequential multi-layer composition.

Reference statistics must come from a disjoint set and remain frozen during evaluation.

**Controls.** Match perceptual distortion and operator ranges. Do not select an operator or strength from target predictions.

**Measure.** Held-out ASR, SSIM/LPIPS, DCT moments, high-pass residual statistics, operator parameters, and success stratified by image type.

**Support.** Optimized degradations outperform the fixed grid at matched quality and reduce the same statistical discrepancy on unseen detectors.

**Falsification.** Performance is indistinguishable from arbitrary post-processing, depends on target feedback, or generalizes only to the optimized source.

## H7 — DCT-statistic alignment and semantic attack gradients solve different parts of the problem

**Hypothesis.** DCT alignment erases low-level synthetic traces, while semantic/spatial gradients move discriminative features. Combining separately normalized objectives can attack a frequency detector without sacrificing transfer to a semantic detector.

**Test.** Compare:

1. DCT-statistic alignment only;
2. semantic/spatial objective only;
3. fixed normalized fusion;
4. alternating DCT and semantic updates.

Estimate target-class DCT statistics from a disjoint calibration set. Predeclare the statistics, bands, and fusion weights.

**Controls.** Match final RGB quality and compute. Include a shuffled-reference-statistics control to test whether real-class alignment, rather than generic spectral change, causes the effect.

**Measure.** Frequency-detector ASR, semantic-detector ASR, worst-target ASR, changes in DCT moments, semantic embedding distance, SSIM, and LPIPS.

**Support.** Fusion improves the worst detector while preserving the benefit of each individual objective.

**Falsification.** The objectives trade one detector against the other, or shuffled statistics work equally well.

## H8 — Detector-independent reconstruction can initialize a stronger attack

**Hypothesis.** A frozen reconstruction operator removes brittle forensic traces, placing the image in a region from which a small per-image optimized perturbation transfers more easily.

**Test.** Compare four variants at matched final perceptual quality:

1. reconstruction only;
2. optimized perturbation only;
3. reconstruction followed by optimized perturbation;
4. optimized perturbation followed by reconstruction.

Use fixed super-resolution or diffusion-restoration strengths; do not choose strength from detector feedback.

**Controls.** Include deterministic downsample/upsample and canonical re-encoding controls. Measure final distortion relative to the original image, not merely relative to the reconstructed intermediate.

**Measure.** Held-out ASR, SSIM/LPIPS, identity or semantic similarity where appropriate, DCT/residual statistics, and optimization iterations needed for success.

**Support.** Reconstruction-first achieves a better ASR–quality trade-off than both components alone and requires fewer attack updates.

**Falsification.** Reconstruction destroys useful gradient directions, offers no benefit beyond resizing/re-encoding, or causes unacceptable content changes.

## H9 — Latent optimization and multi-domain transformations are complementary

**Hypothesis.** DDIM latent optimization searches semantically coherent directions, while spectral and block transformations discourage source-specific solutions. Their combination should improve transfer at a matched perceptual quality point.

**Test.** Compare:

1. DDIM reconstruction only;
2. latent optimization only;
3. latent optimization plus spectrum transforms;
4. latent optimization plus block transforms;
5. latent optimization plus both transform families;
6. the full transform variant with and without source ensembling, only when a detector remains held out.

**Controls.** Freeze the diffusion model, text condition, inversion steps, optimized timestep, latent radius, augmentation distributions, and candidate-selection rule. Never use target predictions for selection.

**Measure.** Held-out ASR along an SSIM/LPIPS Pareto curve, identity/semantic consistency, RGB perturbation norms for description, runtime, and reconstruction-only drift.

**Support.** A transformed latent variant dominates latent-only at one or more predeclared perceptual operating points.

**Falsification.** Apparent gains come entirely from greater distortion, reconstruction drift, or target-informed selection.

## H10 — Shared saliency is more transferable than single-model saliency

**Hypothesis.** Regions important to several detector families contain shared forensic evidence; masks from one detector overfit its attention pattern.

**Test.** Generate masks from frozen source explanations and compare:

1. single-source saliency;
2. intersection of high-saliency regions;
3. union of high-saliency regions;
4. agreement-weighted soft saliency;
5. unrestricted perturbation.

**Controls.** Equalize the effective perturbation energy and perturbed area. Freeze thresholds and mask construction before target evaluation.

**Measure.** Held-out ASR, successful perturbation area, quality metrics, source-to-target mask overlap, and performance on face/background or foreground/background regions.

**Support.** Agreement-weighted or intersection masks improve ASR per perturbed pixel and generalize across targets.

**Falsification.** Single-model masks transfer equally well, or shared masks help only by allowing more perturbation energy.

## H11 — Generator-family-conditioned universal perturbations capture reusable vulnerabilities

**Hypothesis.** GAN and diffusion images contain different reusable attack directions. Separate family-conditioned perturbations should outperform one mixed universal perturbation on unseen images from the same family.

**Test.** Optimize universal perturbations on disjoint calibration data for:

- GAN images;
- diffusion images;
- mixed images.

Evaluate all three perturbations on unseen GAN and diffusion images, including generators excluded from calibration.

**Controls.** Match perturbation norm, calibration-set size, iterations, and source models. Include a random perturbation with identical spectral energy.

**Measure.** Cross-family and within-family ASR, generator-held-out ASR, quality metrics, perturbation spectra, and per-image success overlap with adaptive attacks.

**Support.** Family-conditioned perturbations improve within-family transfer while retaining measurable generator-held-out success.

**Falsification.** The mixed perturbation performs equally well, or family-specific gains vanish on unseen generators.

## H12 — Resynthesis works by erasing forensic traces rather than changing semantics

**Hypothesis.** Super-resolution and diffusion reconstruction primarily change spectral/residual forensic evidence while preserving semantic content.

**Test.** Before and after each resynthesis operator, measure:

- DCT and Fourier statistics;
- high-pass and reconstruction residuals;
- detector pre-logit embeddings;
- frozen semantic embeddings;
- detector decisions.

Compare successful and unsuccessful images at several fixed reconstruction strengths.

**Controls.** Include pixel-matched noise and deterministic resampling controls. Use the same semantic model for every operator and never optimize it.

**Measure.** The association between attack success and changes in forensic versus semantic features, with paired confidence intervals and a simple mediation or partial-correlation analysis.

**Support.** Success tracks large forensic-feature changes while semantic embeddings remain comparatively stable.

**Falsification.** Semantic displacement explains success as well as or better than forensic-trace removal.

## H13 — Codec-aware optimization improves persistence under unseen preprocessing

**Hypothesis.** Optimizing an attack over a distribution of resize kernels, JPEG qualities, chroma subsampling, mild blur, and color conversions produces perturbations that survive unseen digital processing better than attacks optimized on a single decoded image.

**Test.** Compare no transformation, one transformation family at a time, and a mixed codec-aware expectation-over-transformation objective. Evaluate on both seen transformations and held-out qualities/kernels.

**Controls.** Match perturbation budget, gradient evaluations, and source success. Freeze the transformation distribution before testing held-out preprocessing.

**Measure.** ASR before processing, ASR after each processing chain, worst-transform ASR, SSIM/LPIPS after processing, and perturbation spectral survival.

**Support.** Mixed optimization improves worst-transform and held-out-transform ASR without relying on greater distortion.

**Falsification.** Improvements are confined to transformations seen during optimization or disappear under matched compute.

## H14 — Surrogate-only candidate diversity is more useful than one optimization trajectory

**Hypothesis.** Different non-trainable primitives reach complementary solutions. Selecting or mixing their candidates using only surrogate confidence and perceptual quality should transfer better than committing to a single trajectory.

**Test.** Generate a fixed candidate set from spatial, spectral, block, saliency, statistical, and latent primitives as applicable. Compare:

1. best fixed primitive chosen before evaluation;
2. per-image surrogate-only candidate selection;
3. uniform perturbation mixing;
4. constrained surrogate-only mixing with a diversity penalty.

**Controls.** Use the same candidate pool and total generation budget in every selection condition. The target must not influence candidate generation, weights, or selection.

**Measure.** Held-out ASR, selection frequency by primitive, candidate success overlap, quality, mixing entropy, and transfer regret relative to an oracle reported only as a diagnostic.

**Support.** Surrogate-only selection or constrained mixing improves held-out transfer and exploits genuinely complementary candidates.

**Falsification.** Selection overfits surrogate confidence, uniform mixing is equally effective, or the oracle gap remains large.
