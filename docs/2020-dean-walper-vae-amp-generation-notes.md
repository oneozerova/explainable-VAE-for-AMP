# Antimicrobial Peptide Generation via Latent Space Interpolation


## Overview

This project is based on the paper:

> ACS Omega (2020), DOI: 10.1021/acsomega.0c00442

Link to the artical: https://pubs.acs.org/doi/10.1021/acsomega.0c00442

The core biological property of antimicrobial peptides (AMPs) is their ability to **destroy bacterial membranes**, leading to bacterial cell death.

The study demonstrates that latent space interpolation in a neural network can generate biologically active antimicrobial peptides while preserving key physicochemical properties.

---

# Core Idea

The model learns a structured latent space representation of peptide sequences.

A special latent direction — called the **Antimicrobial Concept Vector** — represents a continuum from:

```
0.0  → low activity (scrambled / inactive)
0.9+ → high antimicrobial activity
```

By interpolating along this vector, new peptide sequences can be generated with progressively improved antimicrobial properties.

---

# Dataset Construction

### Total samples: **17,838**

```
2971 original peptides × 6 latent points = 17,838
```

Each original peptide generates 6 sequences:

* Active peptide (AMP database)
* Scrambled peptide (control)
* 4 interpolated latent points:

  * 0.00
  * 0.30
  * 0.60
  * 0.90

Each point in latent space decodes into a peptide sequence.

Important:

* Scrambled ≈ close to 0.0
* Active ≈ close to 0.9
* They are **close but not identical**

This means **sequence order determines activity**, not just amino acid composition.

---

# Experimental Validation Example

From 3 known AMP sequences in the database:

Generated:

* p1, p2, p3 (interpolated peptides)
* s1, s2, s3 (scrambled controls)

Lab results:

| Peptide | Result           |
| ------- | ---------------- |
| p1      | Active           |
| p2      | Active           |
| s1      | Inactive         |
| s2      | Inactive         |
| p3      | Active           |
| s3      | Active (anomaly) |

The p3/s3 case was an exception where both were active.

This experimentally confirms:

> Sequence structure (not just composition) determines antimicrobial activity.

---

# Model Architecture & Training

### Architecture

* LSTM with **1024 hidden dimensions**
* Latent space interpolation

### Training setup

* Optimizer: **Adam (Gradient Descent)**
* 5-fold cross-validation
* Training stopped at **250 epochs**

  * Loss stopped decreasing significantly

---

# Key Peptide Properties

The interpolation method was validated by tracking biological properties along the antimicrobial concept vector.

If the method works correctly, moving from 0.0 → 0.9 should improve biological properties.

## 1. Net Charge

* Sum of amino acid charges:

  * K, R → positive
  * D, E → negative
  * H → partial
* Good antimicrobial range:

```
+2 to +6
```

Net charge strongly correlates with activity.

---

## 2. Length & Amino Acid Frequency

* In the paper: peptides ≤ 12 amino acids
* Scrambled and active versions share:

  * Same length
  * Same amino acid frequency

Therefore:

> Composition alone does NOT explain activity.

---

## 3. Hydrophobicity

* Higher hydrophobicity → better membrane disruption
* More hydrophobic peptides → higher antimicrobial activity

---

## 4. Hydrophobic Moment (Amphipathicity)

This is the most important learned property.

Hydrophobic moment measures:

* How well hydrophobic and polar residues separate along a helix.

The clearer the separation:

* One face hydrophobic
* One face polar

→ The better membrane interaction.

### Key Finding:

The model learned **amphipathicity**.

In simple terms:

> It learned where in the sequence the "fat-distribution" regions must be placed to make the peptide active.

---

## 5. Helical Structure

* Higher % of alpha-helix → higher activity
* Secondary structure predicted using **GOR IV**
* Active peptides show stronger helix formation

---

# Why Analyze These Properties?

To validate that:

> Interpolation in latent space produces biologically meaningful transitions.

As we move along the antimicrobial concept vector:

* Net charge improves
* Hydrophobicity increases
* Hydrophobic moment increases
* Helical content increases

This proves the latent space is structured and meaningful.

---

# Classification Validation (CAMPR3)

The authors also tested generated peptides using the **CAMPR3 classifier**.

The classifier outputs:

```
P(AMP) ∈ [0,1]
```

However:

* No generated peptide exceeded 0.5 probability.
* Reason: model bias toward longer sequences.
* Since peptides were short (≤12 aa), classifier underestimates them.

Interestingly:
Some peptides classified as inactive by CAMPR3
→ were active in laboratory experiments.

This shows:

> Experimental validation remains essential.

---

# Discussion & Future Directions

### 1. VAE + Predictor

Authors suggest combining:

* VAE
* Activity predictor

But must avoid bias in predictor.

---

### 2. Conditional VAE (cVAE)

Possible conditioning variables:

* Gram-positive specificity
* Gram-negative specificity
* High hydrophobic moment
* Desired charge range

---

### 3. Longer Sequences

The study focused on short peptides (≤12 aa).

Since the method works:

* It can be extended to longer peptides
* Computational load increases
* But conceptually scalable

---

# Main Scientific Contribution

The most important learned concept:

> The model learned amphipathicity (hydrophobic moment).

It captured structural distribution patterns in sequences that determine membrane-disrupting capability.

This confirms that:

Latent space interpolation can generate functional antimicrobial peptides with biologically interpretable properties.

---

# Key Takeaways

* Sequence order matters more than composition.
* Amphipathicity is the central determinant of activity.
* Latent space has meaningful biological structure.
* Interpolation direction corresponds to real biochemical improvement.
* Experimental validation is critical.
* Method scalable to longer peptides.

---

If you want, I can also:

* Turn this into a polished academic README
* Add diagrams (latent space vector visualization)
* Convert into a project report
* Add mathematical formalization of the antimicrobial concept vector
