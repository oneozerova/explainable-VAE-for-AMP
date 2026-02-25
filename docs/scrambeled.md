
#  Scrambled Peptides for the cVAE Project

## Why we use scrambled peptides
**Scrambled peptides are a negative control** to check if the cVAE learns **sequence patterns**, not only **amino-acid composition**.

### Goal
Active AMPs (real antimicrobial peptides):
- Correct amino-acid order → can form an amphipathic α-helix
- High hydrophobic moment (~1.0) → strong membrane interaction

Scrambled peptides (shuffled order):
- Same composition (same charge and overall hydrophobicity)
- Random order → low hydrophobic moment (~0.2) → usually inactive

## Expected result in latent space (t-SNE)
After training, **active** and **scrambled** peptides should form **two clear clusters**.
If they overlap, the model likely learned **composition only**, not sequence structure.

```

t-SNE (latent space):
[Active AMPs]  <---- clear separation ---->  [Scrambled]

````

## Implementation: `scramble_sequence`
Scrambling shuffles amino acids but keeps **length** and **composition**.

```python
def scramble_sequence(seq: str) -> str:
    """Randomly shuffle amino acid order (preserves composition and length)."""
    chars = list(seq)
    random.shuffle(chars)
    return ''.join(chars)
````

### Why this simple Python version

* Same idea as `stri_rand_shuffle` for amino-acid strings (A–Z)
* No extra dependencies (no R / rpy2 / stringi)
* Fast for large datasets (10k+ sequences)


### Success criterion

* **Two separate clusters** → model learned sequence patterns
* **Strong overlap** → model relies on composition → retrain / adjust model


## Conditioning (controlled generation)

Use `is_scrambled ∈ {0,1}` as the condition vector **c** to control generation:

* `0` = active-like sequences
* `1` = scrambled-like sequences
