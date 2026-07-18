# Multimodal Data Processing

Formative 2: a user identity and product recommendation system. A face unlocks the right to attempt
a prediction, a voice confirms it, and only then is a product recommended.

Team: Michael (lead), Taps, Anthony, Tedla.

## Setup

```
python3 -m pip install --only-binary=:all: -r requirements.txt
```

`--only-binary=:all:` is required, not a preference. Without it pip tries to compile llvmlite
(a librosa dependency) from source and the install fails unless you have an LLVM toolchain.

## Data collection

The loaders depend on these names exactly. `<member>` is a lowercase first name: `michael`, `taps`,
`anthony`, `tedla`.

```
data/raw/images/<member>_<neutral|smiling|surprised>.jpg    3 per member
data/raw/audio/<member>_<approve|confirm>.wav               2 per member
```

The phrases are "Yes, approve" and "Confirm transaction". Record somewhere quiet, and keep lighting
and framing consistent across photos.

**The fifth identity.** A non-member, collected exactly like a member, under the name `unknown`:

```
data/raw/images/unknown_<neutral|smiling|surprised>.jpg     3, not 1
data/raw/audio/unknown_<approve|confirm>.wav                2
```

`unknown` is trained as a real class, which is what makes an intruder get rejected deterministically
rather than by luck of the confidence score. Three photos is not padding, it is load bearing:

| unknown photos | result |
|---|---|
| 1 | training **fails**, a CV fold ends up with no unknown example to train on |
| 2 | works, but silently drops the face model from 3-fold to 2-fold CV |
| 3 | works at full 3-fold, model returns `unknown` and the gate denies it |

The fold count follows the least represented identity, so short-changing the intruder degrades the
evaluation of every member.

## Running it

Train all three models from the feature CSVs:

```
python3 -m src.train
```

Face and voice train from `data/features/`. The product model trains from
`data/processed/merged_dataset.csv` if it exists, and is skipped with a warning if it does not, so
the biometric half stays usable while the merge is outstanding.

Run one transaction:

```
python3 -m src.cli.app --face data/raw/images/taps_neutral.jpg \
                       --voice data/raw/audio/taps_approve.wav
```

Exit codes: `0` granted, `1` denied, `2` a file was not found.

Run the tests:

```
python3 -m pytest tests/ -q
```

## Layout

```
data/
  raw/{images,audio,tabular}/   collected media and the source datasets
  features/                     image_features.csv, audio_features.csv
  processed/                    merged_dataset.csv
  models/                       trained .joblib files
src/
  config.py                     shared constants, import from here
  schemas.py                    frozen CSV contracts and validators
  images/extract.py             image feature extraction (Taps)
  audio/extract.py              audio feature extraction (Tedla)
  models/
    decision.py                 the multimodal gate
    biometric.py                face and voice models
    recommender.py              product model
    registry.py                 member to customer_id mapping
  cli/app.py                    the system simulation
  train.py                      trains and evaluates all three models
tests/
```

## How the three models fit together

| model | input | output |
|---|---|---|
| face recognition | 388 image features | which member, and a confidence |
| voiceprint verification | 34 audio features | which member, and a confidence |
| product recommendation | merged tabular row | which product, and a confidence |

`src/models/decision.py` composes them. Face first; if it fails, the flow stops and never asks for
a voice. Then the voice, which must clear its own threshold **and** identify the same person the
face did.

That last check is the point of the design. Two independent gates would accept any known face plus
any known voice, so Taps's face with Tedla's voice would unlock Taps's recommendations. Requiring
both modalities to agree is what makes the decision multimodal rather than two unimodal checks in
sequence.

Only after both pass does the identity registry resolve the member to a `customer_id`, which is the
only input the product model gets. No denied path ever reaches the recommender or returns a
`customer_id`.

## Design decisions worth knowing

**The tabular data and the biometric data are not connected, so we declare the link.**
`customer_social_profiles` and `customer_transactions` describe anonymous customers. The faces and
voices are us. Nothing joins them. `src/models/registry.py` maps each member to one `customer_id`,
which is what makes authenticating a person useful: recognising Taps only matters if it lets us
recommend a product for Taps specifically. The mapping is assigned in sorted order, not randomly,
so the demo and the report's worked example stay reproducible.

**Every evaluation is grouped, and the numbers are lower because of it.**
Augmentations of one photo are not independent samples. If some land in train and others in test,
the model is scored on recognising a photograph it has already seen rather than recognising a
person, which reads near perfect and means nothing. Face and voice group by `source_file`; the
product model groups by `customer_id` for the same reason, in case a customer has several
transactions. Expect modest accuracy. It is honest.

**The intruder is a trained class, not a threshold accident.**
The models know four members. Without an `unknown` class, a stranger's face resolves to whichever
member it resembles most, and only the confidence floor stands in the way. So the fifth identity is
collected exactly like a member (3 photos, 2 clips) and trained as a real class, and the gate denies
it deterministically. The confidence floors remain as a second line of defence.

**Bad feature data refuses to train.**
A failed extractor still writes a well formed CSV full of zeros. `schemas.py` rejects all-zero
rows, NaNs, renamed or reordered columns, and unrecognised member, expression or phrase labels, and
`train.py` will not persist a model built on data that fails those checks.

## Contracts between workstreams

Three files are the interfaces the four tracks agree on. Change them only after talking to Michael.

- **`schemas.IMAGE_COLUMNS`** (392): 4 metadata columns, 64 histogram bins, 324 HOG features.
- **`schemas.AUDIO_COLUMNS`** (38): 4 metadata columns, 13 MFCC means and 13 stds, then spectral
  roll-off, centroid, RMS energy and zero crossing rate as mean and std.
- **`schemas.validate_merged`**: the merge must produce a `customer_id` column and a
  `product_category` column. Everything else it carries becomes a model feature automatically,
  including text columns, which get one-hot encoded. The two sources share no key, so the join
  strips the `A` prefix from `customer_id_new` and matches it against `customer_id_legacy`; see
  `src/tabular/merge.py` for the reasoning and the fan-out it avoids.

`src/images/extract.py` and `src/audio/extract.py` each hold one function to implement. They are
called both to build the feature CSVs and by the CLI at demo time to featurise a photo or clip the
system has never seen, which is why they must be callable functions rather than notebook cells.
