# How enunciate works

This page explains the pipeline in more depth than the README and records the
reasoning behind the less obvious choices.

## 1. Audio

`enunciate.audio.load_audio` decodes a path, bytes or file object with
libsndfile (WAV, FLAC, OGG, MP3), averages channels to mono and resamples to
16 kHz with `soxr`. Anything libsndfile rejects is handed to `ffmpeg` when it is
on the PATH, so the command line also accepts m4a or WebM. The browser demo
converts recordings to 16 kHz WAV before upload with the Web Audio API, so a
server never needs ffmpeg.

## 2. Phoneme recognition

`enunciate.recognizer.Recognizer` wraps
[`facebook/wav2vec2-lv-60-espeak-cv-ft`](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft),
a 300M-parameter wav2vec 2.0 model fine-tuned with CTC to output espeak-ng
phoneme symbols (Xu et al., 2021). The important property is what it is not: it
is not a speech-to-text system. There is no language model or lexicon, so a
clearly mispronounced word is transcribed as the sounds that were made. A
speech-to-text system would do the opposite and silently repair "I lub India"
into "I love India", which is why comparing a transcript with the target text
cannot measure pronunciation.

Decoding is greedy CTC: take the best symbol per 20 ms frame, merge repeats and
drop blanks. The first and last frame of each merged run give the phoneme a time
span, which the API returns with every heard phoneme.

The model is loaded lazily, once per process, behind a lock; inference also
takes the lock, so one process scores one recording at a time. `warm_up()` runs
a short forward pass so that the first real request does not pay the
one-time cost of kernel initialisation.

## 3. Reference pronunciation

`enunciate.g2p.Phonemizer` runs `espeak-ng -q --ipa --sep=_ -v en-us --stdin`
as a subprocess and parses the output. espeak-ng is the grapheme-to-phoneme
system that produced the model's training labels, so both sides of the
comparison share one symbol set. Calling the binary directly, rather than
through the `phonemizer` package, keeps the project MIT-licensed; `phonemizer`
is GPL-3.

Three things about espeak-ng's output had to be handled:

- **A trailing newline is mandatory.** espeak-ng 1.52 mangles a final word that
  is not followed by whitespace ("bath" came back as "bat").
- **Context matters.** Phonemising a sentence at once keeps forms such as "the"
  → ðɪ before a vowel and ðə before a consonant, and "a" → ɐ. Phonemising
  word by word gives citation forms (ðɪ, eɪ) that nobody uses in a sentence.
- **Word counts do not always line up.** espeak-ng glues some unstressed pairs
  into one output word ("of the" → ʌvðə, "in the" → ɪnðə) and expands numbers
  and abbreviations into several. When the output word count differs from the
  token count, the tokens are bisected, preferring a cut at clause punctuation,
  and each half is phonemised again until every piece lines up. Context is lost
  only next to the cut. On the LibriSpeech evaluation set this happens in 28 of
  73 sentences, always because of a merge.

The article "a" is overridden to ɐ because espeak-ng gives the letter name eɪ
whenever the word ends a clause.

## 4. Symbol normalisation

espeak-ng writes an r-coloured vowel as one symbol before a consonant ("more" →
ɔːɹ) but as vowel + ɹ before a vowel ("more of" → ɔː ɹ), while the model nearly
always emits the fused symbol. Both sides are rewritten as vowel + ɹ before
alignment (`enunciate.variants.split_rhotics`). This is a representation
change, not leniency, and it is applied in strict mode too.

## 5. Alignment

`enunciate.align.align` is a Levenshtein alignment with a backtrace. The costs
are pluggable: a substitution cost over symbol pairs and gap costs that see the
expected symbol before the gap. Ties are resolved towards matches, then
deletions, then insertions, so an extra sound is attached after the phoneme it
follows. The path is a list of steps, each with a penalty of 0 or 1.

## 6. Accepted variants

Running the recogniser over 73 LibriSpeech dev-clean utterances and tabulating
expected-to-heard confusions (`scripts/evaluate.py`) showed that most of the
disagreement was not mispronunciation but regular features of spoken General
American English, or two spellings of one sound. `enunciate.variants.ENGLISH_RULES`
turns those into zero-penalty steps:

| Rule | Why |
| --- | --- |
| ɔ, ɔː, ɑ, ɑː, ɒ, oː are one class | cot–caught merger; most Americans use one low back vowel for "thought", "on", "dog", "father" |
| ɔɪ ≈ oɪ | two spellings of the "oy" diphthong |
| t ≈ ɾ, d ≈ ɾ | flapping between vowels ("water", "that is"), which espeak-ng applies inconsistently |
| ᵻ ≈ ɪ, ᵻ ≈ ə, ə ≈ ɐ | espeak-ng's reduced-vowel symbols against the model's |
| ɜː ≈ ɜ ≈ ɚ | the r-coloured central vowel; stress is not scored |
| i ≈ iː | the final vowel of "happy" (ɪ stays distinct: "ship" is not "sheep") |
| əl ≈ l̩ ≈ l, n̩ ≈ ən ≈ n | syllabic consonants with or without a schwa |
| ɹ after ɚ, ɜː or ɹ may be extra or missing | linking r ("weather is" → wɛðɚ ɹɪz) and double r at word joins |

Equivalence is per class and not transitive: ᵻ is a variant of both ɪ and ə,
but ɪ and ə remain different sounds. Accepted differences are still reported,
flagged `accepted: true`, so a learner or a client can see them; they simply do
not count against the score. `--strict` or `ENUNCIATE_ACCEPT_VARIANTS=0`
switches the rules off.

## 7. Scoring

Every expected phoneme belongs to one word. Insertions are charged to the word
whose phoneme precedes them, or to the first word when they come before any
expected phoneme. A word's score is `1 - errors / expected phonemes`, floored at
zero, where errors are steps with a penalty. The overall score is
`1 - phoneme error rate` over the whole sentence. Words whose score is at least
0.8 are shown as good, 0.5 to 0.8 as partial and below 0.5 as poor; those
thresholds live only in the command line and the demo page.

## What this is not

- It does not judge fluency, rhythm or intonation.
- It measures distance from espeak-ng's dictionary pronunciation. Reduced
  function words ("his" without its h, "to" as tə) and British or Indian English
  realisations of vowels that the variant rules do not cover are flagged even
  though they are not errors.
- The recogniser's own errors are in the score. The evaluation page in the
  README gives the scale: about 5 phonemes in 100 disagree on clean native
  speech even with the variant rules.
