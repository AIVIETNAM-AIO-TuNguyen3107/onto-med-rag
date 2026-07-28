Implement the evaluation metrics exactly according to the following specification.

Definitions:
- `test` is the test dataset.
- Each `i` is one sample in `test`.
- Each `k` is one candidate field/group in sample `i`.
- `WER(i)` is the Word Error Rate calculated on the `text` field of sample `i`.
- `ground_truth(k)` and `prediction(k)` are the ground-truth and predicted sets for candidate `k`.
- `J_X(i)` is the Jaccard similarity for output field `X` in sample `i`.

Final score:

final_score =
    0.3 * text_score
    + 0.3 * assertions_score
    + 0.4 * candidates_score

1. Text score

Calculate Word Error Rate on the `text` field:

text_score =
    sum(1 - WER(i) for i in test) / len(test)

Do not change the averaging unit: the score is averaged over test samples.

2. Jaccard similarity

For a field X:

- If both the ground-truth set and prediction set are empty:

  J_X(i) = 1

- If the ground-truth set is empty and the prediction set is non-empty:

  J_X(i) = 0

- In all other cases:

  J_X(i) =
      |ground_truth_X(i) ∩ prediction_X(i)|
      / |ground_truth_X(i) ∪ prediction_X(i)|

The last case also produces 0 when the ground-truth set is non-empty
and the prediction set is empty.

Convert values to sets before calculating intersection and union.

3. Assertions score

Assertions are evaluated using Jaccard similarity for the corresponding
disease, medication, and symptom fields.

For each sample, calculate the Jaccard score for each applicable field
and average those values to obtain:

J_assertions(i)

Then calculate:

assertions_score =
    sum(J_assertions(i) for i in test) / len(test)

4. Candidates score

Candidates are evaluated using the same Jaccard definition as assertions.

For each sample, calculate:

J_candidates(i)

The sample weight is:

weight(i) =
    sum(len(ground_truth(k)) + 1 for k in sample i)

Then calculate:

candidates_score =
    sum(
        J_candidates(i) * weight(i)
        for i in test
    )
    /
    sum(
        weight(i)
        for i in test
    )

Equivalently:

candidates_score =
    sum(
        J_candidates(i)
        * sum(len(ground_truth(k)) + 1 for k in i)
        for i in test
    )
    /
    sum(
        sum(len(ground_truth(k)) + 1 for k in i)
        for i in test
    )

5. Wrong concept type

If the predicted concept text is correct but its concept type is wrong,
for example:

- prediction type: CHẨN_ĐOÁN
- ground-truth type: TRIỆU_CHỨNG

then count the concept twice:

- once as the missing ground-truth concept
- once as a newly predicted concept

Both occurrences receive 0 for all three metric categories:

- text
- assertions
- candidates

Do not invent a concept-alignment algorithm unless it already exists in
the codebase or dataset specification. If matching behavior is required
but undefined, clearly identify it as an unresolved requirement instead
of silently choosing an algorithm.

Return:

{
  "text_score": float,
  "assertions_score": float,
  "candidates_score": float,
  "final_score": float
}

Also add unit tests for:
- perfect predictions
- empty ground-truth and prediction sets
- empty ground truth with non-empty prediction
- non-empty ground truth with empty prediction
- partially overlapping sets
- correct concept text but incorrect concept type
- verification of sample-level candidate weighting