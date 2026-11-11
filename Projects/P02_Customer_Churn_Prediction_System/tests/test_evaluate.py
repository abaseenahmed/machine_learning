import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluate import align_features, load_evaluation_data


class EvaluationDataTests(unittest.TestCase):
    def test_align_features_matches_trained_order(self):
        features = pd.DataFrame({"second": [2], "first": [1]})

        aligned = align_features(features, ["first", "second"])

        self.assertEqual(aligned.columns.tolist(), ["first", "second"])
        self.assertEqual(aligned.iloc[0].tolist(), [1, 2])

    def test_align_features_rejects_schema_mismatches(self):
        cases = [
            (
                pd.DataFrame({"first": [1]}),
                ["first", "second"],
                "missing columns",
            ),
            (
                pd.DataFrame({"first": [1], "extra": [2]}),
                ["first"],
                "unexpected columns",
            ),
            (pd.DataFrame({"first": [1]}), ["first", "first"], "unique"),
        ]

        for features, expected_columns, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    align_features(features, expected_columns)

    def test_load_evaluation_data_drops_id_and_rejects_invalid_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory) / "customers.csv"
            pd.DataFrame(
                {
                    "customerID": ["a", "b"],
                    "second": [2, 4],
                    "first": [1, 3],
                    "Churn": ["No", "Yes"],
                }
            ).to_csv(data_path, index=False)

            features, labels = load_evaluation_data(data_path, ["first", "second"])

            self.assertEqual(features.columns.tolist(), ["first", "second"])
            self.assertEqual(labels.tolist(), [0, 1])

            pd.DataFrame(
                {"first": [1], "second": [2], "Churn": ["Maybe"]}
            ).to_csv(data_path, index=False)
            with self.assertRaisesRegex(ValueError, "Unexpected Churn values"):
                load_evaluation_data(data_path, ["first", "second"])


if __name__ == "__main__":
    unittest.main()
