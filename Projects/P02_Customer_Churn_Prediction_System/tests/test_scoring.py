import unittest

import pandas as pd

from scoring import align_customer_features


class CustomerScoringTests(unittest.TestCase):
    def test_aligns_feature_order_and_preserves_optional_customer_ids(self):
        customers = pd.DataFrame(
            {
                "customerID": ["a", "b"],
                "second": [2, 4],
                "first": [1, 3],
            }
        )

        features, customer_ids = align_customer_features(
            customers, ["first", "second"]
        )

        self.assertEqual(features.columns.tolist(), ["first", "second"])
        self.assertEqual(features.iloc[0].tolist(), [1, 2])
        self.assertEqual(customer_ids.tolist(), ["a", "b"])

    def test_aligns_features_without_customer_ids(self):
        customers = pd.DataFrame({"first": [1], "second": [2]})

        features, customer_ids = align_customer_features(
            customers, ["first", "second"]
        )

        self.assertEqual(features.columns.tolist(), ["first", "second"])
        self.assertIsNone(customer_ids)

    def test_rejects_missing_extra_and_empty_customer_data(self):
        cases = [
            (pd.DataFrame({"first": [1]}), ["first", "second"], "missing columns"),
            (
                pd.DataFrame({"first": [1], "second": [2], "Churn": ["Yes"]}),
                ["first", "second"],
                "unexpected columns",
            ),
            (pd.DataFrame(columns=["first", "second"]), ["first", "second"], "at least one row"),
        ]

        for customers, expected_columns, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    align_customer_features(customers, expected_columns)


if __name__ == "__main__":
    unittest.main()
