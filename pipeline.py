import re
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


class BasketPipeline:
    """Loads transactions, cleans them, groups baskets, and mines Apriori rules."""

    def __init__(self, filepath="transactions.csv", min_support=0.10, min_confidence=0.40, min_lift=1.0):
        # Encapsulated configuration attributes (Week 8 OOP)
        self._filepath = filepath
        self._min_support = min_support
        self._min_confidence = min_confidence
        self._min_lift = min_lift

    @staticmethod
    def clean_product_name(name):
        """Uses RegEx (Week 5) to remove special symbols and extra spaces."""
        text = re.sub(r"[^a-zA-Z0-9\s]", "", str(name))
        return re.sub(r"\s+", " ", text).strip().title()

    def load_and_clean(self):
        """Reads CSV, removes cancelled/empty rows, and groups products by InvoiceNo."""
        df = pd.read_csv(self._filepath).dropna(subset=["InvoiceNo", "Product"])

        # Remove cancelled orders (InvoiceNo starting with 'C')
        df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
        df = df[~df["InvoiceNo"].str.startswith("C")]

        # Clean product names using RegEx
        df["Product"] = df["Product"].apply(self.clean_product_name)
        df = df[df["Product"] != ""]

        # Group into dictionary: {InvoiceNo: {Product1, Product2, ...}}
        baskets = df.groupby("InvoiceNo")["Product"].apply(set).to_dict()
        products = sorted(df["Product"].unique().tolist())
        return baskets, products

    def mine_rules(self, basket_list):
        """Runs Apriori and returns rules filtered by support, confidence, and lift."""
        if not basket_list:
            return pd.DataFrame()

        # One-hot encode baskets into a True/False DataFrame
        encoder = TransactionEncoder()
        matrix = encoder.fit([list(b) for b in basket_list]).transform([list(b) for b in basket_list])
        basket_df = pd.DataFrame(matrix, columns=encoder.columns_)

        # Step 1: Find frequent itemsets with Apriori
        frequent_items = apriori(basket_df, min_support=self._min_support, use_colnames=True)
        if frequent_items.empty:
            return pd.DataFrame()

        # Step 2: Generate association rules and filter by lift
        rules = association_rules(frequent_items, metric="confidence", min_threshold=self._min_confidence)
        rules = rules[rules["lift"] >= self._min_lift]
        return rules[["antecedents", "consequents", "support", "confidence", "lift"]].reset_index(drop=True)