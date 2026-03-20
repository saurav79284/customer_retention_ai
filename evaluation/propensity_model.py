import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class PropensityModel:
    """
    Estimates P(action | features)
    using multinomial logistic regression.
    """

    def __init__(self):
        self.model = LogisticRegression(
            max_iter=2000,
            multi_class="multinomial"
        )
        self.encoder = LabelEncoder()
        self.feature_cols = None

    def fit(self, df: pd.DataFrame, feature_cols, action_col="action"):

        self.feature_cols = feature_cols

        X = df[feature_cols]

        y = self.encoder.fit_transform(df[action_col])

        self.model.fit(X, y)

    def predict_proba(self, df: pd.DataFrame):

        X = df[self.feature_cols]

        probs = self.model.predict_proba(X)

        actions = self.encoder.inverse_transform(
            range(len(self.encoder.classes_))
        )

        prob_df = pd.DataFrame(
            probs,
            columns=actions,
            index=df.index
        )

        return prob_df

    def propensity_of_observed(self, df: pd.DataFrame, action_col="action"):

        prob_df = self.predict_proba(df)

        prop = prob_df.lookup(df.index, df[action_col])

        return prop
