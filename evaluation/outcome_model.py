import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


class OutcomeModel:
    """
    Outcome model for Doubly Robust Policy Evaluation.

    Estimates:
        q_hat(x, a) = E[reward | features, action]

    Supports multiple actions.
    """

    def __init__(self, n_estimators=200, random_state=42):

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )

        self.encoder = LabelEncoder()
        self.feature_cols = None
        self.action_col = None
        self.reward_col = None

    def fit(self, df: pd.DataFrame, feature_cols, action_col="action", reward_col="reward"):
        """
        Train outcome model.

        Parameters
        ----------
        df : dataframe
        feature_cols : list of feature columns
        action_col : column containing actions
        reward_col : reward column
        """

        self.feature_cols = feature_cols
        self.action_col = action_col
        self.reward_col = reward_col

        df = df.copy()

        df["action_encoded"] = self.encoder.fit_transform(df[action_col])

        X = df[feature_cols + ["action_encoded"]]
        y = df[reward_col]

        self.model.fit(X, y)

    def predict(self, df: pd.DataFrame):
        """
        Predict q_hat(x,a) for observed actions.
        """

        df = df.copy()

        df["action_encoded"] = self.encoder.transform(df[self.action_col])

        X = df[self.feature_cols + ["action_encoded"]]

        return self.model.predict(X)

    def predict_for_action(self, df: pd.DataFrame, action: str):
        """
        Predict expected reward if we apply a specific action.
        """

        df = df.copy()

        action_id = self.encoder.transform([action])[0]

        df["action_encoded"] = action_id

        X = df[self.feature_cols + ["action_encoded"]]

        return self.model.predict(X)

    def predict_for_all_actions(self, df: pd.DataFrame):
        """
        Returns expected reward for each action.
        Useful for evaluating AI policies.
        """

        results = {}

        for action in self.encoder.classes_:
            results[action] = self.predict_for_action(df, action)

        return pd.DataFrame(results, index=df.index)
