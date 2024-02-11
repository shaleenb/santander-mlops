import joblib
import typer
from feature_engineering import FeatureEngineering
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from evaluate import evaluate_model
from sklearn.pipeline import Pipeline
from utils import load_data


def train_model(X, y):
    features = X.columns.tolist()
    pipeline = Pipeline(
        [
            ("feature_engineering", FeatureEngineering(features=features)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def save_model(model, filepath):
    joblib.dump(model, filepath)


def main(
    data_file_path: str = typer.Option(..., help="Path to the training data file."),
    id_column: str = typer.Option(default="ID_code", help="The name of the ID column."),
    model_file_path: str = typer.Option(..., help="Path to save the trained model."),
):
    df = load_data(data_file_path)
    X = df.drop(["target", id_column], axis=1, errors="ignore")
    y = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    eval_metrics = evaluate_model(model, X_val, y_val)
    print("Metrics on validation data:")
    print(f"\tF1 score: {eval_metrics['f1_score']:.3f}")
    print(f"\tAUC ROC: {eval_metrics['roc_auc']:.3f}")

    joblib.dump(model, model_file_path)
    print(f"Model saved to {model_file_path}")


if __name__ == "__main__":
    typer.run(main)
