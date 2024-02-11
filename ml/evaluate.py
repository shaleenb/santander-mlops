import joblib
from sklearn import metrics
import typer
from utils import load_data


def load_model(filepath):
    return joblib.load(filepath)


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    f1_score = metrics.f1_score(y_test, predictions)
    roc_auc = metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return {"f1_score": f1_score, "roc_auc": roc_auc}


def main(
    data_file_path: str = typer.Argument(..., help="Path to the test data file."),
    id_column: str = typer.Argument(default="ID_code", help="The name of the ID column."),
    model_file_path: str = typer.Argument(..., help="Path to the trained model file."),
):
    # Load test data
    df_test = load_data(data_file_path)
    X_test = df_test.drop(["target", id_column], axis=1, errors="ignore")
    y_test = df_test["target"]

    # Load model
    model = load_model(model_file_path)

    metrics = evaluate_model(model, X_test, y_test)

    # Evaluate model
    print("Metrics:")
    print(f"\tF1 score: {metrics['f1_score']:.3f}")
    print(f"\tAUC ROC: {metrics['roc_auc']:.3f}")


if __name__ == "__main__":
    typer.run(main)
