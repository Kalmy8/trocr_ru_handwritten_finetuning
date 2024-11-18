import typer
from trocr_russian_handwritten_text.modeling.make_dataset import make_dataset_main
from trocr_russian_handwritten_text.modeling.train import train_main
from trocr_russian_handwritten_text.modeling.predict import predict_main

# Add a global description for your CLI
app = typer.Typer(
    help="A command-line tool for managing your ML pipeline: dataset preparation, model training, and prediction.")


@app.command()
def make_dataset():
    """
    Prepare the dataset for training and evaluation.

    This command handles data preprocessing, such as cleaning and formatting,
    to make it ready for the model training pipeline.
    """
    typer.echo("📂 Starting dataset preparation...")
    make_dataset_main()
    typer.echo("✅ Dataset preparation completed!")


@app.command()
def train():
    """
    Train the machine learning model.

    This command trains your model using the prepared dataset.
    Make sure to run 'make_dataset' first.
    """
    typer.echo("🚀 Starting model training...")
    train_main()
    typer.echo("✅ Model training completed!")


@app.command()
def predict():
    """
    Run predictions using the trained model.

    This command uses your trained model to make predictions
    on new data or test datasets.
    """
    typer.echo("🔍 Starting prediction process...")
    predict_main()
    typer.echo("✅ Predictions completed!")


if __name__ == "__main__":
    app()
