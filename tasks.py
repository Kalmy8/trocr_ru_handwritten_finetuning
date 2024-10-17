from invoke import task, Collection
import os
import shutil

PROJECT_NAME = "TrOCR_custom_training"
PYTHON_VERSION = "3.10"
PYTHON_INTERPRETER = "python"
MODULE_NAME = "trocr_russian_handwritten_text"
DEPENDENCY_FILE = "environment.yml"
ENVIRONMENT_MANAGER = "mamba"
DATASET_STORAGE = {"none": "none"}


@task
def requirements(ctx):
    """Install Python Dependencies"""
    if DEPENDENCY_FILE == "requirements.txt":
        ctx.run(f"{PYTHON_INTERPRETER} -m pip install -U pip")
        ctx.run(f"{PYTHON_INTERPRETER} -m pip install -r requirements.txt")
    elif DEPENDENCY_FILE == "environment.yml":
        ctx.run(f"conda env update --name {PROJECT_NAME} --file environment.yml --prune")
    elif DEPENDENCY_FILE == "Pipfile":
        ctx.run("pipenv install")

@task
def clean(ctx):
    """Delete all compiled Python files"""
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".pyc") or file.endswith(".pyo"):
                os.remove(os.path.join(root, file))
        for dir in dirs:
            if dir == "__pycache__":
                shutil.rmtree(os.path.join(root, dir))
                
@task
def lint(ctx):
    print("Lint using flake8 and black (use `invoke format` to do formatting)")
    ctx.run(f"flake8 {MODULE_NAME}")
    ctx.run(f"isort --check --diff --profile black {MODULE_NAME}")
    ctx.run(f"black --check --config pyproject.toml {MODULE_NAME}")


@task
def format(ctx):
    print("Format source code with isort and black")
    ctx.run(f"isort --profile black {MODULE_NAME}")
    ctx.run(f"black --config pyproject.toml {MODULE_NAME}")

@task
def sync_data_down(ctx):
    """Download Data from storage system"""
    if DATASET_STORAGE:
        if "none" in DATASET_STORAGE:
            print("No DATASET_STORAGE found. Please ensure the dataset is available.")

        elif "s3" in DATASET_STORAGE:
            bucket = DATASET_STORAGE["s3"].get("bucket", "")
            profile = DATASET_STORAGE["s3"].get("aws_profile", "default")
            profile_option = f" --profile {profile}" if profile != "default" else ""
            ctx.run(f"aws s3 sync s3://{bucket}/data/ data/{profile_option}")
        elif "azure" in DATASET_STORAGE:
            container = DATASET_STORAGE["azure"].get("container", "")
            ctx.run(f"az storage blob download-batch -s {container}/data/ -d data/")
        elif "gcs" in DATASET_STORAGE:
            bucket = DATASET_STORAGE["gcs"].get("bucket", "")
            ctx.run(f"gsutil -m rsync -r gs://{bucket}/data/ data/")

    # Your code to sync data up
    print("Syncing data up...")

@task
def sync_data_up(ctx):
    """Upload Data to storage system"""
    if DATASET_STORAGE:
        if "none" in DATASET_STORAGE:
            print("No DATASET_STORAGE found. Please ensure the dataset is available.")

        elif "s3" in DATASET_STORAGE:
            bucket = DATASET_STORAGE["s3"].get("bucket", "")
            profile = DATASET_STORAGE["s3"].get("aws_profile", "default")
            profile_option = f" --profile {profile}" if profile != "default" else ""
            ctx.run(f"aws s3 sync data/ s3://{bucket}/data{profile_option}")
        elif "azure" in DATASET_STORAGE:
            container = DATASET_STORAGE["azure"].get("container", "")
            ctx.run(f"az storage blob upload-batch -d {container}/data/ -s data/")
        elif "gcs" in DATASET_STORAGE:
            bucket = DATASET_STORAGE["gcs"].get("bucket", "")
            ctx.run(f"gsutil -m rsync -r data/ gs://{bucket}/data/")

@task
def create_environment(ctx):
    """Set up python interpreter environment"""
    if ENVIRONMENT_MANAGER == "none":
        print("Environment manager was not configured during project creation.",
              "Set up an environment manually or re-do the project creation process.", sep ='\n')

    elif ENVIRONMENT_MANAGER in ("conda", "mamba"):
        env_manager = "mamba" if ENVIRONMENT_MANAGER == "mamba" else "conda"
        if DEPENDENCY_FILE != "environment.yml":
            ctx.run(f"{env_manager} create --name {PROJECT_NAME} python={PYTHON_VERSION} -y -v")
        else:
            ctx.run(f"{env_manager} env create --name {PROJECT_NAME} -f environment.yml")
        print(f">>> {env_manager} env created. Activate with:\n{env_manager} activate {PROJECT_NAME}")

    elif ENVIRONMENT_MANAGER == "virtualenv":
        ctx.run(f"virtualenv {PROJECT_NAME} --python={PYTHON_INTERPRETER}")
        print(f">>> New virtualenv created. Activate with:\nsource {PROJECT_NAME}/bin/activate")
    elif ENVIRONMENT_MANAGER == "pipenv":
        ctx.run(f"pipenv --python {PYTHON_VERSION}")
        print(f">>> New pipenv created. Activate with:\npipenv shell")

@task
def data(ctx):
    """Make Dataset"""
    requirements(ctx)
    ctx.run(f"{PYTHON_INTERPRETER} {MODULE_NAME}/dataset.py")

@task
def help(ctx):
    """Display help message"""
    print("Available tasks:")
    for task in ctx.collection:
        print(f"{task.name}: {task.help}")

# Aliases for some common tasks
ns = Collection()
ns.add_task(requirements)
ns.add_task(clean)
ns.add_task(lint)
ns.add_task(format)
ns.add_task(sync_data_down)
ns.add_task(sync_data_up)
ns.add_task(create_environment)
ns.add_task(data)
ns.add_task(help)
