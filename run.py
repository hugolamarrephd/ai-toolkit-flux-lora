# Backblaze B2 environment variables: B2_ID, B2_TOKEN
import time
import os
from pathlib import Path


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import sys
from typing import Union, OrderedDict
from dotenv import load_dotenv
from b2sdk.v2 import (
    InMemoryAccountInfo,
    B2Api,
    Synchronizer,
    parse_folder,
    ScanPoliciesManager,
    CompareVersionMode,
    NewerFileSyncMode,
    KeepOrDeleteMode,
    SyncReport,
)

# Load the .env file if it exists
load_dotenv()
WORKSPACE_PATH = os.environ.get("WORKSPACE", "/home/workspace/")
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "hlam-ai-models")
DATASET_BUCKET = os.environ.get("DATASET_BUCKET", "hlam-ai-datasets")
IMAGES_PATH = WORKSPACE_PATH + "images/"
OUTPUTS_PATH = WORKSPACE_PATH + "outputs/"


# Verify that the project folder exists in datasets bucket
def verify_project_folder_exists(conn, bucket, project):
    """Verify that the PROJECT folder exists in the datasets bucket"""
    try:
        bucket = conn.get_bucket_by_name(bucket)
        # List files in the project folder to verify it exists
        folders = set(
            folder.replace("/", "")
            for _, folder in bucket.ls(recursive=False, latest_only=True)
        )
        if project not in folders:
            print(
                f"ERROR: Project folder '{project}' not found in {bucket} bucket; "
                f"available projects are: {folders}"
            )
            print(
                f"...Please create a folder containing the images to train FLUX on "
                f"at b2://{bucket}/{project}/"
            )
            sys.exit(1)

        print(f"✓ Project folder '{project}' found in {bucket} bucket")
    except Exception as e:
        print(f"ERROR: Failed to verify project folder in bucket: {e}")
        sys.exit(1)


def sync(conn, source, destination):
    """Sync source to destination using Backblaze B2"""
    source = parse_folder(source, conn)
    destination = parse_folder(destination, conn)
    policies_manager = ScanPoliciesManager(exclude_all_symlinks=True)
    synchronizer = Synchronizer(
        max_workers=10,
        policies_manager=policies_manager,
        dry_run=False,
        allow_empty_source=True,
        compare_version_mode=CompareVersionMode.SIZE,
        compare_threshold=10,
        newer_file_mode=NewerFileSyncMode.REPLACE,
        keep_days_or_delete=KeepOrDeleteMode.DELETE,
    )
    with SyncReport(sys.stdout, False) as reporter:
        synchronizer.sync_folders(
            source_folder=source,
            dest_folder=destination,
            now_millis=int(round(time.time() * 1000)),
            reporter=reporter,
        )


sys.path.insert(0, os.getcwd())
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ["DISABLE_TELEMETRY"] = "YES"

# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # set torch to trace mode
    import torch

    torch.autograd.set_detect_anomaly(True)
import argparse
from toolkit.job import get_job
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file

accelerator = get_accelerator()


def main():
    setup_log_to_file(OUTPUTS_PATH + "log.txt")
    if (
        not Path(WORKSPACE_PATH).exists()
        or not Path(IMAGES_PATH).exists()
        or not Path(OUTPUTS_PATH).exists()
    ):
        print(
            f"Missing directories: {WORKSPACE_PATH}, {IMAGES_PATH} and/or {OUTPUTS_PATH}"
        )
        Path(WORKSPACE_PATH).mkdir(parents=True, exist_ok=True)
        Path(IMAGES_PATH).mkdir(parents=True, exist_ok=True)
        Path(OUTPUTS_PATH).mkdir(parents=True, exist_ok=True)
        print("✓ Directories created successfully")
    # Backblaze setup
    B2 = B2Api(InMemoryAccountInfo())  # type: ignore
    B2.authorize_account(
        "production",
        os.environ["B2_ID"],
        os.environ["B2_TOKEN"],
    )
    # Extract project name from command line arguments
    # ```
    # $ python3 run.py PROJECT_NAME
    # ```
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "project",
        type=str,
        help="Name of project",
    )
    args = parser.parse_args()

    PROJECT = args.project
    verify_project_folder_exists(B2, DATASET_BUCKET, PROJECT)
    verify_project_folder_exists(B2, MODEL_BUCKET, PROJECT)
    sync(
        B2,
        f"b2://{DATASET_BUCKET}/" + PROJECT,
        IMAGES_PATH,
    )  # download images and configuration
    sync(
        B2,
        f"b2://{MODEL_BUCKET}/" + PROJECT,
        OUTPUTS_PATH,
    )  # persist previous runs, logs, etc. if any
    config_file = IMAGES_PATH + ".config.yaml"
    try:
        job = get_job(config_file, args.project)
        job.run()
        job.cleanup()
    except Exception as e:
        print_acc(f"Error running job: {e}")
        try:
            job.process[0].on_error(e)
        except Exception as e2:
            print_acc(f"Error running on_error: {e2}")
    except KeyboardInterrupt as e:
        try:
            job.process[0].on_error(e)
        except Exception as e2:
            print_acc(f"Error running on_error: {e2}")
    sync(
        B2,
        OUTPUTS_PATH,
        f"b2://{MODEL_BUCKET}/" + PROJECT,
    )


def try_to_stop_runpod_pod():
    try:
        import subprocess
        pod_id = os.getenv("RUNPOD_POD_ID")
        print("RunPod pod ID:", pod_id)
        if pod_id:
            # Use runpodctl to stop the current pod
            subprocess.run(["runpodctl", "stop", "pod", pod_id])
            print("RunPod pod stopped successfully")
    except Exception as e:
        print("RunPod stop pod error:", e)
        print("Could not stop pod")


if __name__ == "__main__":
    main()
    try_to_stop_runpod_pod()
