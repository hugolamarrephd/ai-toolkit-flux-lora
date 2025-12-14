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
IMAGES_PATH = WORKSPACE_PATH + "images/"
OUTPUTS_PATH = WORKSPACE_PATH + "outputs/"
if (
    not Path(WORKSPACE_PATH).exists()
    or not Path(IMAGES_PATH).exists()
    or not Path(OUTPUTS_PATH).exists()
):
    print(
        f"ERROR: Missing directories: {WORKSPACE_PATH}, {IMAGES_PATH} and/or {OUTPUTS_PATH}"
    )
    response = (
        input("Would you like to create these directories? (yes/no): ").strip().lower()
    )
    if response in ["yes", "y"]:
        Path(WORKSPACE_PATH).mkdir(parents=True, exist_ok=True)
        Path(IMAGES_PATH).mkdir(parents=True, exist_ok=True)
        Path(OUTPUTS_PATH).mkdir(parents=True, exist_ok=True)
        print("✓ Directories created successfully")
    else:
        print("Aborting: directories not created")
        sys.exit(1)

PROJECT = input("Enter PROJECT_NAME: ")
# Backblaze setup
B2 = B2Api(InMemoryAccountInfo())  # type: ignore
B2.authorize_account(
    "production",
    os.environ["B2_ID"],
    os.environ["B2_TOKEN"],
)


# Verify that the project folder exists in hlam-ai-datasets bucket
def verify_datasets_project_folder_exists():
    """Verify that the PROJECT folder exists in the hlam-ai-datasets bucket"""
    try:
        bucket = B2.get_bucket_by_name("hlam-ai-datasets")
        # List files in the project folder to verify it exists
        folders = set(
            folder.replace("/", "")
            for _, folder in bucket.ls(recursive=False, latest_only=True)
        )
        if PROJECT not in folders:
            print(
                f"ERROR: Project folder '{PROJECT}' not found in hlam-ai-datasets bucket; "
                f"available projects are: {folders}"
            )
            print(
                f"...Please create a folder containing the images to train FLUX on "
                f"at b2://hlam-ai-datasets/{PROJECT}/"
            )
            sys.exit(1)

        print(f"✓ Project folder '{PROJECT}' found in hlam-ai-datasets bucket")
    except Exception as e:
        print(f"ERROR: Failed to verify project folder in bucket: {e}")
        sys.exit(1)


def verify_models_is_accessible():
    """Verify that hlam-ai-models bucket exists and we have write access"""
    try:
        bucket = B2.get_bucket_by_name("hlam-ai-models")

        # Test write access by checking if we can get bucket info
        bucket_info = bucket.get_id()
        print(bucket_info)

        if not bucket_info:
            print("ERROR: Unable to access hlam-ai-models bucket info")
            sys.exit(1)

        print(f"✓ hlam-ai-models bucket exists and is accessible")

    except Exception as e:
        print(f"ERROR: Failed to access hlam-ai-models bucket: {e}")
        print("Please ensure:")
        print("  - hlam-ai-models bucket exists in Backblaze B2")
        print("  - Your B2 credentials have write access to the bucket")
        sys.exit(1)


def sync(source, destination):
    """Sync source to destination using Backblaze B2"""
    source = parse_folder(source, B2)
    destination = parse_folder(destination, B2)
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


def print_end_message(jobs_completed, jobs_failed):
    if not accelerator.is_main_process:
        return
    failure_string = (
        f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}"
        if jobs_failed > 0
        else ""
    )
    completed_string = (
        f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"
    )

    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def main():
    parser = argparse.ArgumentParser()

    # require at lease one config file
    parser.add_argument(
        "config_file_list",
        nargs="+",
        type=str,
        help="Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially",
    )

    # flag to continue if failed job
    parser.add_argument(
        "-r",
        "--recover",
        action="store_true",
        help="Continue running additional jobs even if a job fails",
    )

    # flag to continue if failed job
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="Name to replace [name] tag in config file, useful for shared config file",
    )

    parser.add_argument(
        "-l", "--log", type=str, default=None, help="Log file to write output to"
    )
    args = parser.parse_args()

    if args.log is not None:
        setup_log_to_file(args.log)

    config_file_list = args.config_file_list
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    jobs_completed = 0
    jobs_failed = 0

    if accelerator.is_main_process:
        print_acc(
            f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}"
        )

    for config_file in config_file_list:
        try:
            job = get_job(config_file, args.name)
            job.run()
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            print_acc(f"Error running job: {e}")
            jobs_failed += 1
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e
        except KeyboardInterrupt as e:
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e


if __name__ == "__main__":
    verify_datasets_project_folder_exists()
    verify_models_is_accessible()
    sync("b2://hlam-ai-datasets/" + PROJECT, IMAGES_PATH)
    main()
    sync(OUTPUTS_PATH, "b2://hlam-ai-models/" + PROJECT)
