# ENVIRONMENT VARIABLES
- `WORKSPACE` - path to the local workspace. Should have been initialized with `./images/` and `./outputs/`.
- `MODEL_BUCKET` - Name of the Backblaze B2 Bucket used to persist models.
  Models will be saved at `b2://{MODEL_BUCKET}/{PROJECT_NAME}`
- `DATASET_BUCKET` - Name of the Backblaze B2 Bucket used to pull datasets.
  Images and captions are expected to be located at `b2://{DATASET_BUCKET}/{PROJECT_NAME}`.
- `B2_ID` - Identifier of the B2 TOKEN (eg. ACCOUNT_ID)
- `B2_TOKEN` - B2 Key/Token 
