# Future Improvements

**TODO**: Incomplete List of Future Improvements...

## Experiment Handling in Docker Deployment
- [x] Check GCS Bucket for experiment_metadata.json and confirm if new experiment is to be created or resumed.
- [x] If new experiment, create a new entry in the experiment_metadata.json file.
- [ ] If resuming an experiment pull data from GCS Bucket and store it in the local directory.
- [ ] Explore solutions to enable **live TensorBoard logging** by streaming logs to GCS or another cloud-based logging system.
- [x] Continue from last tuning checkpoint.

## Callbacks Configurable through Config/ENV Variables
- Make Callbacks Configurable so e.g. TensorBoard Callback can be turned off/on through config file or environment variable.
- [ ] TODO

## Abstraction Layer for Cloud Platform
- Should be able to switch between cloud platforms without changing the code.
- Should support multiple cloud platforms (GCP, AWS, Azure, IBM Cloud, ...)
- Single Script for Batch Processing Request to Cloud Platform
- [ ] TODO

## Common Configuration / Improved Environment Variables/File 
- scripts/ use a lot of hardcode values in several places the same values are repeated, should be moved to a common configuration file.

## Parameterizable Generic Data Pipeline
- [ ] TODO

## Splitter
- add ExpandingWindowSplitter and RollingWindowSplitter

## Data Preparation / Manipulation
- Removing Every nth Point
- Averaging/Aggregation Every nth Point
- Down sampling with Interpolation

## UI for Experiment Management and Configuration setup




