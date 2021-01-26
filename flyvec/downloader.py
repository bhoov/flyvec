# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_Download Helpers.ipynb (unless otherwise specified).

__all__ = ['get_config_dir', 'get_model_dir', 'download_flyvec_data', 'unzip_data', 'prepare_flyvec_data']

# Cell
import boto3
from pathlib import Path
import zipfile
from progressbar.progressbar import ProgressBar
import tempfile
import yaml

# Cell
def get_config_dir():
    config_dir = Path.home() / ".cache" / "flyvec"
    return config_dir

def get_model_dir():
    return get_config_dir() / "data"

# Cell
def download_flyvec_data(outfile=None, force=False):
    """Download the zipped flyvec model from the cloud to a local file. If `outfile` is not provided,
    use (the OS's) TEMPDIR / 'flyvec-data.zip'

    """
    tmp_file = Path(outfile) if outfile is not None else Path(tempfile.gettempdir()) / "flyvec-data.zip"
    if tmp_file.exists() and not force:
        print(f"Found existing {tmp_file}, reusing")
        return tmp_file

    access_key = "07598db5c9364ad29002fe8e22daddd3"
    secret_key = "a7bec64c8840439576380beb238b161117f2aeb3e7f993f0"
    service_endpoint = 'https://s3.ap.cloud-object-storage.appdomain.cloud'
    session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name="ap-geo")

    s3 = session.resource("s3", endpoint_url=service_endpoint)
    bucket = s3.Bucket("hoo-flyvec")
    obj = bucket.Object("data.zip")
    down_progress = ProgressBar(obj.content_length)

    print("Downloading flyvec data:")
    down_progress.start()

    def download_progress(chunk):
        down_progress.update(down_progress.currval + chunk)

    with open(str(tmp_file), 'wb') as fd:
        obj.download_fileobj(fd, Callback=download_progress)

    down_progress.finish()

    return tmp_file

# Cell
def unzip_data(path_to_zipped_data, outdir=None):
    """Unzip the flyvec models to the config directory. If `outdir` is not provided, use default flyvec configuration dir"""

    config_dir = get_config_dir() if outdir is None else Path(outdir)
    if not config_dir.exists(): config_dir.mkdir(parents=True)

    with zipfile.ZipFile(str(path_to_zipped_data), mode='r') as zd:
        zd.extractall(config_dir)

    return config_dir

# Cell
def prepare_flyvec_data(force=False):
    """Create pipeline to download flyvec data with default settings.

    Args:
        force: If true, don't check for existance of files
    """

    if not force:
        # Check if file exists
        model_dir = get_model_dir()
        conf_f = model_dir / "config.yaml"

        if conf_f.exists():
            with open(conf_f, "r") as fp:
                conf = yaml.load(fp, Loader=yaml.FullLoader)

            synapses_f = model_dir / conf['synapses']
            tokenizer_f = model_dir / conf['tokenizer']
            if synapses_f.exists() and tokenizer_f.exists():
                return

    tmp_file = download_flyvec_data(force=force)
    unzip_data(tmp_file)