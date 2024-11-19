"""Get RHEL image data for AWS."""

import json
import re
from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=None)
def get_rhel_images() -> pd.DataFrame:
    """Get RHEL image data for AWS.

    The Red Hat Cloud Experience team maintains the JSON file at:
    https://cloudx-json-bucket.s3.amazonaws.com/raw/aws/aws.json

    Schema:
        #   Column               Non-Null Count  Dtype
        ---  ------               --------------  -----
        0   PlatformDetails      6594 non-null   object
        1   UsageOperation       6594 non-null   object
        2   BlockDeviceMappings  6594 non-null   object
        3   Description          6594 non-null   object
        4   EnaSupport           6594 non-null   bool
        5   Hypervisor           6594 non-null   object
        6   ImageOwnerAlias      6594 non-null   object
        7   Name                 6594 non-null   object
        8   RootDeviceName       6594 non-null   object
        9   RootDeviceType       6594 non-null   object
        10  SriovNetSupport      6594 non-null   object
        11  VirtualizationType   6594 non-null   object
        12  DeprecationTime      6594 non-null   object
        13  ImageId              6594 non-null   object
        14  ImageLocation        6594 non-null   object
        15  State                6594 non-null   object
        16  OwnerId              6594 non-null   object
        17  CreationDate         6594 non-null   object
        18  Public               6594 non-null   bool
        19  Architecture         6594 non-null   object
        20  ImageType            6594 non-null   object
        21  Region               6594 non-null   object
        22  BootMode             1625 non-null   object

    Returns:
        list: RHEL image data for AWS
    """
    with open("get_data/raw_data/clouds/aws.json") as fileh:
        data = json.load(fileh)

    df = pd.json_normalize(data)
    return df


@lru_cache(maxsize=None)
def get_unique_image_names() -> list:
    """Get unique image names.

    Returns:
        list: Unique image names
    """
    df = get_rhel_images()

    # Remove old GP2 storage type images.
    df = df[~df["Name"].str.endswith("GP2")]

    # Remove any beta versions.
    df = df[~df["Name"].str.contains("BETA")]

    # Exclude images containing GA (not sure why they are there).
    df = df[~df["Name"].str.contains("GA")]

    # Exclude ARM images for now.
    df = df[~df["Name"].str.contains("arm64")]

    return list(df["Name"].sort_values().unique())


def get_latest_image_names() -> list:
    """Get latest image names.

    Returns:
        list: Latest image names
    """
    image_names = get_unique_image_names()
    images = []
    for image_name in image_names:
        image_data = parse_image_name(image_name)

        # Remove the date as well as the trailing items after the arch.
        image_name_no_date = "-".join(
            re.sub(r"-[\d]{8}", "", image_name).split("-")[:3]
        )

        images.append([image_name, image_name_no_date, int(image_data["date"])])

    df = pd.DataFrame(images, columns=["Name", "NameNoDate", "Date"])
    df = df.sort_values(by=["NameNoDate", "Date"])

    latest_versions = df.loc[df.groupby("NameNoDate")["Date"].idxmax()]

    return list(latest_versions["Name"])


def get_image_data_for_image(name: str) -> dict:
    """Get image data for a specific image.

    Args:
        name (str): Name of the image

    Returns:
        dict: Image data
    """
    images = get_rhel_images()

    if name not in images["Name"].values:
        return {}

    return dict(images[images["Name"] == name].to_dict(orient="records")[0])


def get_regions_for_image(name: str) -> list:
    """Get regions for a specific image.

    Args:
        name (str): Name of the image

    Returns:
        list: Regions for the image
    """
    images = get_rhel_images()

    if name not in images["Name"].values:
        return []

    df = images[images["Name"] == name]
    images_in_regions = df[["Region", "ImageId"]].to_dict(orient="records")

    return list(images_in_regions)


def parse_image_name(image_name: str) -> dict[str, str]:
    """Parse an AWS image name and return extra data about the image.

    Original script: https://github.com/redhatcloudx/transformer/blob/main/src/cloudimagedirectory/format/format_aws.py
    Regex101: https://regex101.com/r/mXCl73/1

    Args:
        image_name: String containing the image name, such as:
                    RHEL-9.0.0_HVM_BETA-20211026-x86_64-10-Hourly2-GP2

    Returns:
        Dictionary with additional information about the image.
    """
    # See Regex101 link above to tinker with this regex. Each group is named to make
    # it easier to handle parsed data. Explanation of names:
    #
    #     intprod = internal product (such as HA)
    #     extprod = external product (such as SAP)
    #     version = RHEL version (such as 9.0.0)
    #     virt = virtualization type (such as HVM)
    #     beta = beta vs non-beta release
    #     date = date image was produced
    #     arch = architecture (such as x86_64 or arm64)
    #     release = release number of the image
    #     billing = Hourly2 or Access2
    #     storage = storage type (almost always GP2)
    #
    aws_image_name_regex = (
        r"RHEL_*(?P<intprod>\w*)?-*(?P<extprod>\w*)?-(?P<version>[\d\-\.]*)_"
        r"(?P<virt>[A-Z]*)_*(?P<beta>\w*)?-(?P<date>\d+)-(?P<arch>\w*)-"
        r"(?P<release>\d+)-(?P<billing>[\w\d]*)-(?P<storage>\w*)"
    )
    matches = re.match(aws_image_name_regex, image_name, re.IGNORECASE)
    if matches:
        return matches.groupdict()

    return {}


def render_image_chunks(image_name: str) -> list:
    """Render chunks of RHEL image data.

    Args:
        image_name: Name of the image

    Returns:
        list: Chunks of RHEL image data
    """
    # Get the image information from the main dataframe.
    i = get_image_data_for_image(image_name)

    # Parse the name of the image to get some additional data.
    d = parse_image_name(image_name)

    # Get all the AMI IDs for this image in each region.
    regions = get_regions_for_image(image_name)

    intprod_names = {
        "": "",
        "SAP": "with SAP ",
        "HA": "with High Availability (HA) ",
    }

    chunks = []
    base_chunk = (
        "RHEL "
        f"{d['version']}{' Beta' if d['beta'] == 'BETA' else ''} "
        f"{intprod_names[d['intprod']]}"
        f"{i['Architecture']} "
    )

    for region in regions:
        chunks.append(
            base_chunk + f"AWS {region['Region']} AMI {region['ImageId']}"
        )

    return chunks


if __name__ == "__main__":
    image_names = get_latest_image_names()

    from pprint import pprint

    pprint(image_names)

    chunks = []
    for image_name in image_names:
        chunks.extend(render_image_chunks(image_name))

    with open("plaintext/clouds/aws.txt", "w") as fileh:
        fileh.write(f"\n{'━'*120}\n".join(chunks))
