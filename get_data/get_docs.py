"""Download RHEL docs and extract plaintext chunks."""

import logging
import os
import re
import sys
from glob import glob

import requests
from unstructured.partition.html import partition_html

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s;%(levelname)s;%(message)s",
)
logger = logging.getLogger(__name__)

HTML_OUTPUT_DIR = "get_data/raw_data/docs"
PLAINTEXT_OUTPUT_DIR = "./plaintext/9"


def download_html_docs() -> None:
    """Download HTML docs from the Red Hat website."""
    doc_url_stubs = [
        "accessing_identity_management_services",
        "administering_the_system_using_the_gnome_desktop_environment",
        "automatically_installing_rhel",
        "automating_system_administration_by_using_rhel_system_roles",
        "boot_options_for_rhel_installer",
        "building_running_and_managing_containers",
        "composing_a_customized_rhel_system_image",
        "composing_installing_and_managing_rhel_for_edge_images",
        "configuring_and_managing_cloud-init_for_rhel_9",
        "configuring_and_managing_high_availability_clusters",
        "configuring_and_managing_logical_volumes",
        "configuring_and_managing_networking",
        "configuring_and_managing_virtualization",
        "configuring_and_using_a_cups_printing_server",
        "configuring_and_using_database_servers",
        "configuring_and_using_network_file_services",
        "configuring_a_red_hat_high_availability_cluster_on_red_hat_openstack_platform",
        "configuring_authentication_and_authorization_in_rhel",
        "configuring_basic_system_settings",
        "configuring_device_mapper_multipath",
        "configuring_firewalls_and_packet_filters",
        "configuring_gfs2_file_systems",
        "configuring_infiniband_and_rdma_networks",
        "considerations_in_adopting_rhel_9",
        "customizing_anaconda",
        "customizing_the_gnome_desktop_environment",
        "deduplicating_and_compressing_logical_volumes_on_rhel",
        "deploying_mail_servers",
        "deploying_rhel_9_on_amazon_web_services",
        "deploying_rhel_9_on_google_cloud_platform",
        "deploying_rhel_9_on_microsoft_azure",
        "deploying_web_servers_and_reverse_proxies",
        "developing_c_and_cpp_applications_in_rhel_9",
        "getting_started_with_the_gnome_desktop_environment",
        "getting_the_most_from_your_support_experience",
        "installing_and_using_dynamic_programming_languages",
        "installing_identity_management",
        "installing_trust_between_idm_and_ad",
        "integrating_rhel_systems_directly_with_windows_active_directory",
        "interactively_installing_rhel_from_installation_media",
        "interactively_installing_rhel_over_the_network",
        "managing_and_monitoring_security_updates",
        "managing_certificates_in_idm",
        "managing_file_systems",
        "managing_idm_users_groups_hosts_and_access_control_rules",
        "managing_monitoring_and_updating_the_kernel",
        "managing_networking_infrastructure_services",
        "managing_replication_in_identity_management",
        "managing_smart_card_authentication",
        "managing_software_with_the_dnf_tool",
        "managing_storage_devices",
        "managing_systems_using_the_rhel_9_web_console",
        "migrating_to_identity_management_on_rhel_9",
        "monitoring_and_managing_system_status_and_performance",
        "package_manifest",
        "packaging_and_distributing_software",
        "performing_disaster_recovery_with_identity_management",
        "planning_identity_management",
        "preparing_for_disaster_recovery_with_identity_management",
        "securing_networks",
        "security_hardening",
        "tuning_performance_in_identity_management",
        "upgrading_from_rhel_8_to_rhel_9",
        "using_ansible_to_install_and_manage_identity_management",
        "using_external_red_hat_utilities_with_identity_management",
        "using_idm_api",
        "using_idm_healthcheck_to_monitor_your_idm_environment",
        "using_image_mode_for_rhel_to_build_deploy_and_manage_operating_systems",
        "using_selinux",
        "using_systemd_unit_files_to_customize_and_optimize_your_system",
        "working_with_dns_in_identity_management",
        "working_with_vaults_in_identity_management",
    ]

    for doc_url_stub in doc_url_stubs:
        output_file = f"{HTML_OUTPUT_DIR}/{doc_url_stub}.html"

        if os.path.exists(output_file):
            logger.info(f"Skipping download for {doc_url_stub}; already exists")
            continue

        logger.info(f"Downloading {doc_url_stub}")
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
            "Accept-Language": "en-US,en;q=0.7,es-ES;q=0.3",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Referer": "https://docs.redhat.com/",
            "Origin": "https://docs.redhat.com",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
        }
        page_url = f"https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html-single/{doc_url_stub}/index"
        resp = requests.get(page_url, headers=headers, timeout=10)
        with open(output_file, "w") as fileh:
            fileh.write(resp.text)

    return None


def get_text_from_file(filename: str) -> str:
    """Get the partitioned text from a file."""
    elements = partition_html(
        filename=filename,
        skip_headers_and_footers=True,
    )
    text = "\n".join([x.text for x in elements])

    return text


def remove_legal_notice(text: str) -> str:
    """Remove the legal notice from the text."""
    if re.match(r"^Legal Notice", text, flags=re.MULTILINE):
        return re.split(r"^Legal Notice", text, flags=re.MULTILINE)[1]

    return text


def fix_first_chunk(text: str) -> str:
    """Remove the feedback request from the first chunk."""
    feedback_string = r"Providing feedback on Red Hat documentation"
    return re.split(feedback_string, text, flags=re.MULTILINE)[0].strip()


def extract_chunks(text: str) -> list[str]:
    """Extract chunks from the text."""
    header_pattern = r"(^(?:Chapter )*([\d]+[\.]+)+)|(^Abstract)"
    matches = list(re.finditer(header_pattern, text, flags=re.MULTILINE))

    # Extract chunks
    chunks = []
    for i, match in enumerate(matches):
        # Get the position of the first match.
        start = match.start()

        # Get the position of the next match (which is the end of the current chunk).
        # Just use the end of the text if there is no next match.
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        # Extract the chunk and remove extra spaces.
        chunk = text[start:end].strip()

        # Remove the Legal Notice boilerplate.
        from nltk.tokenize import LineTokenizer

        if "Legal Notice" in chunk:
            lines = LineTokenizer(blanklines="keep").tokenize(chunk)
            notice_index = lines.index("Legal Notice")
            chunk = "\n".join(lines[0:notice_index])

        # Add it to our list of chunks.
        chunks.append(chunk)

    # Fix the first chunk
    chunks[0] = fix_first_chunk(chunks[0])

    return chunks


if __name__ == "__main__":
    download_html_docs()
    html_files = glob(f"{HTML_OUTPUT_DIR}/*.html")

    for html_file in html_files:
        print(html_file)
        text = get_text_from_file(html_file)
        chunks = extract_chunks(text)

        file_stub = os.path.basename(html_file).rsplit(".", 1)[0]
        print(
            f"Writing {len(chunks)} chunks to {PLAINTEXT_OUTPUT_DIR}/{file_stub}.txt"
        )
        with open(f"{PLAINTEXT_OUTPUT_DIR}/{file_stub}.txt", "w") as fileh:
            fileh.write(f"\n{'━'*120}\n".join(chunks))
