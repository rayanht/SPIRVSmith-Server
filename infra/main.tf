terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "4.16.0"
    }
  }
}

provider "google" {
  credentials = var.gcp_credentials

  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "google-beta" {
  credentials = var.gcp_credentials

  project = var.project_id
  region  = var.region
  zone    = var.zone
}


resource "google_service_account" "default" {
  account_id   = "spirv-default"
  display_name = "SPIRVSmith Service Account"
}

resource "google_artifact_registry_repository" "spirvsmith_repo" {
  provider = google-beta

  location      = var.region
  repository_id = "spirvsmith-docker-images"
  description   = "Docker repo for SPIRVSmith images"
  format        = "DOCKER"
}

resource "google_bigquery_dataset" "spirv_dataset" {
  dataset_id    = "spirv"
  friendly_name = "Shaders metadata"
  location      = "US"
}

resource "google_bigquery_table" "spirv_shader_data_table" {
  dataset_id          = google_bigquery_dataset.spirv_dataset.dataset_id
  table_id            = "shader_data"
  deletion_protection = false

  schema = <<EOF
[
  {
    "name": "shader_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Shader identifier. Not a primary key."
  },
  {
    "name": "buffer_dump",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "A single reported buffer dump. Initially NULL."
  },
  {
    "name": "platform_os",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "One of Linux, Darwin, or Windows"
  },
  {
    "name": "platform_hardware_type",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "One of CPU/GPU"
  },
  {
    "name": "platform_hardware_vendor",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "One of AMD/NVIDIA/INTEL"
  },
  {
    "name": "platform_hardware_model",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "Fully-qualified name of the hardware running the shader"
  },
    {
    "name": "platform_hardware_driver_version",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "GPU driver version, if applicable"
  },
  {
    "name": "platform_backend",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "e.g. Vulkan or SwiftShader"
  },
  {
    "name": "insertion_time",
    "type": "TIMESTAMP",
    "mode": "REQUIRED",
    "description": "Time of insertion into the database."
  }
]
EOF

}

resource "google_storage_bucket" "spirv_shaders_bucket" {
  name          = "spirv_shaders_bucket"
  location      = "US-WEST1"
  force_destroy = true
}


resource "google_compute_address" "static" {
  name = "ipv4-address"
}

resource "google_compute_instance" "spirvsmith_server" {
  name                      = "spirvsmith-server"
  description               = "SPIRVSmith server"
  machine_type              = var.machine_type
  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image = "cos-cloud/cos-stable"
    }
  }
  service_account {
    email  = "spirvsmith-primary@spirvsmith.iam.gserviceaccount.com"
    scopes = ["cloud-platform"]
  }

  metadata = {
    gce-container-declaration = <<EOT
spec:
  containers:
    - image: python:3.10-slim
      name: containervm
      securityContext:
        privileged: false
      stdin: false
      tty: false
      volumeMounts: []
      restartPolicy: Always
      volumes: []
EOT
    google-logging-enabled    = "true"
  }
  network_interface {
    access_config {
      nat_ip = google_compute_address.static.address
    }
  }
}

