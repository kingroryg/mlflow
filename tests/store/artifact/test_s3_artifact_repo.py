import json
import os
import posixpath
import tarfile
from datetime import datetime, timezone
from unittest import mock
from unittest.mock import ANY

import botocore.exceptions
import pytest
import requests

from mlflow.entities.multipart_upload import MultipartUploadPart
from mlflow.exceptions import MlflowException, MlflowTraceDataCorrupted
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.optimized_s3_artifact_repo import (
    OptimizedS3ArtifactRepository,
)
from mlflow.store.artifact.s3_artifact_repo import (
    _MAX_CACHE_SECONDS,
    S3ArtifactRepository,
    _cached_get_s3_client,
)
from tests.helper_functions import set_boto_credentials  # noqa: F401


@pytest.fixture
def s3_artifact_root(mock_s3_bucket):
    return f"s3://{mock_s3_bucket}"


@pytest.fixture(params=[True, False])
def s3_artifact_repo(s3_artifact_root, request):
    if request.param:
        return OptimizedS3ArtifactRepository(
            posixpath.join(s3_artifact_root, "some/path")
        )
    return S3ArtifactRepository(posixpath.join(s3_artifact_root, "some/path"))


@pytest.fixture(autouse=True)
def reset_cached_get_s3_client():
    _cached_get_s3_client.cache_clear()


def teardown_function():
    if "MLFLOW_S3_UPLOAD_EXTRA_ARGS" in os.environ:
        del os.environ["MLFLOW_S3_UPLOAD_EXTRA_ARGS"]


def test_file_artifact_is_logged_and_downloaded_successfully(
    s3_artifact_repo, tmp_path
):
    file_name = "test.txt"
    file_path = os.path.join(tmp_path, file_name)
    file_text = "Hello world!"

    with open(file_path, "w") as f:
        f.write(file_text)

    s3_artifact_repo.log_artifact(file_path)
    with open(s3_artifact_repo.download_artifacts(file_name)) as f:
        assert f.read() == file_text


def test_file_artifact_is_logged_with_content_metadata(
    s3_artifact_repo, s3_artifact_root, tmp_path
):
    file_name = "test.txt"
    file_path = os.path.join(tmp_path, file_name)
    file_text = "Hello world!"

    with open(file_path, "w") as f:
        f.write(file_text)

    s3_artifact_repo.log_artifact(file_path)

    bucket, _ = s3_artifact_repo.parse_s3_compliant_uri(s3_artifact_root)
    s3_client = s3_artifact_repo._get_s3_client()
    response = s3_client.head_object(Bucket=bucket, Key="some/path/test.txt")
    assert response.get("ContentType") == "text/plain"
    assert response.get("ContentEncoding") == "aws-chunked"


def test_get_s3_client_hits_cache(s3_artifact_root, monkeypatch):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))
    repo._get_s3_client()
    cache_info = _cached_get_s3_client.cache_info()
    assert cache_info.hits == 0
    assert cache_info.misses == 1
    assert cache_info.currsize == 1

    repo._get_s3_client()
    cache_info = _cached_get_s3_client.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 1
    assert cache_info.currsize == 1

    monkeypatch.setenv("MLFLOW_EXPERIMENTAL_S3_SIGNATURE_VERSION", "s3v2")
    repo._get_s3_client()
    cache_info = _cached_get_s3_client.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 2
    assert cache_info.currsize == 2

    with mock.patch(
        "mlflow.store.artifact.s3_artifact_repo._get_utcnow_timestamp",
        return_value=datetime.now(timezone.utc).timestamp() + _MAX_CACHE_SECONDS,
    ):
        repo._get_s3_client()
    cache_info = _cached_get_s3_client.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 3
    assert cache_info.currsize == 3


@pytest.mark.parametrize(
    ("ignore_tls_env", "verify"),
    [("0", None), ("1", False), ("true", False), ("false", None)],
)
def test_get_s3_client_verify_param_set_correctly(
    s3_artifact_root, ignore_tls_env, verify, monkeypatch
):
    monkeypatch.setenv("MLFLOW_S3_IGNORE_TLS", ignore_tls_env)
    with mock.patch("boto3.client") as mock_get_s3_client:
        repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))
        repo._get_s3_client()
        mock_get_s3_client.assert_called_with(
            "s3",
            config=ANY,
            endpoint_url=ANY,
            verify=verify,
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            region_name=ANY,
        )


def test_s3_client_config_set_correctly(s3_artifact_root):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))
    s3_client = repo._get_s3_client()
    assert s3_client.meta.config.s3.get("addressing_style") == "auto"


def test_s3_creds_passed_to_client(s3_artifact_root):
    with mock.patch("boto3.client") as mock_get_s3_client:
        repo = S3ArtifactRepository(
            s3_artifact_root,
            access_key_id="my-id",
            secret_access_key="my-key",
            session_token="my-session-token",
        )
        repo._get_s3_client()
        mock_get_s3_client.assert_called_with(
            "s3",
            config=ANY,
            endpoint_url=ANY,
            verify=None,
            aws_access_key_id="my-id",
            aws_secret_access_key="my-key",
            aws_session_token="my-session-token",
            region_name=ANY,
        )


def test_file_artifacts_are_logged_with_content_metadata_in_batch(
    s3_artifact_repo, s3_artifact_root, tmp_path
):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    subdir_path = str(subdir)
    nested_path = os.path.join(subdir_path, "nested")
    os.makedirs(nested_path)
    path_a = os.path.join(subdir_path, "a.txt")
    path_b = os.path.join(subdir_path, "b.tar.gz")
    path_c = os.path.join(nested_path, "c.csv")

    with open(path_a, "w") as f:
        f.write("A")
    with tarfile.open(path_b, "w:gz") as f:
        f.add(path_a)
    with open(path_c, "w") as f:
        f.write("col1,col2\n1,3\n2,4\n")

    s3_artifact_repo.log_artifacts(subdir_path)

    bucket, _ = s3_artifact_repo.parse_s3_compliant_uri(s3_artifact_root)
    s3_client = s3_artifact_repo._get_s3_client()

    response_a = s3_client.head_object(Bucket=bucket, Key="some/path/a.txt")
    assert response_a.get("ContentType") == "text/plain"
    assert response_a.get("ContentEncoding") == "aws-chunked"

    response_b = s3_client.head_object(Bucket=bucket, Key="some/path/b.tar.gz")
    assert response_b.get("ContentType") == "application/x-tar"
    assert response_b.get("ContentEncoding") == "gzip,aws-chunked"

    response_c = s3_client.head_object(Bucket=bucket, Key="some/path/nested/c.csv")
    assert response_c.get("ContentType") == "text/csv"
    assert response_c.get("ContentEncoding") == "aws-chunked"


def test_file_and_directories_artifacts_are_logged_and_downloaded_successfully_in_batch(
    s3_artifact_repo, tmp_path
):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    subdir_path = str(subdir)
    nested_path = os.path.join(subdir_path, "nested")
    os.makedirs(nested_path)
    with open(os.path.join(subdir_path, "a.txt"), "w") as f:
        f.write("A")
    with open(os.path.join(subdir_path, "b.txt"), "w") as f:
        f.write("B")
    with open(os.path.join(nested_path, "c.txt"), "w") as f:
        f.write("C")

    s3_artifact_repo.log_artifacts(subdir_path)

    # Download individual files and verify correctness of their contents
    with open(s3_artifact_repo.download_artifacts("a.txt")) as f:
        assert f.read() == "A"
    with open(s3_artifact_repo.download_artifacts("b.txt")) as f:
        assert f.read() == "B"
    with open(s3_artifact_repo.download_artifacts("nested/c.txt")) as f:
        assert f.read() == "C"

    # Download the nested directory and verify correctness of its contents
    downloaded_dir = s3_artifact_repo.download_artifacts("nested")
    assert os.path.basename(downloaded_dir) == "nested"
    with open(os.path.join(downloaded_dir, "c.txt")) as f:
        assert f.read() == "C"

    # Download the root directory and verify correctness of its contents
    downloaded_dir = s3_artifact_repo.download_artifacts("")
    dir_contents = os.listdir(downloaded_dir)
    assert "nested" in dir_contents
    assert os.path.isdir(os.path.join(downloaded_dir, "nested"))
    assert "a.txt" in dir_contents
    assert "b.txt" in dir_contents


def test_file_and_directories_artifacts_are_logged_and_listed_successfully_in_batch(
    s3_artifact_repo, tmp_path
):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    subdir_path = str(subdir)
    nested_path = os.path.join(subdir_path, "nested")
    os.makedirs(nested_path)
    with open(os.path.join(subdir_path, "a.txt"), "w") as f:
        f.write("A")
    with open(os.path.join(subdir_path, "b.txt"), "w") as f:
        f.write("B")
    with open(os.path.join(nested_path, "c.txt"), "w") as f:
        f.write("C")

    s3_artifact_repo.log_artifacts(subdir_path)

    root_artifacts_listing = sorted(
        [(f.path, f.is_dir, f.file_size) for f in s3_artifact_repo.list_artifacts()]
    )
    assert root_artifacts_listing == [
        ("a.txt", False, 1),
        ("b.txt", False, 1),
        ("nested", True, None),
    ]

    nested_artifacts_listing = sorted(
        [
            (f.path, f.is_dir, f.file_size)
            for f in s3_artifact_repo.list_artifacts("nested")
        ]
    )
    assert nested_artifacts_listing == [("nested/c.txt", False, 1)]


def test_download_directory_artifact_succeeds_when_artifact_root_is_s3_bucket_root(
    s3_artifact_root, tmp_path
):
    file_a_name = "a.txt"
    file_a_text = "A"
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    subdir_path = str(subdir)
    nested_path = os.path.join(subdir_path, "nested")
    os.makedirs(nested_path)
    with open(os.path.join(nested_path, file_a_name), "w") as f:
        f.write(file_a_text)

    repo = get_artifact_repository(s3_artifact_root)
    repo.log_artifacts(subdir_path)

    downloaded_dir_path = repo.download_artifacts("nested")
    assert file_a_name in os.listdir(downloaded_dir_path)
    with open(os.path.join(downloaded_dir_path, file_a_name)) as f:
        assert f.read() == file_a_text


def test_download_file_artifact_succeeds_when_artifact_root_is_s3_bucket_root(
    s3_artifact_root, tmp_path
):
    file_a_name = "a.txt"
    file_a_text = "A"
    file_a_path = os.path.join(tmp_path, file_a_name)
    with open(file_a_path, "w") as f:
        f.write(file_a_text)

    repo = get_artifact_repository(s3_artifact_root)
    repo.log_artifact(file_a_path)

    downloaded_file_path = repo.download_artifacts(file_a_name)
    with open(downloaded_file_path) as f:
        assert f.read() == file_a_text


def test_get_s3_file_upload_extra_args():
    os.environ.setdefault(
        "MLFLOW_S3_UPLOAD_EXTRA_ARGS",
        '{"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "123456"}',
    )

    parsed_args = S3ArtifactRepository.get_s3_file_upload_extra_args()

    assert parsed_args == {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "123456"}


def test_get_s3_file_upload_extra_args_env_var_not_present():
    parsed_args = S3ArtifactRepository.get_s3_file_upload_extra_args()

    assert parsed_args is None


def test_get_s3_file_upload_extra_args_invalid_json():
    os.environ.setdefault(
        "MLFLOW_S3_UPLOAD_EXTRA_ARGS",
        '"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "123456"}',
    )

    with pytest.raises(json.decoder.JSONDecodeError, match=r".+"):
        S3ArtifactRepository.get_s3_file_upload_extra_args()


def test_delete_artifacts(s3_artifact_repo, tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    nested_path = subdir / "nested"
    nested_path.mkdir()
    path_a = subdir / "a.txt"

    path_a.write_text("A")
    with tarfile.open(str(subdir / "b.tar.gz"), "w:gz") as f:
        f.add(str(path_a))
    (nested_path / "c.csv").write_text("col1,col2\n1,3\n2,4\n")

    s3_artifact_repo.log_artifacts(str(subdir))

    # confirm that artifacts are present
    artifact_file_names = [obj.path for obj in s3_artifact_repo.list_artifacts()]
    assert "a.txt" in artifact_file_names
    assert "b.tar.gz" in artifact_file_names
    assert "nested" in artifact_file_names

    s3_artifact_repo.delete_artifacts()
    assert s3_artifact_repo.list_artifacts() == []


def test_delete_artifacts_single_object(s3_artifact_repo, tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    path_a = subdir / "a.txt"
    path_a.write_text("A")

    s3_artifact_repo.log_artifacts(str(subdir))

    # confirm that artifact is present
    artifact_file_names = [obj.path for obj in s3_artifact_repo.list_artifacts()]
    assert "a.txt" in artifact_file_names

    s3_artifact_repo.delete_artifacts(artifact_path="a.txt")
    assert s3_artifact_repo.list_artifacts() == []


@pytest.mark.parametrize("artifact_path", ["subdir", "subdir/"])
def test_list_and_delete_artifacts_path(s3_artifact_repo, tmp_path, artifact_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    path_a = subdir / "a.txt"
    path_a.write_text("A")

    s3_artifact_repo.log_artifacts(str(subdir), artifact_path.rstrip("/"))

    # confirm that artifact is present
    artifact_file_names = [
        obj.path for obj in s3_artifact_repo.list_artifacts(artifact_path)
    ]
    assert "subdir/a.txt" in artifact_file_names

    s3_artifact_repo.delete_artifacts(artifact_path=artifact_path)
    assert s3_artifact_repo.list_artifacts(artifact_path) == []
    assert s3_artifact_repo.list_artifacts() == []


@pytest.mark.parametrize(
    ("boto_error_code", "expected_mlflow_error"),
    [
        ("AccessDenied", "PERMISSION_DENIED"),
        ("NoSuchBucket", "RESOURCE_DOES_NOT_EXIST"),
        ("NoSuchKey", "RESOURCE_DOES_NOT_EXIST"),
        ("InvalidAccessKeyId", "UNAUTHENTICATED"),
        ("SignatureDoesNotMatch", "UNAUTHENTICATED"),
    ],
)
def test_list_artifacts_error_handling(
    s3_artifact_root, boto_error_code, expected_mlflow_error
):
    artifact_path = "some/path/"
    s3_repo = S3ArtifactRepository(posixpath.join(s3_artifact_root, artifact_path))

    with mock.patch.object(s3_repo, "_get_s3_client") as mock_client:
        mock_paginator = mock.Mock()
        boto_error_message = "Error message from the client"
        mock_paginator.paginate.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": boto_error_code, "Message": boto_error_message}},
            "ListObjectsV2",
        )
        mock_client.return_value.get_paginator.return_value = mock_paginator

        with pytest.raises(
            MlflowException,
            match=f"Failed to list artifacts in {s3_repo.artifact_uri}:",
        ) as exc_info:
            s3_repo.list_artifacts(artifact_path)
        assert exc_info.value.error_code == expected_mlflow_error
        assert boto_error_message in exc_info.value.message


def test_delete_artifacts_pagination(s3_artifact_repo, tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    # The maximum number of objects that can be listed in a single call is 1000
    # https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html
    for i in range(1100):
        (subdir / f"{i}.txt").write_text("A")

    s3_artifact_repo.log_artifacts(str(subdir))

    # confirm that artifacts are present
    artifact_file_names = [obj.path for obj in s3_artifact_repo.list_artifacts()]
    for i in range(1100):
        assert f"{i}.txt" in artifact_file_names

    s3_artifact_repo.delete_artifacts()
    assert s3_artifact_repo.list_artifacts() == []


def test_create_multipart_upload(s3_artifact_root):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))
    create = repo.create_multipart_upload("local_file")

    # confirm that a mpu is created with the correct upload_id
    bucket, _ = repo.parse_s3_compliant_uri(s3_artifact_root)
    s3_client = repo._get_s3_client()
    response = s3_client.list_multipart_uploads(Bucket=bucket)
    uploads = response.get("Uploads")
    assert len(uploads) == 1
    assert uploads[0]["UploadId"] == create.upload_id


def test_complete_multipart_upload(s3_artifact_root):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))
    local_file = "local_file"
    create = repo.create_multipart_upload(local_file, 2)

    # cannot complete invalid upload
    fake_parts = [
        MultipartUploadPart(part_number=1, etag="fake_etag1"),
        MultipartUploadPart(part_number=2, etag="fake_etag2"),
    ]
    with pytest.raises(botocore.exceptions.ClientError, match=r"InvalidPart"):
        repo.complete_multipart_upload(local_file, create.upload_id, fake_parts)

    # can complete valid upload
    parts = []
    data = b"0" * 5 * 1024 * 1024
    for credential in create.credentials:
        url = credential.url
        response = requests.put(url, data=data)
        parts.append(
            MultipartUploadPart(
                part_number=credential.part_number, etag=response.headers["ETag"]
            )
        )

    repo.complete_multipart_upload(local_file, create.upload_id, parts)

    # verify upload is completed
    bucket, _ = repo.parse_s3_compliant_uri(s3_artifact_root)
    s3_client = repo._get_s3_client()
    response = s3_client.list_multipart_uploads(Bucket=bucket)
    assert response.get("Uploads") is None


def test_abort_multipart_upload(s3_artifact_root):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))
    local_file = "local_file"
    create = repo.create_multipart_upload(local_file, 2)

    # cannot abort a non-existing upload
    with pytest.raises(botocore.exceptions.ClientError, match=r"NoSuchUpload"):
        repo.abort_multipart_upload(local_file, "fake_upload_id")

    # can abort the created upload
    repo.abort_multipart_upload(local_file, create.upload_id)

    # verify upload is aborted
    bucket, _ = repo.parse_s3_compliant_uri(s3_artifact_root)
    s3_client = repo._get_s3_client()
    response = s3_client.list_multipart_uploads(Bucket=bucket)
    assert response.get("Uploads") is None


def test_trace_data(s3_artifact_root):
    repo = get_artifact_repository(s3_artifact_root)
    # s3 download_file raises exception directly if the file doesn't exist
    with pytest.raises(Exception, match=r"Trace data not found"):
        repo.download_trace_data()
    repo.upload_trace_data("invalid data")
    with pytest.raises(
        MlflowTraceDataCorrupted, match=r"Trace data is corrupted for path="
    ):
        repo.download_trace_data()

    mock_trace_data = {"spans": [], "request": {"test": 1}, "response": {"test": 2}}
    repo.upload_trace_data(json.dumps(mock_trace_data))
    assert repo.download_trace_data() == mock_trace_data


# ============================================================================
# Bucket Ownership Security Tests
# ============================================================================
# These tests verify protection against bucket impersonation attacks where:
# 1. A user creates and uses a bucket (e.g., 'my-mlflow-artifacts')
# 2. The bucket is deleted
# 3. An attacker creates a new bucket with the same name
# 4. MLflow continues to use the same bucket URI, unknowingly sending
#    artifacts to the attacker's bucket
#
# The ExpectedBucketOwner parameter prevents this by verifying bucket
# ownership on every S3 API call.
# ============================================================================


def test_bucket_owner_parameter_included_in_upload(
    s3_artifact_root, tmp_path, monkeypatch
):
    """
    Test that ExpectedBucketOwner is included in upload operations when configured.

    This prevents an attacker from creating a bucket with the same name and
    intercepting artifact uploads.
    """
    monkeypatch.setenv("MLFLOW_S3_BUCKET_OWNER", "123456789012")

    repo = S3ArtifactRepository(s3_artifact_root)
    assert repo._expected_bucket_owner == "123456789012"

    # Create a test file
    file_path = tmp_path / "test.txt"
    file_path.write_text("test content")

    # Mock the S3 client to verify ExpectedBucketOwner is passed
    with mock.patch.object(repo, "_get_s3_client") as mock_get_client:
        mock_s3_client = mock.Mock()
        mock_get_client.return_value = mock_s3_client

        repo.log_artifact(str(file_path))

        # Verify upload_file was called with ExpectedBucketOwner in ExtraArgs
        mock_s3_client.upload_file.assert_called_once()
        call_kwargs = mock_s3_client.upload_file.call_args.kwargs
        assert "ExtraArgs" in call_kwargs
        assert "ExpectedBucketOwner" in call_kwargs["ExtraArgs"]
        assert call_kwargs["ExtraArgs"]["ExpectedBucketOwner"] == "123456789012"


def test_bucket_owner_parameter_included_in_download(
    s3_artifact_root, tmp_path, monkeypatch
):
    """
    Test that ExpectedBucketOwner is included in download operations when configured.

    This prevents downloading artifacts from an attacker's bucket with the same name.
    """
    monkeypatch.setenv("MLFLOW_S3_BUCKET_OWNER", "123456789012")

    repo = S3ArtifactRepository(s3_artifact_root)

    # Mock the S3 client to verify ExpectedBucketOwner is passed
    with mock.patch.object(repo, "_get_s3_client") as mock_get_client:
        mock_s3_client = mock.Mock()
        mock_get_client.return_value = mock_s3_client

        local_path = str(tmp_path / "downloaded.txt")
        repo._download_file("test.txt", local_path)

        # Verify download_file was called with ExpectedBucketOwner in ExtraArgs
        mock_s3_client.download_file.assert_called_once()
        call_args = mock_s3_client.download_file.call_args
        assert "ExtraArgs" in call_args.kwargs
        assert "ExpectedBucketOwner" in call_args.kwargs["ExtraArgs"]
        assert call_args.kwargs["ExtraArgs"]["ExpectedBucketOwner"] == "123456789012"


def test_bucket_owner_parameter_included_in_list(s3_artifact_root, monkeypatch):
    """
    Test that ExpectedBucketOwner is included in list operations when configured.

    This prevents listing artifacts from an attacker's bucket with the same name.
    """
    monkeypatch.setenv("MLFLOW_S3_BUCKET_OWNER", "123456789012")

    repo = S3ArtifactRepository(s3_artifact_root)

    # Mock the S3 client and paginator
    with mock.patch.object(repo, "_get_s3_client") as mock_get_client:
        mock_s3_client = mock.Mock()
        mock_paginator = mock.Mock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = []
        mock_get_client.return_value = mock_s3_client

        repo.list_artifacts()

        # Verify paginate was called with ExpectedBucketOwner
        mock_paginator.paginate.assert_called_once()
        call_kwargs = mock_paginator.paginate.call_args.kwargs
        assert "ExpectedBucketOwner" in call_kwargs
        assert call_kwargs["ExpectedBucketOwner"] == "123456789012"


def test_bucket_owner_parameter_included_in_delete(s3_artifact_root, monkeypatch):
    """
    Test that ExpectedBucketOwner is included in delete operations when configured.

    This prevents deleting artifacts from an attacker's bucket with the same name.
    """
    monkeypatch.setenv("MLFLOW_S3_BUCKET_OWNER", "123456789012")

    # Use a repo with a path to match the test structure
    repo = S3ArtifactRepository(posixpath.join(s3_artifact_root, "some/path"))

    # Mock the S3 client and paginator
    with mock.patch.object(repo, "_get_s3_client") as mock_get_client:
        mock_s3_client = mock.Mock()
        mock_paginator = mock.Mock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        # Return a key that matches the artifact path
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "some/path/test.txt"}]}
        ]
        mock_get_client.return_value = mock_s3_client

        repo.delete_artifacts("test.txt")

        # Verify both list and delete operations include ExpectedBucketOwner
        paginate_kwargs = mock_paginator.paginate.call_args.kwargs
        assert "ExpectedBucketOwner" in paginate_kwargs
        assert paginate_kwargs["ExpectedBucketOwner"] == "123456789012"

        delete_kwargs = mock_s3_client.delete_objects.call_args.kwargs
        assert "ExpectedBucketOwner" in delete_kwargs
        assert delete_kwargs["ExpectedBucketOwner"] == "123456789012"


def test_bucket_owner_parameter_included_in_multipart_upload(
    s3_artifact_root, monkeypatch
):
    """
    Test that ExpectedBucketOwner is included in multipart upload operations.

    This prevents uploading large artifacts to an attacker's bucket with the same name.
    """
    monkeypatch.setenv("MLFLOW_S3_BUCKET_OWNER", "123456789012")

    repo = S3ArtifactRepository(s3_artifact_root)

    # Mock the S3 client
    with mock.patch.object(repo, "_get_s3_client") as mock_get_client:
        mock_s3_client = mock.Mock()
        mock_s3_client.create_multipart_upload.return_value = {
            "UploadId": "test-upload-id"
        }
        mock_get_client.return_value = mock_s3_client

        # Test create_multipart_upload
        repo.create_multipart_upload("test.txt", num_parts=2)

        # Verify create_multipart_upload was called with ExpectedBucketOwner
        create_kwargs = mock_s3_client.create_multipart_upload.call_args.kwargs
        assert "ExpectedBucketOwner" in create_kwargs
        assert create_kwargs["ExpectedBucketOwner"] == "123456789012"

        # Verify presigned URLs include ExpectedBucketOwner
        presign_calls = mock_s3_client.generate_presigned_url.call_args_list
        for call in presign_calls:
            params = call.kwargs.get("Params", {})
            assert "ExpectedBucketOwner" in params
            assert params["ExpectedBucketOwner"] == "123456789012"


def test_bucket_owner_parameter_included_in_complete_multipart(
    s3_artifact_root, monkeypatch
):
    """
    Test that ExpectedBucketOwner is included in complete multipart upload operations.
    """
    monkeypatch.setenv("MLFLOW_S3_BUCKET_OWNER", "123456789012")

    repo = S3ArtifactRepository(s3_artifact_root)

    with mock.patch.object(repo, "_get_s3_client") as mock_get_client:
        mock_s3_client = mock.Mock()
        mock_get_client.return_value = mock_s3_client

        parts = [MultipartUploadPart(part_number=1, etag="etag1")]
        repo.complete_multipart_upload("test.txt", "upload-id", parts=parts)

        # Verify complete_multipart_upload was called with ExpectedBucketOwner
        complete_kwargs = mock_s3_client.complete_multipart_upload.call_args.kwargs
        assert "ExpectedBucketOwner" in complete_kwargs
        assert complete_kwargs["ExpectedBucketOwner"] == "123456789012"


def test_bucket_owner_parameter_included_in_abort_multipart(
    s3_artifact_root, monkeypatch
):
    """
    Test that ExpectedBucketOwner is included in abort multipart upload operations.
    """
    monkeypatch.setenv("MLFLOW_S3_BUCKET_OWNER", "123456789012")

    repo = S3ArtifactRepository(s3_artifact_root)

    with mock.patch.object(repo, "_get_s3_client") as mock_get_client:
        mock_s3_client = mock.Mock()
        mock_get_client.return_value = mock_s3_client

        repo.abort_multipart_upload("test.txt", "upload-id")

        # Verify abort_multipart_upload was called with ExpectedBucketOwner
        abort_kwargs = mock_s3_client.abort_multipart_upload.call_args.kwargs
        assert "ExpectedBucketOwner" in abort_kwargs
        assert abort_kwargs["ExpectedBucketOwner"] == "123456789012"


def test_bucket_owner_from_constructor_parameter(s3_artifact_root, tmp_path):
    """
    Test that bucket owner can be set via constructor parameter.

    This allows programmatic control over bucket ownership verification.
    """
    repo = S3ArtifactRepository(s3_artifact_root, expected_bucket_owner="999888777666")

    assert repo._expected_bucket_owner == "999888777666"
    assert repo._get_bucket_owner_params() == {"ExpectedBucketOwner": "999888777666"}

    # Verify it's used in operations
    file_path = tmp_path / "test.txt"
    file_path.write_text("test content")

    with mock.patch.object(repo, "_get_s3_client") as mock_get_client:
        mock_s3_client = mock.Mock()
        mock_get_client.return_value = mock_s3_client

        repo.log_artifact(str(file_path))

        call_kwargs = mock_s3_client.upload_file.call_args.kwargs
        assert call_kwargs["ExtraArgs"]["ExpectedBucketOwner"] == "999888777666"


def test_bucket_owner_constructor_overrides_environment(s3_artifact_root, monkeypatch):
    """
    Test that constructor parameter takes precedence over environment variable.
    """
    monkeypatch.setenv("MLFLOW_S3_BUCKET_OWNER", "123456789012")

    repo = S3ArtifactRepository(s3_artifact_root, expected_bucket_owner="999888777666")

    # Constructor parameter should override environment variable
    assert repo._expected_bucket_owner == "999888777666"
    assert repo._get_bucket_owner_params() == {"ExpectedBucketOwner": "999888777666"}


def test_bucket_owner_not_included_when_not_configured(
    s3_artifact_root, tmp_path, monkeypatch
):
    """
    Test backward compatibility: ExpectedBucketOwner is not included when not configured.

    This ensures existing code continues to work without modifications.
    """
    # Ensure environment variable is not set
    monkeypatch.delenv("MLFLOW_S3_BUCKET_OWNER", raising=False)

    repo = S3ArtifactRepository(s3_artifact_root)

    assert repo._expected_bucket_owner is None
    assert repo._get_bucket_owner_params() == {}

    # Verify it's not included in operations
    file_path = tmp_path / "test.txt"
    file_path.write_text("test content")

    with mock.patch.object(repo, "_get_s3_client") as mock_get_client:
        mock_s3_client = mock.Mock()
        mock_get_client.return_value = mock_s3_client

        repo.log_artifact(str(file_path))

        call_kwargs = mock_s3_client.upload_file.call_args.kwargs
        # ExpectedBucketOwner should not be in ExtraArgs
        assert "ExpectedBucketOwner" not in call_kwargs.get("ExtraArgs", {})


def test_bucket_impersonation_attack_scenario(s3_artifact_root, tmp_path, monkeypatch):
    """
    Test the complete bucket impersonation attack scenario.

    Scenario:
    1. User creates and uses bucket 'my-mlflow-artifacts' (account 123456789012)
    2. Bucket is deleted
    3. Attacker creates bucket with same name (account 999888777666)
    4. MLflow with ExpectedBucketOwner prevents access to attacker's bucket

    Without ExpectedBucketOwner: Artifacts would be sent to attacker's bucket
    With ExpectedBucketOwner: AWS returns AccessDenied error
    """
    # Simulate legitimate user's bucket with ownership verification
    monkeypatch.setenv("MLFLOW_S3_BUCKET_OWNER", "123456789012")

    repo = S3ArtifactRepository(s3_artifact_root)

    file_path = tmp_path / "sensitive_model.pkl"
    file_path.write_text("sensitive model data")

    # Mock S3 client to simulate AccessDenied when bucket owner doesn't match
    with mock.patch.object(repo, "_get_s3_client") as mock_get_client:
        mock_s3_client = mock.Mock()

        # Simulate AWS returning AccessDenied because bucket is owned by attacker
        error_response = {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}
        mock_s3_client.upload_file.side_effect = botocore.exceptions.ClientError(
            error_response, "PutObject"
        )
        mock_get_client.return_value = mock_s3_client

        # Attempt to upload should fail with AccessDenied
        with pytest.raises(botocore.exceptions.ClientError) as exc_info:
            repo.log_artifact(str(file_path))

        assert exc_info.value.response["Error"]["Code"] == "AccessDenied"

        # Verify ExpectedBucketOwner was included in the request
        call_kwargs = mock_s3_client.upload_file.call_args.kwargs
        assert call_kwargs["ExtraArgs"]["ExpectedBucketOwner"] == "123456789012"


def test_optimized_s3_repo_bucket_owner_parameter(
    s3_artifact_root, tmp_path, monkeypatch
):
    """
    Test that OptimizedS3ArtifactRepository also includes ExpectedBucketOwner.

    Both S3 repository implementations must have the same security protection.
    """
    monkeypatch.setenv("MLFLOW_S3_BUCKET_OWNER", "123456789012")

    repo = OptimizedS3ArtifactRepository(s3_artifact_root)

    assert repo._get_bucket_owner_params() == {"ExpectedBucketOwner": "123456789012"}

    # Verify it's used in upload operations
    file_path = tmp_path / "test.txt"
    file_path.write_text("test content")

    with mock.patch.object(repo, "_get_s3_client") as mock_get_client:
        mock_s3_client = mock.Mock()
        mock_get_client.return_value = mock_s3_client

        repo._upload_file(mock_s3_client, str(file_path), "bucket", "key")

        call_kwargs = mock_s3_client.upload_file.call_args.kwargs
        assert call_kwargs["ExtraArgs"]["ExpectedBucketOwner"] == "123456789012"


def test_bucket_owner_with_real_s3_operations(s3_artifact_repo, tmp_path, monkeypatch):
    """
    Integration test: Verify bucket owner parameter works with real S3 operations.

    This test uses the actual S3 mock bucket to ensure the parameter doesn't
    break existing functionality.
    """
    # Set a bucket owner (won't be validated by moto, but ensures parameter is passed)
    monkeypatch.setenv("MLFLOW_S3_BUCKET_OWNER", "123456789012")

    # Create a new repo instance to pick up the environment variable
    repo = S3ArtifactRepository(s3_artifact_repo.artifact_uri)

    # Test upload
    file_path = tmp_path / "test.txt"
    file_path.write_text("test content")
    repo.log_artifact(str(file_path))

    # Test list
    artifacts = repo.list_artifacts()
    assert len(artifacts) > 0

    # Test download
    downloaded = repo.download_artifacts("test.txt")
    with open(downloaded) as f:
        assert f.read() == "test content"

    # Test delete
    repo.delete_artifacts("test.txt")
    artifacts_after_delete = repo.list_artifacts()
    assert len(artifacts_after_delete) == 0
