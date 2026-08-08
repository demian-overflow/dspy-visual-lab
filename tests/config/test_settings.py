from config.settings import Settings


def test_settings_has_langfuse_fields_with_cloud_default():
    s = Settings(_env_file=None)
    assert s.langfuse_public_key is None
    assert s.langfuse_secret_key is None
    assert s.langfuse_host == "https://cloud.langfuse.com"
