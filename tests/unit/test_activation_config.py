import runpy


def test_activation_preserves_other_values_and_original_backup(tmp_path):
    path = tmp_path / ".env"
    original = "DATABASE_URL=fixture-not-a-real-secret\nALLOW_MOCK_AI=true\n# keep comment\n"
    path.write_text(original)
    update = runpy.run_path("scripts/activate_real_model.py")["update_environment"]
    backup = update(path, {"ALLOW_MOCK_AI": "false", "DEFAULT_MODEL": "production"})
    assert backup.read_text() == original
    assert "DATABASE_URL=fixture-not-a-real-secret" in path.read_text()
    assert "ALLOW_MOCK_AI=false" in path.read_text()
    assert "DEFAULT_MODEL=production" in path.read_text()
    assert "# keep comment" in path.read_text()
