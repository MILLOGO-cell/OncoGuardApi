# tests/test_files_integration.py - Tests de validation de la correction

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestFileEndpointsIntegration:
    """
    Tests d'intégration pour valider que la correction backend/frontend fonctionne
    """

    def test_list_files_with_pgm_kind(self):
        """✅ Test que kind="pgm" fonctionne maintenant"""
        response = client.get("/api/v1/ingest/files?kind=pgm")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # Vérifier que tous les items ont kind="pgm"
        for item in data:
            assert item["kind"] == "pgm"
            assert "filename" in item
            assert "size_bytes" in item
            assert "created_at" in item
            assert "download_url" in item

    def test_list_files_with_dicom_kind(self):
        """✅ Test que kind="dicom" fonctionne toujours"""
        response = client.get("/api/v1/ingest/files?kind=dicom")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        for item in data:
            assert item["kind"] == "dicom"

    def test_list_files_rejects_old_png_kind(self):
        """❌ Test que kind="png" renvoie une erreur maintenant"""
        response = client.get("/api/v1/ingest/files?kind=png")
        assert response.status_code == 400
        data = response.json()
        assert "kind doit être parmi" in data["detail"]
        assert "pgm" in data["detail"]

    def test_list_files_all_kinds(self):
        """✅ Test que sans kind, on reçoit tous les fichiers"""
        response = client.get("/api/v1/ingest/files")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # Vérifier que les kinds retournés sont valides
        valid_kinds = {"pgm", "dicom", "tagged"}
        for item in data:
            assert item["kind"] in valid_kinds

    def test_download_pgm_file(self):
        """✅ Test téléchargement d'un fichier PGM"""
        # D'abord obtenir un fichier existant
        response = client.get("/api/v1/ingest/files?kind=pgm&limit=1")
        assert response.status_code == 200
        data = response.json()
        
        if len(data) > 0:
            filename = data[0]["filename"]
            response = client.get(f"/api/v1/ingest/download/pgm/{filename}")
            assert response.status_code == 200
            assert response.headers["content-type"] == "image/png"

    def test_download_with_old_png_kind_fails(self):
        """❌ Test que download avec kind="png" échoue maintenant"""
        response = client.get("/api/v1/ingest/download/png/test.png")
        assert response.status_code == 400

    def test_export_zip_with_pgm(self):
        """✅ Test export ZIP avec kind="pgm" """
        response = client.post(
            "/api/v1/ingest/export/zip",
            params={"kind": "pgm", "all_files": True}
        )
        # Peut être 200 (si fichiers) ou 404 (si aucun fichier)
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            assert response.headers["content-type"] == "application/zip"

    def test_export_zip_rejects_png_kind(self):
        """❌ Test que export ZIP avec kind="png" échoue"""
        response = client.post(
            "/api/v1/ingest/export/zip",
            params={"kind": "png", "all_files": True}
        )
        assert response.status_code == 400

    def test_pydantic_validation_rejects_invalid_kind(self):
        """✅ Test que la validation Pydantic rejette les valeurs invalides"""
        from app.api.v1.schemas.files import FileItem
        from pydantic import ValidationError
        
        # Devrait fonctionner avec "pgm"
        valid_item = FileItem(
            kind="pgm",
            filename="test.png",
            size_bytes=1024,
            created_at=1234567890.0,
            download_url="/download/pgm/test.png"
        )
        assert valid_item.kind == "pgm"
        
        # Devrait échouer avec "png"
        with pytest.raises(ValidationError) as exc_info:
            FileItem(
                kind="png",  # ❌ Invalide maintenant
                filename="test.png",
                size_bytes=1024,
                created_at=1234567890.0,
                download_url="/download/png/test.png"
            )
        
        error = exc_info.value
        assert "kind" in str(error)


# Tests pour le frontend TypeScript (format Jest/Vitest)
"""
// tests/filesApi.test.ts

import { describe, it, expect, vi } from 'vitest';
import { fetchFileList, downloadFileUrl, FileKind } from '@/lib/api/filesApi';

describe('filesApi', () => {
  it('should accept "pgm" as valid FileKind', () => {
    const kind: FileKind = 'pgm';
    expect(kind).toBe('pgm');
  });

  it('should accept "dicom" as valid FileKind', () => {
    const kind: FileKind = 'dicom';
    expect(kind).toBe('dicom');
  });

  it('should NOT accept "png" as FileKind (TypeScript compile error)', () => {
    // @ts-expect-error - "png" is no longer a valid kind
    const kind: FileKind = 'png';
    // Ce test ne devrait pas compiler si TypeScript est strict
  });

  it('should generate correct download URL for pgm', () => {
    const url = downloadFileUrl('pgm', 'bfa001.png');
    expect(url).toContain('/download/pgm/bfa001.png');
  });

  it('should fetch pgm files successfully', async () => {
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve([
          {
            kind: 'pgm',
            filename: 'bfa001.png',
            size_bytes: 1024,
            created_at: 1234567890,
            download_url: '/api/v1/ingest/download/pgm/bfa001.png'
          }
        ]),
      })
    ) as any;

    const files = await fetchFileList({ kind: 'pgm' });
    expect(files).toHaveLength(1);
    expect(files[0].kind).toBe('pgm');
  });
});
"""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])