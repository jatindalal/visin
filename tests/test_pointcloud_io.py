import tempfile
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from visin.core.pointcloud_io import (
    DelimitedTextLoadOptions,
    InvalidPointCloudError,
    MissingPointCloudDependencyError,
    PcdLoadOptions,
    PlyLoadOptions,
    PointCloudLoadOptions,
    PointCloudReader,
    UnsupportedPointCloudFormatError,
    load_point_cloud,
)


class PointCloudIoTests(unittest.TestCase):
    def test_supports_known_suffixes(self):
        self.assertTrue(PointCloudReader.supports(".pcd"))
        self.assertTrue(PointCloudReader.supports("scan.csv"))
        self.assertTrue(PointCloudReader.supports("XYZ"))
        self.assertFalse(PointCloudReader.supports("scan.las"))

    def test_unsupported_format_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.las"
            path.write_text("0 0 0\n", encoding="utf-8")

            with self.assertRaises(UnsupportedPointCloudFormatError):
                PointCloudReader.load(path)

    def test_text_reader_loads_xyz_and_returns_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.xyz"
            path.write_text("1 2 3\n4 5 6\n", encoding="utf-8")

            result = PointCloudReader.load(path)

        self.assertEqual(result.format, "xyz")
        self.assertEqual(result.point_count, 2)
        self.assertEqual(result.source_path, path)
        self.assertEqual(result.points.dtype, np.float32)
        self.assertTrue(result.points.flags.c_contiguous)
        np.testing.assert_array_equal(
            result.points,
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        )

    def test_format_override_takes_precedence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.data"
            path.write_text("1,2,3\n4,5,6\n", encoding="utf-8")

            result = PointCloudReader.load(
                path,
                DelimitedTextLoadOptions(format_override="csv"),
            )

        self.assertEqual(result.format, "csv")
        np.testing.assert_array_equal(
            result.points,
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        )

    def test_text_reader_honors_column_mapping_and_skip_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.txt"
            path.write_text("# header\n10 1 2 3\n11 4 5 6\n", encoding="utf-8")

            result = PointCloudReader.load(
                path,
                DelimitedTextLoadOptions(skip_rows=1, xyz_columns=(1, 2, 3)),
            )

        np.testing.assert_array_equal(
            result.points,
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        )

    def test_pts_header_detection_can_be_enabled_or_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.pts"
            path.write_text("2\n1 2 3\n4 5 6\n", encoding="utf-8")

            result = PointCloudReader.load(path)
            self.assertEqual(result.point_count, 2)

            with self.assertRaises(InvalidPointCloudError):
                PointCloudReader.load(
                    path,
                    DelimitedTextLoadOptions(detect_pts_header=False),
                )

    def test_non_finite_values_are_dropped_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.xyz"
            path.write_text("1 2 3\nnan 4 5\n6 7 8\n", encoding="utf-8")

            result = PointCloudReader.load(path)

        self.assertEqual(result.point_count, 2)
        np.testing.assert_array_equal(
            result.points,
            np.array([[1, 2, 3], [6, 7, 8]], dtype=np.float32),
        )

    def test_non_finite_values_can_fail_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.xyz"
            path.write_text("1 2 3\nnan 4 5\n", encoding="utf-8")

            with self.assertRaises(InvalidPointCloudError):
                PointCloudReader.load(
                    path,
                    DelimitedTextLoadOptions(drop_non_finite=False),
                )

    def test_incompatible_options_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.xyz"
            path.write_text("1 2 3\n", encoding="utf-8")

            with self.assertRaises(InvalidPointCloudError):
                PointCloudReader.load(path, PcdLoadOptions())

    def test_load_point_cloud_returns_points_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.csv"
            path.write_text("1,2,3\n", encoding="utf-8")

            points = load_point_cloud(path)

        np.testing.assert_array_equal(points, np.array([[1, 2, 3]], dtype=np.float32))

    def test_pcd_reader_loads_with_lazy_dependency(self):
        fake_module = ModuleType("pypcd4.pypcd4")
        captured = {}

        class FakePointCloud:
            @staticmethod
            def from_path(path):
                captured["path_name"] = path.name
                return FakePointCloud()

            def numpy(self, fields):
                captured["fields"] = fields
                return np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

        fake_module.PointCloud = FakePointCloud

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.pcd"
            path.write_text("VERSION .7\n", encoding="utf-8")

            with patch("importlib.import_module", return_value=fake_module):
                result = PointCloudReader.load(path)

        self.assertEqual(result.format, "pcd")
        self.assertEqual(result.point_count, 2)
        self.assertEqual(captured["path_name"], "scan.pcd")
        self.assertEqual(captured["fields"], ("x", "y", "z"))

    def test_pcd_reader_reports_missing_dependency(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.pcd"
            path.write_text("VERSION .7\n", encoding="utf-8")

            with patch(
                "importlib.import_module",
                side_effect=ImportError("missing pypcd4"),
            ):
                with self.assertRaises(MissingPointCloudDependencyError):
                    PointCloudReader.load(path)

    def test_ply_reader_loads_with_lazy_dependency(self):
        fake_module = ModuleType("plyfile")

        class FakePlyData:
            @staticmethod
            def read(path):
                self.assertEqual(path.name, "scan.ply")
                vertex_data = np.array(
                    [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
                    dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
                )
                return {"vertex": type("Vertex", (), {"data": vertex_data})()}

        fake_module.PlyData = FakePlyData

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.ply"
            path.write_text("ply\n", encoding="utf-8")

            with patch("importlib.import_module", return_value=fake_module):
                result = PointCloudReader.load(path)

        self.assertEqual(result.format, "ply")
        self.assertEqual(result.point_count, 2)

    def test_ply_reader_reports_missing_dependency(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.ply"
            path.write_text("ply\n", encoding="utf-8")

            with patch(
                "importlib.import_module",
                side_effect=ImportError("missing plyfile"),
            ):
                with self.assertRaises(MissingPointCloudDependencyError):
                    PointCloudReader.load(path)

    def test_base_options_upgrade_to_reader_specific_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.csv"
            path.write_text("1,2,3\n", encoding="utf-8")

            result = PointCloudReader.load(
                path,
                PointCloudLoadOptions(format_override="csv"),
            )

        self.assertEqual(result.point_count, 1)

    def test_missing_path_raises_invalid_point_cloud(self):
        with self.assertRaises(InvalidPointCloudError):
            PointCloudReader.load("/tmp/does-not-exist.xyz")

    def test_text_reader_rejects_invalid_column_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.xyz"
            path.write_text("1 2\n", encoding="utf-8")

            with self.assertRaises(InvalidPointCloudError):
                PointCloudReader.load(path)


if __name__ == "__main__":
    unittest.main()
