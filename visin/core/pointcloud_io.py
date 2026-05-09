from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib
from pathlib import Path

import numpy as np


class UnsupportedPointCloudFormatError(ValueError):
    pass


class MissingPointCloudDependencyError(RuntimeError):
    pass


class InvalidPointCloudError(RuntimeError):
    pass


@dataclass(frozen=True)
class PointCloudData:
    points: np.ndarray
    source_path: Path
    format: str
    point_count: int


@dataclass(frozen=True)
class PointCloudLoadOptions:
    format_override: str | None = None
    drop_non_finite: bool = True


@dataclass(frozen=True)
class PcdLoadOptions(PointCloudLoadOptions):
    pass


@dataclass(frozen=True)
class PlyLoadOptions(PointCloudLoadOptions):
    pass


@dataclass(frozen=True)
class DelimitedTextLoadOptions(PointCloudLoadOptions):
    delimiter: str | None = None
    skip_rows: int = 0
    xyz_columns: tuple[int, int, int] = (0, 1, 2)
    detect_pts_header: bool = True


class BasePointCloudFormatReader(ABC):
    supported_formats: tuple[str, ...] = ()
    options_type = PointCloudLoadOptions

    @classmethod
    def supports(cls, format_name: str) -> bool:
        return format_name in cls.supported_formats

    @classmethod
    @abstractmethod
    def default_options(cls) -> PointCloudLoadOptions:
        raise NotImplementedError

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        format_name: str,
        options: PointCloudLoadOptions,
    ) -> PointCloudData:
        raw_points = cls._read_points(path, format_name=format_name, options=options)
        points = _finalize_points(
            raw_points,
            path=path,
            format_name=format_name,
            drop_non_finite=options.drop_non_finite,
        )
        return PointCloudData(
            points=points,
            source_path=path,
            format=format_name,
            point_count=points.shape[0],
        )

    @classmethod
    @abstractmethod
    def _read_points(
        cls,
        path: Path,
        *,
        format_name: str,
        options: PointCloudLoadOptions,
    ) -> np.ndarray:
        raise NotImplementedError


class PcdPointCloudReader(BasePointCloudFormatReader):
    supported_formats = ("pcd",)
    options_type = PcdLoadOptions

    @classmethod
    def default_options(cls) -> PcdLoadOptions:
        return PcdLoadOptions()

    @classmethod
    def _read_points(
        cls,
        path: Path,
        *,
        format_name: str,
        options: PointCloudLoadOptions,
    ) -> np.ndarray:
        del format_name, options

        try:
            point_cloud_module = importlib.import_module("pypcd4.pypcd4")
        except ImportError as exc:
            raise MissingPointCloudDependencyError(
                "PCD support requires the optional dependency 'pypcd4'."
            ) from exc

        try:
            point_cloud = point_cloud_module.PointCloud.from_path(path)
            return point_cloud.numpy(("x", "y", "z"))
        except Exception as exc:
            raise InvalidPointCloudError(
                f"Failed to read PCD point cloud '{path}': {exc}"
            ) from exc


class PlyPointCloudReader(BasePointCloudFormatReader):
    supported_formats = ("ply",)
    options_type = PlyLoadOptions

    @classmethod
    def default_options(cls) -> PlyLoadOptions:
        return PlyLoadOptions()

    @classmethod
    def _read_points(
        cls,
        path: Path,
        *,
        format_name: str,
        options: PointCloudLoadOptions,
    ) -> np.ndarray:
        del format_name, options

        try:
            ply_module = importlib.import_module("plyfile")
        except ImportError as exc:
            raise MissingPointCloudDependencyError(
                "PLY support requires the optional dependency 'plyfile'."
            ) from exc

        try:
            ply_data = ply_module.PlyData.read(path)
            vertex_data = ply_data["vertex"].data
            if vertex_data.dtype.names is None:
                raise InvalidPointCloudError(
                    f"PLY file '{path}' does not expose named vertex properties"
                )
            required_fields = ("x", "y", "z")
            missing_fields = [
                field for field in required_fields if field not in vertex_data.dtype.names
            ]
            if missing_fields:
                missing_display = ", ".join(missing_fields)
                raise InvalidPointCloudError(
                    f"PLY file '{path}' is missing required vertex fields: {missing_display}"
                )
            return np.column_stack([vertex_data[field] for field in required_fields])
        except InvalidPointCloudError:
            raise
        except Exception as exc:
            raise InvalidPointCloudError(
                f"Failed to read PLY point cloud '{path}': {exc}"
            ) from exc


class DelimitedTextPointCloudReader(BasePointCloudFormatReader):
    supported_formats = ("asc", "csv", "pts", "txt", "xyz")
    options_type = DelimitedTextLoadOptions

    @classmethod
    def default_options(cls) -> DelimitedTextLoadOptions:
        return DelimitedTextLoadOptions()

    @classmethod
    def _read_points(
        cls,
        path: Path,
        *,
        format_name: str,
        options: PointCloudLoadOptions,
    ) -> np.ndarray:
        if not isinstance(options, DelimitedTextLoadOptions):
            raise InvalidPointCloudError(
                f"Delimited text reader requires {DelimitedTextLoadOptions.__name__}"
            )

        effective_skip_rows = options.skip_rows
        if options.detect_pts_header and format_name == "pts":
            effective_skip_rows += _detect_pts_count_header(path)

        delimiter = options.delimiter
        if delimiter is None and format_name == "csv":
            delimiter = ","

        try:
            points = np.loadtxt(path, delimiter=delimiter, skiprows=effective_skip_rows)
        except Exception as exc:
            raise InvalidPointCloudError(
                f"Failed to read text point cloud '{path}': {exc}"
            ) from exc

        if points.ndim == 1:
            points = points.reshape(1, -1)

        max_column_index = max(options.xyz_columns)
        if points.shape[1] <= max_column_index:
            raise InvalidPointCloudError(
                f"Point cloud text file '{path}' does not contain columns {options.xyz_columns}"
            )

        return points[:, options.xyz_columns]


class PointCloudReader:
    _readers: tuple[type[BasePointCloudFormatReader], ...] = (
        PcdPointCloudReader,
        PlyPointCloudReader,
        DelimitedTextPointCloudReader,
    )
    _formats_to_readers = {
        format_name: reader
        for reader in _readers
        for format_name in reader.supported_formats
    }

    @classmethod
    def load(
        cls,
        path: str | Path,
        options: PointCloudLoadOptions | None = None,
    ) -> PointCloudData:
        source_path = Path(path).expanduser()
        if not source_path.exists():
            raise InvalidPointCloudError(f"Point cloud path does not exist: '{source_path}'")

        format_name = cls._resolve_format_name(source_path, options)
        reader = cls._formats_to_readers[format_name]
        resolved_options = cls._resolve_options(reader, options)
        return reader.load(source_path, format_name=format_name, options=resolved_options)

    @classmethod
    def supports(cls, path_or_suffix: str | Path) -> bool:
        try:
            cls._normalize_format_name(path_or_suffix)
        except UnsupportedPointCloudFormatError:
            return False
        return True

    @classmethod
    def _resolve_format_name(
        cls,
        path: Path,
        options: PointCloudLoadOptions | None,
    ) -> str:
        if options is not None and options.format_override is not None:
            return cls._normalize_format_name(options.format_override)
        return cls._normalize_format_name(path)

    @classmethod
    def _normalize_format_name(cls, path_or_suffix: str | Path) -> str:
        raw_value = str(path_or_suffix).strip().lower()
        if not raw_value:
            cls._raise_unsupported_format(raw_value)

        if raw_value.startswith(".") and raw_value.count(".") == 1:
            suffix = raw_value
        elif "." in raw_value:
            suffix = Path(raw_value).suffix.lower()
        else:
            suffix = raw_value

        format_name = suffix.lstrip(".")
        if format_name not in cls._formats_to_readers:
            cls._raise_unsupported_format(path_or_suffix)
        return format_name

    @classmethod
    def _raise_unsupported_format(cls, path_or_suffix: str | Path) -> None:
        supported = ", ".join(sorted(cls._formats_to_readers))
        raise UnsupportedPointCloudFormatError(
            f"Unsupported point cloud format '{path_or_suffix}'. Supported formats: {supported}"
        )

    @classmethod
    def _resolve_options(
        cls,
        reader: type[BasePointCloudFormatReader],
        options: PointCloudLoadOptions | None,
    ) -> PointCloudLoadOptions:
        if options is None:
            return reader.default_options()

        if isinstance(options, reader.options_type):
            return options

        if type(options) is PointCloudLoadOptions:
            return reader.options_type(
                format_override=options.format_override,
                drop_non_finite=options.drop_non_finite,
            )

        raise InvalidPointCloudError(
            f"Options type '{type(options).__name__}' is incompatible with reader "
            f"'{reader.__name__}'"
        )


def load_point_cloud(
    path: str | Path,
    options: PointCloudLoadOptions | None = None,
) -> np.ndarray:
    return PointCloudReader.load(path, options).points


def _detect_pts_count_header(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()

    if not first_line:
        return 0

    first_line_fields = first_line.split()
    if len(first_line_fields) != 1:
        return 0

    try:
        int(first_line_fields[0])
    except ValueError:
        return 0

    return 1


def _finalize_points(
    points: np.ndarray,
    *,
    path: Path,
    format_name: str,
    drop_non_finite: bool,
) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise InvalidPointCloudError(
            f"Point cloud '{path}' ({format_name}) did not resolve to an Nx3 array of xyz values"
        )

    if drop_non_finite:
        points = points[np.isfinite(points).all(axis=1)]
    elif not np.isfinite(points).all():
        raise InvalidPointCloudError(
            f"Point cloud '{path}' ({format_name}) contains non-finite xyz values"
        )

    if points.size == 0:
        raise InvalidPointCloudError(
            f"Point cloud '{path}' ({format_name}) does not contain any usable xyz points"
        )

    return np.ascontiguousarray(points, dtype=np.float32)


__all__ = [
    "DelimitedTextLoadOptions",
    "InvalidPointCloudError",
    "MissingPointCloudDependencyError",
    "PcdLoadOptions",
    "PlyLoadOptions",
    "PointCloudData",
    "PointCloudLoadOptions",
    "PointCloudReader",
    "UnsupportedPointCloudFormatError",
    "load_point_cloud",
]
