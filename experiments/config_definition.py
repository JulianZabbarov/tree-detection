import dataclasses


@dataclasses.dataclass
class DataConfig:
    path_to_images: str
    image_size: int


@dataclasses.dataclass
class ExportConfig:
    annotations_path: str
    type: str
    annotations_format: str
    image_format: str
    sort_values: list[str]
    column_order: list[str]
    index_as_label_suffix: bool
    image_path: str | None = None


@dataclasses.dataclass
class PipelineConfig:
    data: DataConfig
    export: ExportConfig
