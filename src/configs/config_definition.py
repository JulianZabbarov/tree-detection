import dataclasses


@dataclasses.dataclass
class TrainingConfig:
    annotations_folder: str
    images_folder: str
    patch_size: int
    unsupervised_annotations_folder: str = None
    unsupervised_images_folder: str = None
    num_epochs: int = 10
    seeds: list[int] = dataclasses.field(default_factory=lambda: [0, 1, 2, 3, 4])


@dataclasses.dataclass
class DataConfig:
    path_to_images: str
    predict_tile: bool
    tile_size: int | None = None


@dataclasses.dataclass
class ExportConfig:
    annotations_path: str
    type: str
    annotations_format: str
    image_format: str
    sort_values: list[str]
    column_order: list[str]
    index_as_label_suffix: bool
    image_size: int | None = None
    image_path: str | None = None
    plot_predictions: bool = False


@dataclasses.dataclass
class VisualizationConfig:
    image_folder: str
    image_format: str
    label_folder: str
    export_folder: str
    predictions_folder: str | None = None


@dataclasses.dataclass
class PipelineConfig:
    data: DataConfig | None = None
    training: TrainingConfig | None = None
    export: ExportConfig | None = None
    visualization: VisualizationConfig | None = None
