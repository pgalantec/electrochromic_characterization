"""
Interactive LEGO patch selector application using Dash and Plotly.
This app allows users to select regions on the first frame of a video
for LEGO color calibration patches and region of interest for mask generation.
"""

import glob
import json
from datetime import datetime
from pathlib import Path

import cv2
import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import skimage.io
import skimage.morphology
import yaml
from dash import Input, Output, State, dcc, html
from skimage import draw

from src.utils.io import load_params

# LEGO color information with real-world references
LEGO_COLOR_INFO = {
    "black": {
        "rgb": [18, 18, 21],
        "name": "Black",
        "hex": "#121215",
        "description": "LEGO Black (Element ID: 6179803)",
    },
    "dark_grey": {
        "rgb": [98, 101, 102],
        "name": "Dark Stone Grey",
        "hex": "#626566",
        "description": "LEGO Dark Stone Grey (Element ID: 4211398)",
    },
    "clear_grey": {
        "rgb": [160, 163, 164],
        "name": "Medium Stone Grey",
        "hex": "#A0A3A4",
        "description": "LEGO Medium Stone Grey (Element ID: 4211413)",
    },
    "white": {
        "rgb": [244, 238, 228],
        "name": "White",
        "hex": "#F4EEE4",
        "description": "LEGO White (Element ID: 302901)",
    },
}

# Order for patch selection (from darkest to lightest)
PATCH_ORDER = ["black", "dark_grey", "clear_grey", "white"]

# Selection modes
SELECTION_MODES = {"PATCHES": "patches", "ROI": "roi"}


class LegoSelectorApp:
    def __init__(self):
        # Initialize app with Bootstrap theme
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.LUX],
            suppress_callback_exceptions=True,
            # Cambiar la configuración de assets
            assets_folder="../assets",  # Ruta relativa desde src/
            assets_url_path="/assets/",  # URL path para servir assets
        )

        # Load current parameters
        self.params = load_params()

        # Setup layout and callbacks
        self.setup_layout()
        self.setup_callbacks()

    def get_dataset_videos(self):
        """Get all video files from the dataset directory."""
        dataset_path = Path("dataset")
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        video_files = []

        if dataset_path.exists():
            for ext in video_extensions:
                video_files.extend(glob.glob(str(dataset_path / "**" / ext), recursive=True))

        return [Path(f) for f in video_files]

    def setup_layout(self):
        """Setup the main application layout."""
        self.app.layout = dbc.Container(
            [
                # Header section with logos and title
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        # Left logo
                                                        dbc.Col(
                                                            [
                                                                html.Img(
                                                                    src="/assets/logo-bcmaterials.png",
                                                                    style={
                                                                        "height": "120px",
                                                                        "width": "auto",
                                                                        "object-fit": "contain",
                                                                    },
                                                                    className="img-fluid",
                                                                )
                                                            ],
                                                            width=2,
                                                            className="d-flex align-items-center justify-content-center",
                                                        ),
                                                        # Title
                                                        dbc.Col(
                                                            [
                                                                html.H1(
                                                                    "Electrochromic Characterization using LEGO bricks",
                                                                    className="text-primary text-center mb-2",
                                                                    style={
                                                                        "fontWeight": "800",
                                                                        "fontSize": "2.5rem",
                                                                        "textShadow": "1px 1px 2px rgba(0,0,0,0.1)",
                                                                    },
                                                                ),
                                                                html.P(
                                                                    "🧱 LEGO Color Patch & ROI Selector for Electrochromic Analysis",
                                                                    className="text-center text-muted mb-0",
                                                                    style={
                                                                        "fontSize": "1.1rem",
                                                                        "fontWeight": "500",
                                                                    },
                                                                ),
                                                            ],
                                                            width=8,
                                                            className="d-flex flex-column justify-content-center",
                                                        ),
                                                        # Right logo
                                                        dbc.Col(
                                                            [
                                                                html.Img(
                                                                    src="/assets/TECNALIA.png",
                                                                    style={
                                                                        "height": "180px",
                                                                        "width": "auto",
                                                                        "object-fit": "contain",
                                                                    },
                                                                    className="img-fluid",
                                                                )
                                                            ],
                                                            width=2,
                                                            className="d-flex align-items-center justify-content-center",
                                                        ),
                                                    ],
                                                    className="g-0",
                                                    align="center",
                                                ),
                                                html.Hr(className="my-3"),
                                                html.P(
                                                    "Select regions for LEGO color calibration patches and region of interest (ROI) "
                                                    "for mask generation in the first frame of your video. This tool enables precise "
                                                    "characterization of electrochromic materials using standardized LEGO color references.",
                                                    className="lead text-center",
                                                    style={"fontSize": "1rem", "lineHeight": "1.6"},
                                                ),
                                            ],
                                            style={"padding": "1.5rem"},
                                        ),
                                    ],
                                    style={
                                        "background": "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)",
                                        "border": "none",
                                        "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
                                    },
                                    className="mb-4",
                                )
                            ],
                            width=12,
                        )
                    ]
                ),
                # Video Configuration section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [html.H4("📹 Video Configuration", className="mb-0")]
                                        ),
                                        dbc.CardBody(
                                            [
                                                # Video selection
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Select video from dataset:"
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="video-dropdown",
                                                                    options=self.get_video_options(),
                                                                    placeholder="Choose a video from dataset...",
                                                                    className="mb-3",
                                                                ),
                                                            ],
                                                            width=12,
                                                        )
                                                    ]
                                                ),
                                                # Experiment configuration
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Experiment Name:"),
                                                                dbc.Input(
                                                                    id="exp-name-input",
                                                                    type="text",
                                                                    placeholder="Enter experiment name (e.g., ExpCalY-OptGamma)",
                                                                    value="ExpCalY-OptGamma",
                                                                    className="mb-3",
                                                                ),
                                                            ],
                                                            width=12,
                                                        )
                                                    ]
                                                ),
                                                # Frame type and calibration
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Frame Color Space:"),
                                                                dcc.Dropdown(
                                                                    id="frame-dtype-dropdown",
                                                                    options=[
                                                                        {
                                                                            "label": "Y (Luminance)",
                                                                            "value": "Y",
                                                                        },
                                                                        {
                                                                            "label": "Gray (Grayscale)",
                                                                            "value": "gray",
                                                                        },
                                                                        {
                                                                            "label": "L (Lab Lightness)",
                                                                            "value": "L",
                                                                        },
                                                                    ],
                                                                    value="Y",
                                                                    className="mb-3",
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                    ]
                                                ),
                                                # Buttons
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    "Load First Frame",
                                                                    id="load-frame-btn",
                                                                    color="primary",
                                                                    disabled=True,
                                                                    className="w-100",
                                                                ),
                                                            ],
                                                            width=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    "Reset All Selections",
                                                                    id="reset-btn",
                                                                    color="secondary",
                                                                    outline=True,
                                                                    disabled=True,
                                                                    className="w-100",
                                                                ),
                                                            ],
                                                            width=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                # Space for future buttons
                                                            ],
                                                            width=4,
                                                        ),
                                                    ]
                                                ),
                                                # Configuration preview
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Div(
                                                                    id="config-preview",
                                                                    className="mt-3",
                                                                ),
                                                            ],
                                                            width=12,
                                                        )
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="mb-4",
                                )
                            ],
                            width=12,
                        )
                    ]
                ),
                # Selection tabs
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Tabs(
                                    id="selection-tabs",
                                    value="patches-tab",
                                    children=[
                                        dcc.Tab(
                                            label="🧱 LEGO Patches",
                                            value="patches-tab",
                                            className="custom-tab",
                                        ),
                                        dcc.Tab(
                                            label="🎯 Region of Interest",
                                            value="roi-tab",
                                            className="custom-tab",
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                            ],
                            width=12,
                        )
                    ]
                ),
                # Current selection info
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Alert(
                                    id="current-selection-info",
                                    color="info",
                                    style={"display": "none"},
                                )
                            ],
                            width=12,
                        )
                    ]
                ),
                # Main content: Frame display + Controls
                dbc.Row(
                    [
                        # Frame display
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [html.H5(id="frame-title", className="mb-0")]
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="frame-display",
                                                    config={
                                                        "displayModeBar": True,
                                                        "displaylogo": False,
                                                        "modeBarButtonsToAdd": [
                                                            "drawrect",
                                                            "drawclosedpath",
                                                            "eraseshape",
                                                        ],
                                                    },
                                                    style={"height": "600px"},
                                                ),
                                                html.Div(id="selection-status", className="mt-2"),
                                                html.Div(
                                                    [
                                                        dbc.Button(
                                                            "Generate ROI Mask",
                                                            id="generate-mask-btn",
                                                            color="primary",
                                                            disabled=True,
                                                            className="mt-2 me-2",
                                                            style={"display": "none"},
                                                        ),
                                                        dbc.Button(
                                                            id="confirm-selection-btn",
                                                            children="Confirm Selection",
                                                            color="success",
                                                            disabled=True,
                                                            className="mt-2 me-2",
                                                        ),
                                                        dbc.Button(
                                                            "Clear All Annotations",
                                                            id="clear-annotations-btn",
                                                            color="warning",
                                                            outline=True,
                                                            disabled=True,
                                                            className="mt-2",
                                                        ),
                                                    ],
                                                    id="action-buttons",
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=8,
                        ),
                        # Control panel
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [html.H5(id="control-panel-title", className="mb-0")]
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.Div(id="tab-content"),
                                                html.Hr(),
                                                html.Div(
                                                    [
                                                        html.H6("Progress:"),
                                                        dbc.Progress(
                                                            id="progress-bar",
                                                            value=0,
                                                            max=4,
                                                            className="mb-3",
                                                            striped=True,
                                                            animated=True,
                                                        ),
                                                    ],
                                                    id="progress-section",
                                                    style={"display": "none"},
                                                ),
                                                html.H6(id="summary-title"),
                                                html.Div(id="selections-summary"),
                                                html.Hr(),
                                                dbc.Button(
                                                    "💾 Save Configuration",
                                                    id="save-config-btn",
                                                    color="success",
                                                    disabled=True,
                                                    className="w-100 mb-2",
                                                ),
                                                dbc.Alert(
                                                    id="save-status",
                                                    dismissable=True,
                                                    style={"display": "none"},
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=4,
                        ),
                    ]
                ),
                # Hidden components
                dcc.Store(id="app-state"),
                dcc.Store(id="current-selection"),
                dcc.Store(id="selection-mode"),
                dcc.Store(id="frame-data"),
            ],
            fluid=True,
        )

    def get_video_options(self):
        """Get available video options from dataset directory."""
        video_files = self.get_dataset_videos()
        options = []

        for video_path in video_files:
            # Convert to absolute path first, then get relative path from current working directory
            abs_path = video_path.resolve()
            try:
                # Try to get relative path from current working directory
                rel_path = abs_path.relative_to(Path.cwd())
                path_str = str(rel_path)
            except ValueError:
                # If relative path fails, use the absolute path
                path_str = str(abs_path)

            # Create a clean label showing the filename and parent directory
            parent_name = (
                video_path.parent.name if video_path.parent.name != "dataset" else "dataset"
            )
            label = f"{video_path.name}"
            if parent_name != "dataset":
                label += f" ({parent_name})"

            options.append({"label": label, "value": path_str})

        return options

    def create_patch_controls(self, current_patch_key, total_patches, completed_patches):
        """Create controls for the current patch selection."""
        if current_patch_key not in LEGO_COLOR_INFO:
            return html.Div("All patches completed! 🎉")

        color_info = LEGO_COLOR_INFO[current_patch_key]
        step_number = completed_patches + 1

        return html.Div(
            [
                dbc.Card(
                    [
                        dbc.CardHeader(
                            [
                                html.H6(
                                    f"Step {step_number}/{total_patches}: Select {color_info['name']}",
                                    className="mb-0",
                                )
                            ]
                        ),
                        dbc.CardBody(
                            [
                                # Color preview
                                html.Div(
                                    [
                                        html.Div(
                                            style={
                                                "width": "40px",
                                                "height": "40px",
                                                "backgroundColor": color_info["hex"],
                                                "border": "3px solid #333",
                                                "borderRadius": "8px",
                                                "display": "inline-block",
                                                "marginRight": "15px",
                                            }
                                        ),
                                        html.Div(
                                            [
                                                html.Strong(color_info["name"]),
                                                html.Br(),
                                                html.Small(
                                                    f"RGB: {color_info['rgb']}",
                                                    className="text-muted",
                                                ),
                                            ],
                                            style={
                                                "display": "inline-block",
                                                "verticalAlign": "top",
                                            },
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                html.P(
                                    color_info["description"],
                                    className="text-sm text-muted mb-3",
                                ),
                                dbc.Alert(
                                    [
                                        "🖱️ Use the rectangle tool ⬜ to draw over the ",
                                        html.Strong(color_info["name"]),
                                        " LEGO patch, then click 'Confirm Selection'.",
                                    ],
                                    color="info",
                                    className="mb-0",
                                ),
                            ]
                        ),
                    ],
                    color="primary",
                    outline=True,
                )
            ]
        )

    def create_roi_controls(self):
        """Create controls for ROI selection."""
        return html.Div(
            [
                dbc.Card(
                    [
                        dbc.CardHeader([html.H6("Select Region of Interest", className="mb-0")]),
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            style={
                                                "width": "40px",
                                                "height": "40px",
                                                "backgroundColor": "#17a2b8",
                                                "border": "3px solid #333",
                                                "borderRadius": "8px",
                                                "display": "inline-block",
                                                "marginRight": "15px",
                                            }
                                        ),
                                        html.Div(
                                            [
                                                html.Strong("Region of Interest"),
                                                html.Br(),
                                                html.Small(
                                                    "Area for mask generation",
                                                    className="text-muted",
                                                ),
                                            ],
                                            style={
                                                "display": "inline-block",
                                                "verticalAlign": "top",
                                            },
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                html.P(
                                    "Draw the region where the electrochromic material changes occur. "
                                    "You can use rectangles ⬜ or freehand paths ✏️ to define the area.",
                                    className="text-sm text-muted mb-3",
                                ),
                                dbc.Alert(
                                    [
                                        "🖱️ Use drawing tools to define the ROI area:",
                                        html.Br(),
                                        "• ⬜ Rectangle tool for simple rectangular areas",
                                        html.Br(),
                                        "• ✏️ Closed path tool for custom shapes",
                                        html.Br(),
                                        "• 🗑️ Eraser tool to remove annotations",
                                        html.Br(),
                                        "Click 'Generate ROI Mask' when finished.",
                                    ],
                                    color="info",
                                    className="mb-0",
                                ),
                            ]
                        ),
                    ],
                    color="info",
                    outline=True,
                )
            ]
        )

    def load_first_frame(self, video_path):
        """Load and return the first frame of the video."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()

            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
            else:
                return None
        except Exception as e:
            print(f"Error loading frame: {e}")
            return None

    def generate_mask_from_annotations(self, figure_data, frame_shape, output_path):
        """Generate binary mask from Plotly annotations and save as Mask.png
        This mask should only include ROI annotations, NOT LEGO patches."""
        try:
            # Create empty mask
            mask = np.zeros(frame_shape[:2], dtype=np.uint8)

            if "layout" in figure_data and "shapes" in figure_data["layout"]:
                shapes = figure_data["layout"]["shapes"]

                # Filter out LEGO patch shapes - only process ROI shapes
                roi_shapes = []
                for shape in shapes:
                    # Skip shapes that are LEGO patches (they have specific properties or we can identify them)
                    # We'll only process shapes that are meant for ROI
                    if self._is_roi_shape(shape):
                        roi_shapes.append(shape)

                for shape in roi_shapes:
                    if shape["type"] == "rect":
                        # Handle rectangle
                        x0, y0 = int(shape["x0"]), int(shape["y0"])
                        x1, y1 = int(shape["x1"]), int(shape["y1"])
                        x0, x1 = min(x0, x1), max(x0, x1)
                        y0, y1 = min(y0, y1), max(y0, y1)

                        # Ensure coordinates are within frame bounds
                        x0 = max(0, min(x0, frame_shape[1] - 1))
                        x1 = max(0, min(x1, frame_shape[1] - 1))
                        y0 = max(0, min(y0, frame_shape[0] - 1))
                        y1 = max(0, min(y1, frame_shape[0] - 1))

                        mask[y0 : y1 + 1, x0 : x1 + 1] = 255

                    elif shape["type"] == "path":
                        # Handle custom path
                        path_string = shape["path"]
                        # Parse SVG path commands to get coordinates
                        coords = self._parse_svg_path(path_string)
                        if len(coords) > 2:
                            # Create polygon from coordinates
                            coords_array = np.array(coords)
                            x_coords = np.clip(
                                coords_array[:, 0].astype(int), 0, frame_shape[1] - 1
                            )
                            y_coords = np.clip(
                                coords_array[:, 1].astype(int), 0, frame_shape[0] - 1
                            )

                            # Fill polygon
                            rr, cc = draw.polygon(y_coords, x_coords, frame_shape[:2])
                            mask[rr, cc] = 255

            # Apply morphological operations to clean up the mask
            if np.any(mask > 0):
                kernel = skimage.morphology.disk(3)
                mask = skimage.morphology.binary_closing(mask > 128, kernel)
                mask = skimage.morphology.binary_opening(mask, kernel)
                mask = skimage.morphology.remove_small_objects(mask, min_size=100)
                mask = (mask * 255).astype(np.uint8)

            # Save mask
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            mask_file = output_path / "Mask.png"
            skimage.io.imsave(mask_file, mask)

            # Calculate mask statistics for feedback
            total_pixels = frame_shape[0] * frame_shape[1]
            mask_pixels = np.sum(mask > 0)
            coverage_percent = (mask_pixels / total_pixels) * 100

            return (
                True,
                f"ROI mask saved to {mask_file} (Coverage: {coverage_percent:.1f}% of frame)",
            )

        except Exception as e:
            return False, f"Error generating ROI mask: {str(e)}"

    def _is_roi_shape(self, shape):
        """Determine if a shape is for ROI (not a LEGO patch)
        This can be based on shape properties, timing, or context."""

        # Simple approach: shapes drawn in ROI tab are ROI shapes
        # More sophisticated approach would be to track shape IDs or properties

        # For now, we'll assume all shapes in the current context are ROI shapes
        # when this function is called from the ROI tab
        return True

    def save_experiment_config(self, current_state, output_dir):
        """Save experiment configuration to JSON file in the output directory"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare configuration data
            config = {
                "experiment_info": {
                    "video_path": current_state["video_path"],
                    "exp_name": current_state["exp_name"],
                    "frame_dtype": current_state["frame_dtype"],
                    "created_at": datetime.now().isoformat(),
                    "app_version": "lego_app_v3",
                },
                "lego_patches": {},
                "roi": {
                    "mask_generated": current_state.get("roi_mask_generated", False),
                    "mask_file": "Mask.png"
                    if current_state.get("roi_mask_generated", False)
                    else None,
                },
                "lego_color_references": LEGO_COLOR_INFO,
            }

            # Add patch data
            for patch_key in PATCH_ORDER:
                if patch_key in current_state.get("confirmed_patches", {}):
                    coords = current_state["confirmed_patches"][patch_key]
                    x0, y0, x1, y1 = coords
                    config["lego_patches"][patch_key] = {
                        "coordinates": [[x0, y0], [x1, y1]],
                        "color_info": LEGO_COLOR_INFO[patch_key],
                        "confirmed": True,
                    }
                else:
                    config["lego_patches"][patch_key] = {
                        "coordinates": None,
                        "color_info": LEGO_COLOR_INFO[patch_key],
                        "confirmed": False,
                    }

            # Save config to JSON
            config_file = output_dir / "config.json"
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            return True, f"Configuration saved to {config_file}"

        except Exception as e:
            return False, f"Error saving configuration: {str(e)}"

    def _parse_svg_path(self, path_string):
        """Parse SVG path string to extract coordinates"""
        import re

        coords = []

        # Remove path commands and extract numbers
        numbers = re.findall(r"-?\d+\.?\d*", path_string)

        # Group numbers in pairs (x, y)
        for i in range(0, len(numbers) - 1, 2):
            try:
                x, y = float(numbers[i]), float(numbers[i + 1])
                coords.append([x, y])
            except (ValueError, IndexError):
                continue

        return coords

    def setup_callbacks(self):
        """Setup all Dash callbacks."""

        @self.app.callback(
            [
                Output("load-frame-btn", "disabled"),
                Output("config-preview", "children"),
                Output("app-state", "data", allow_duplicate=True),  # Añadir para resetear estado
            ],
            [
                Input("video-dropdown", "value"),
                Input("exp-name-input", "value"),
                Input("frame-dtype-dropdown", "value"),
            ],
            State("app-state", "data"),
            prevent_initial_call=True,
        )
        def update_load_button_and_preview(
            selected_video_path, exp_name, frame_dtype, current_state
        ):
            load_disabled = selected_video_path is None or not exp_name or not exp_name.strip()

            # Detectar si la configuración ha cambiado
            config_changed = False
            if current_state and current_state.get("frame_loaded"):
                # Verificar si algún parámetro de configuración cambió
                if (
                    current_state.get("video_path") != selected_video_path
                    or current_state.get("exp_name") != exp_name
                    or current_state.get("frame_dtype") != frame_dtype
                ):
                    config_changed = True

            # Si la configuración cambió, resetear el estado
            if config_changed:
                reset_state = {
                    "current_patch_index": 0,
                    "confirmed_patches": {},
                    "roi_annotations": [],
                    "roi_mask_generated": False,
                    "frame_loaded": False,  # Marcar como no cargado para forzar recarga
                    "video_path": None,
                    "exp_name": None,
                    "frame_dtype": None,
                }
            else:
                reset_state = current_state

            if selected_video_path and exp_name and exp_name.strip():
                preview = dbc.Alert(
                    [
                        html.H6("📋 Current Configuration:"),
                        html.Ul(
                            [
                                html.Li(f"📹 Video: {Path(selected_video_path).name}"),
                                html.Li(f"🏷️ Experiment: {exp_name}"),
                                html.Li(f"🎨 Color Space: {frame_dtype}"),
                            ]
                        ),
                        # Mostrar aviso si la configuración cambió
                        html.Hr(className="my-2") if config_changed else "",
                        dbc.Alert(
                            [
                                html.Strong("⚠️ Configuration changed! "),
                                "Previous selections have been reset. Click 'Load First Frame' to start with the new configuration.",
                            ],
                            color="warning",
                            className="mb-0 mt-2",
                        )
                        if config_changed
                        else "",
                    ],
                    color="light",
                    className="mb-0",
                )
            else:
                preview = html.P(
                    "Complete the configuration to see preview.", className="text-muted"
                )

            return load_disabled, preview, reset_state

        @self.app.callback(
            [
                Output("selection-mode", "data"),
                Output("frame-title", "children"),
                Output("control-panel-title", "children"),
                Output("current-selection-info", "children"),
                Output("current-selection-info", "style"),
                Output("tab-content", "children"),
                Output("progress-section", "style"),
                Output("summary-title", "children"),
                Output("confirm-selection-btn", "children"),
                Output("generate-mask-btn", "style"),
            ],
            Input("selection-tabs", "value"),
            State("app-state", "data"),
        )
        def update_selection_mode(tab_value, current_state):
            if tab_value == "patches-tab":
                mode = SELECTION_MODES["PATCHES"]
                frame_title = "🖼️ First Frame - LEGO Patch Selection"
                panel_title = "🧱 LEGO Patch Selection"

                if current_state and current_state.get("frame_loaded"):
                    completed_count = len(current_state.get("confirmed_patches", {}))
                    if completed_count < len(PATCH_ORDER):
                        current_patch_key = PATCH_ORDER[completed_count]
                        info = html.Div(
                            [
                                html.Strong(f"Step {completed_count + 1} of {len(PATCH_ORDER)}: "),
                                f"Select the {LEGO_COLOR_INFO[current_patch_key]['name']} patch",
                            ]
                        )
                        info_style = {"display": "block"}
                        controls = self.create_patch_controls(
                            current_patch_key, len(PATCH_ORDER), completed_count
                        )
                    else:
                        info = html.Div(
                            [
                                html.Strong("🎉 All patches selected! "),
                                "Switch to ROI tab or save configuration.",
                            ]
                        )
                        info_style = {"display": "block"}
                        controls = dbc.Alert("All LEGO patches completed! ✅", color="success")
                else:
                    info = ""
                    info_style = {"display": "none"}
                    controls = html.P("Load a video frame first.", className="text-muted")

                progress_style = {"display": "block"}
                summary_title = "Selected Patches:"
                button_text = "Confirm Selection"
                mask_btn_style = {"display": "none"}

            else:  # roi-tab
                mode = SELECTION_MODES["ROI"]
                frame_title = "🖼️ First Frame - Region of Interest Selection"
                panel_title = "🎯 Region of Interest Selection"

                if current_state and current_state.get("frame_loaded"):
                    if current_state.get("roi_mask_generated"):
                        info = html.Div(
                            [
                                html.Strong("✅ ROI Mask Generated! "),
                                "Binary mask has been created and saved. You can modify the region or save configuration.",
                            ]
                        )
                        info_style = {"display": "block"}
                    else:
                        info = html.Div(
                            [
                                html.Strong("Draw Region of Interest: "),
                                "Use the drawing tools to define the area for mask generation.",
                            ]
                        )
                        info_style = {"display": "block"}

                    controls = self.create_roi_controls()
                else:
                    info = ""
                    info_style = {"display": "none"}
                    controls = html.P("Load a video frame first.", className="text-muted")

                progress_style = {"display": "none"}
                summary_title = "ROI Selection:"
                button_text = "Confirm ROI Selection"
                mask_btn_style = {"display": "inline-block"}

            return (
                mode,
                frame_title,
                panel_title,
                info,
                info_style,
                controls,
                progress_style,
                summary_title,
                button_text,
                mask_btn_style,
            )

        @self.app.callback(
            [
                Output("frame-display", "figure"),
                Output("app-state", "data"),
                Output("frame-data", "data"),
                Output("reset-btn", "disabled"),
                Output("selections-summary", "children"),
                Output("progress-bar", "value"),
                Output("clear-annotations-btn", "disabled"),
            ],
            [
                Input("load-frame-btn", "n_clicks"),
                Input("reset-btn", "n_clicks"),
                Input("confirm-selection-btn", "n_clicks"),
                Input("selection-mode", "data"),
            ],
            [
                State("video-dropdown", "value"),
                State("exp-name-input", "value"),
                State("frame-dtype-dropdown", "value"),
                State("app-state", "data"),
                State("frame-display", "figure"),
                State("frame-display", "relayoutData"),
                State("selection-tabs", "value"),
            ],
        )
        def update_frame_display(
            load_clicks,
            reset_clicks,
            confirm_clicks,
            selection_mode,
            selected_video_path,
            exp_name,
            frame_dtype,
            current_state,
            current_figure,
            relayout_data,
            active_tab,
        ):
            ctx = dash.callback_context

            # Initialize state if None
            if current_state is None:
                current_state = {
                    "current_patch_index": 0,
                    "confirmed_patches": {},
                    "roi_annotations": [],
                    "roi_mask_generated": False,
                    "frame_loaded": False,
                    "video_path": None,
                    "exp_name": None,
                    "frame_dtype": None,
                }

            # Handle manual reset button
            if ctx.triggered and "reset-btn" in ctx.triggered[0]["prop_id"]:
                current_state["current_patch_index"] = 0
                current_state["confirmed_patches"] = {}
                current_state["roi_annotations"] = []
                current_state["roi_mask_generated"] = False

            # Handle confirm selection
            if (
                ctx.triggered
                and "confirm-selection-btn" in ctx.triggered[0]["prop_id"]
                and current_state.get("frame_loaded")
            ):
                if active_tab == "patches-tab":
                    # Extract patch coordinates from current figure annotations
                    shapes = []
                    if (
                        current_figure
                        and "layout" in current_figure
                        and "shapes" in current_figure["layout"]
                    ):
                        shapes = current_figure["layout"]["shapes"]
                    elif relayout_data and "shapes" in relayout_data:
                        shapes = relayout_data["shapes"]

                    if shapes:
                        # Get the last rectangle drawn (most recent patch)
                        # Filter only rectangles that are not already confirmed patches
                        existing_patch_rects = []
                        for patch_key, coords in current_state.get("confirmed_patches", {}).items():
                            existing_patch_rects.append(
                                {
                                    "x0": coords[0],
                                    "y0": coords[1],
                                    "x1": coords[2],
                                    "y1": coords[3],
                                }
                            )

                        for shape in reversed(shapes):
                            if shape["type"] == "rect":
                                # Check if this rectangle is not an existing patch
                                is_existing_patch = False
                                shape_coords = {
                                    "x0": int(shape["x0"]),
                                    "y0": int(shape["y0"]),
                                    "x1": int(shape["x1"]),
                                    "y1": int(shape["y1"]),
                                }

                                for existing_rect in existing_patch_rects:
                                    if (
                                        abs(shape_coords["x0"] - existing_rect["x0"]) < 5
                                        and abs(shape_coords["y0"] - existing_rect["y0"]) < 5
                                        and abs(shape_coords["x1"] - existing_rect["x1"]) < 5
                                        and abs(shape_coords["y1"] - existing_rect["y1"]) < 5
                                    ):
                                        is_existing_patch = True
                                        break

                                if not is_existing_patch:
                                    x0, y0 = int(shape["x0"]), int(shape["y0"])
                                    x1, y1 = int(shape["x1"]), int(shape["y1"])
                                    coords = [
                                        min(x0, x1),
                                        min(y0, y1),
                                        max(x0, x1),
                                        max(y0, y1),
                                    ]

                                    current_patch_key = PATCH_ORDER[
                                        current_state["current_patch_index"]
                                    ]
                                    current_state["confirmed_patches"][current_patch_key] = coords

                                    # Move to next patch
                                    if current_state["current_patch_index"] < len(PATCH_ORDER) - 1:
                                        current_state["current_patch_index"] += 1
                                    break

                elif active_tab == "roi-tab":
                    # Store ROI annotations for later mask generation
                    shapes = []
                    if (
                        current_figure
                        and "layout" in current_figure
                        and "shapes" in current_figure["layout"]
                    ):
                        shapes = current_figure["layout"]["shapes"]
                    elif relayout_data and "shapes" in relayout_data:
                        shapes = relayout_data["shapes"]

                    # Filter out LEGO patch shapes - only keep ROI shapes
                    roi_shapes = []
                    for shape in shapes:
                        # Skip shapes that match confirmed LEGO patches
                        is_lego_patch = False
                        if shape["type"] == "rect":
                            for patch_key, patch_coords in current_state.get(
                                "confirmed_patches", {}
                            ).items():
                                if (
                                    abs(int(shape["x0"]) - patch_coords[0]) < 5
                                    and abs(int(shape["y0"]) - patch_coords[1]) < 5
                                    and abs(int(shape["x1"]) - patch_coords[2]) < 5
                                    and abs(int(shape["y1"]) - patch_coords[3]) < 5
                                ):
                                    is_lego_patch = True
                                    break

                        if not is_lego_patch:
                            roi_shapes.append(shape)

                    current_state["roi_annotations"] = roi_shapes
                    current_state["roi_mask_generated"] = True

            # Verificar si la configuración está completa antes de procesar
            if (
                selected_video_path is None
                or load_clicks is None
                or not exp_name
                or not exp_name.strip()
            ):
                empty_fig = self._create_empty_figure(
                    "Complete configuration and click 'Load First Frame'"
                )
                return (
                    empty_fig,
                    current_state,
                    None,
                    True,
                    self._create_default_summary(),
                    0,
                    True,
                )

            # Verificar si necesitamos recargar el frame (configuración cambió o es primera vez)
            need_reload = (
                not current_state.get("frame_loaded")
                or current_state.get("video_path") != selected_video_path
                or current_state.get("exp_name") != exp_name
                or current_state.get("frame_dtype") != frame_dtype
            )

            if need_reload:
                # Load frame
                video_path = Path(selected_video_path)
                frame = self.load_first_frame(video_path)

                if frame is None:
                    error_fig = self._create_empty_figure(
                        f"Error loading video: {video_path.name}", "red"
                    )
                    return (
                        error_fig,
                        current_state,
                        None,
                        True,
                        html.P("Error loading video.", className="text-danger"),
                        0,
                        True,
                    )

                # Update state with new configuration (this will reset all selections)
                current_state = {
                    "current_patch_index": 0,
                    "confirmed_patches": {},
                    "roi_annotations": [],
                    "roi_mask_generated": False,
                    "frame_loaded": True,
                    "video_path": str(video_path),
                    "exp_name": exp_name,
                    "frame_dtype": frame_dtype,
                }

                # Store frame data for mask generation
                frame_data = {"shape": frame.shape, "video_path": str(video_path)}
            else:
                # Use existing frame
                video_path = Path(current_state["video_path"])
                frame = self.load_first_frame(video_path)
                frame_data = {"shape": frame.shape, "video_path": str(video_path)}

            # Create figure
            fig = px.imshow(frame, aspect="auto")
            fig.update_layout(
                title=f"First Frame: {video_path.name} | Experiment: {exp_name}",
                dragmode="drawrect" if active_tab == "patches-tab" else "drawclosedpath",
                margin={"l": 0, "r": 0, "t": 40, "b": 0},
            )

            # Add confirmed patches as permanent annotations (only show in patches tab)
            if active_tab == "patches-tab":
                for patch_key, coords in current_state.get("confirmed_patches", {}).items():
                    color_info = LEGO_COLOR_INFO[patch_key]
                    x0, y0, x1, y1 = coords

                    fig.add_shape(
                        type="rect",
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        line={"color": color_info["hex"], "width": 3},
                        fillcolor=color_info["hex"],
                        opacity=0.3,
                        layer="below",
                    )

                    # Add label
                    fig.add_annotation(
                        x=(x0 + x1) / 2,
                        y=y0 - 15,
                        text=f"✓ {color_info['name']}",
                        showarrow=False,
                        bgcolor=color_info["hex"],
                        bordercolor="white",
                        borderwidth=2,
                        font={
                            "color": "white" if patch_key in ["black", "dark_grey"] else "black",
                            "size": 12,
                        },
                    )

            # Add ROI annotations as permanent shapes (only show in ROI tab)
            elif active_tab == "roi-tab":
                for shape in current_state.get("roi_annotations", []):
                    if shape["type"] == "rect":
                        fig.add_shape(
                            type="rect",
                            x0=shape["x0"],
                            y0=shape["y0"],
                            x1=shape["x1"],
                            y1=shape["y1"],
                            line={"color": "#17a2b8", "width": 3},
                            fillcolor="#17a2b8",
                            opacity=0.2,
                            layer="below",
                        )
                    elif shape["type"] == "path":
                        fig.add_shape(
                            type="path",
                            path=shape["path"],
                            line={"color": "#17a2b8", "width": 3},
                            fillcolor="#17a2b8",
                            opacity=0.2,
                            layer="below",
                        )

            # Create summary
            summary = self._create_complete_summary(current_state)
            completed_patches = len(current_state["confirmed_patches"])

            return (
                fig,
                current_state,
                frame_data,
                False,
                summary,
                completed_patches,
                False,
            )

        @self.app.callback(
            [
                Output("selection-status", "children"),
                Output("confirm-selection-btn", "disabled"),
                Output("generate-mask-btn", "disabled"),
            ],
            [
                Input("frame-display", "figure"),
                Input("frame-display", "relayoutData"),  # Add this to detect shape changes
                Input("selection-tabs", "value"),
            ],
            State("app-state", "data"),
        )
        def update_selection_status(figure, relayout_data, active_tab, current_state):
            if not current_state or not current_state.get("frame_loaded"):
                return "", True, True

            # Extract shapes from figure or relayoutData
            shapes = []
            if figure and "layout" in figure and "shapes" in figure["layout"]:
                shapes = figure["layout"]["shapes"]
            elif relayout_data and "shapes" in relayout_data:
                shapes = relayout_data["shapes"]

            if active_tab == "patches-tab":
                # Check if there are rectangle annotations
                rectangles = [s for s in shapes if s.get("type") == "rect"]

                if rectangles:
                    current_patch_index = current_state.get("current_patch_index", 0)
                    if current_patch_index < len(PATCH_ORDER):
                        current_patch_key = PATCH_ORDER[current_patch_index]
                        color_info = LEGO_COLOR_INFO[current_patch_key]

                        # Crear preview del color y información detallada
                        color_preview = html.Div(
                            [
                                html.Div(
                                    style={
                                        "width": "30px",
                                        "height": "30px",
                                        "backgroundColor": color_info["hex"],
                                        "border": "2px solid #333",
                                        "borderRadius": "5px",
                                        "display": "inline-block",
                                        "marginRight": "12px",
                                        "verticalAlign": "middle",
                                    }
                                ),
                                html.Div(
                                    [
                                        html.Strong(
                                            f"{color_info['name']} LEGO Patch",
                                            style={"fontSize": "14px"},
                                        ),
                                        html.Br(),
                                        html.Small(
                                            f"RGB: {color_info['rgb']} | {color_info['description']}",
                                            className="text-muted",
                                            style={"fontSize": "11px"},
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "verticalAlign": "middle",
                                        "lineHeight": "1.2",
                                    },
                                ),
                            ],
                            style={"marginBottom": "8px"},
                        )

                        status = dbc.Alert(
                            [
                                color_preview,
                                html.Hr(style={"margin": "8px 0"}),
                                html.Strong("✏️ Rectangle annotation detected for "),
                                html.Strong(color_info["name"], style={"color": color_info["hex"]}),
                                html.Strong(" patch"),
                                html.Br(),
                                html.Small(
                                    "Click 'Confirm Selection' to save this patch and proceed to the next one.",
                                    className="text-muted",
                                ),
                            ],
                            color="warning",
                            className="mb-0",
                            style={"border-left": f"4px solid {color_info['hex']}"},
                        )
                        return status, False, True

                return "", True, True

            else:  # roi-tab
                # Check if there are any annotations (rectangles or paths)
                if shapes:
                    status = dbc.Alert(
                        [
                            html.Strong("🎯 ROI annotations detected: "),
                            f"{len(shapes)} annotation(s) drawn. ",
                            html.Br(),
                            html.Small(
                                "Click 'Generate ROI Mask' to create the binary mask for electrochromic analysis.",
                                className="text-muted",
                            ),
                        ],
                        color="info",
                        className="mb-0",
                    )
                    return status, True, False

                return "", True, True

        @self.app.callback(
            Output("frame-display", "figure", allow_duplicate=True),
            Input("clear-annotations-btn", "n_clicks"),
            [State("frame-display", "figure"), State("frame-data", "data")],
            prevent_initial_call=True,
        )
        def clear_annotations(n_clicks, current_figure, frame_data):
            if n_clicks and current_figure:
                # Remove all shapes from the figure
                new_figure = current_figure.copy()
                if "layout" in new_figure:
                    new_figure["layout"]["shapes"] = []
                return new_figure
            return dash.no_update

        @self.app.callback(
            [
                Output("save-status", "children"),
                Output("save-status", "style"),
                Output("save-status", "color"),
                Output("app-state", "data", allow_duplicate=True),
            ],
            [
                Input("save-config-btn", "n_clicks"),
                Input("generate-mask-btn", "n_clicks"),
            ],
            [
                State("app-state", "data"),
                State("frame-display", "figure"),
                State("frame-display", "relayoutData"),
                State("frame-data", "data"),
                State("selection-tabs", "value"),  # Add current tab context
            ],
            prevent_initial_call=True,
        )
        def handle_save_and_mask_generation(
            save_clicks,
            mask_clicks,
            current_state,
            figure,
            relayout_data,
            frame_data,
            active_tab,
        ):
            ctx = dash.callback_context
            triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if triggered_id == "generate-mask-btn" and mask_clicks:
                # Generate mask from ROI annotations only
                if not current_state or not frame_data:
                    return (
                        "❌ Missing data for mask generation.",
                        {"display": "block"},
                        "danger",
                        current_state,
                    )

                # Use stored ROI annotations or get current shapes if in ROI tab
                roi_shapes = current_state.get("roi_annotations", [])

                # If no stored ROI annotations, get current shapes from ROI tab
                if not roi_shapes and active_tab == "roi-tab":
                    shapes = []
                    if relayout_data and "shapes" in relayout_data:
                        shapes = relayout_data["shapes"]
                    elif figure and "layout" in figure and "shapes" in figure["layout"]:
                        shapes = figure["layout"]["shapes"]

                    # Filter out LEGO patch shapes
                    for shape in shapes:
                        is_lego_patch = False
                        if shape["type"] == "rect":
                            for patch_key, patch_coords in current_state.get(
                                "confirmed_patches", {}
                            ).items():
                                if (
                                    abs(int(shape["x0"]) - patch_coords[0]) < 5
                                    and abs(int(shape["y0"]) - patch_coords[1]) < 5
                                    and abs(int(shape["x1"]) - patch_coords[2]) < 5
                                    and abs(int(shape["y1"]) - patch_coords[3]) < 5
                                ):
                                    is_lego_patch = True
                                    break

                        if not is_lego_patch:
                            roi_shapes.append(shape)

                if not roi_shapes:
                    return (
                        "❌ No ROI annotations found. Please draw ROI regions first.",
                        {"display": "block"},
                        "warning",
                        current_state,
                    )

                try:
                    # Create output directory: output/video_name/experiment_name/
                    video_path = Path(current_state["video_path"])
                    exp_name = current_state["exp_name"]
                    output_dir = Path("output") / video_path.stem / exp_name

                    # Create figure data with only ROI shapes for mask generation
                    figure_data = {"layout": {"shapes": roi_shapes}}

                    success, message = self.generate_mask_from_annotations(
                        figure_data, frame_data["shape"], output_dir
                    )

                    if success:
                        # Update state to mark mask as generated
                        updated_state = current_state.copy()
                        updated_state["roi_mask_generated"] = True
                        updated_state["roi_annotations"] = roi_shapes  # Store the ROI shapes

                        # Also save configuration to JSON in the same directory
                        config_success, config_message = self.save_experiment_config(
                            updated_state, output_dir
                        )

                        full_message = f"✅ {message}\n"
                        if config_success:
                            full_message += f"✅ {config_message}\n"
                        else:
                            full_message += f"⚠️ {config_message}\n"

                        full_message += f"📁 Files saved to: {output_dir}\n"
                        full_message += f"🎯 ROI regions processed: {len(roi_shapes)} annotation(s)"

                        return (
                            full_message,
                            {"display": "block"},
                            "success",
                            updated_state,
                        )
                    else:
                        return message, {"display": "block"}, "danger", current_state

                except Exception as e:
                    return (
                        f"❌ Error generating ROI mask: {str(e)}",
                        {"display": "block"},
                        "danger",
                        current_state,
                    )

            elif triggered_id == "save-config-btn" and save_clicks:
                # Rest of save configuration code remains the same...
                # (Keep existing save configuration logic)

                if (
                    not current_state
                    or not current_state.get("video_path")
                    or len(current_state.get("confirmed_patches", {})) < 4
                ):
                    return (
                        "❌ Complete all patch selections first.",
                        {"display": "block"},
                        "warning",
                        current_state,
                    )

                try:
                    # Create output directory: output/video_name/experiment_name/
                    video_path = Path(current_state["video_path"])
                    exp_name = current_state["exp_name"]
                    output_dir = Path("output") / video_path.stem / exp_name

                    # Save configuration to JSON
                    config_success, config_message = self.save_experiment_config(
                        current_state, output_dir
                    )

                    if not config_success:
                        return (
                            f"❌ {config_message}",
                            {"display": "block"},
                            "danger",
                            current_state,
                        )

                    # Convert patch coordinates to the expected format for params.yaml
                    patches = []
                    for patch_key in PATCH_ORDER:
                        if patch_key in current_state["confirmed_patches"]:
                            coords = current_state["confirmed_patches"][patch_key]
                            x0, y0, x1, y1 = coords
                            patches.append([[x0, y0], [x1, y1]])

                    # For ROI, we'll store a flag indicating mask was generated
                    roi_data = {"mask_generated": current_state.get("roi_mask_generated", False)}

                    # Update or create video configuration in params.yaml
                    params = load_params()

                    # Create new video configuration
                    video_config = {
                        "video_path": current_state["video_path"],
                        "exp_name": current_state["exp_name"],
                        "frame_dtype": current_state["frame_dtype"],
                        "patches": patches,
                        "roi": roi_data,
                    }

                    # Add to videos list
                    if "videos" not in params:
                        params["videos"] = []

                    # Check if video already exists in config
                    video_exists = False
                    for i, existing_video in enumerate(params["videos"]):
                        if (
                            existing_video["video_path"] == current_state["video_path"]
                            and existing_video["exp_name"] == current_state["exp_name"]
                        ):
                            params["videos"][i] = video_config
                            video_exists = True
                            break

                    if not video_exists:
                        params["videos"].append(video_config)

                    # Save to params.yaml
                    with open("params.yaml", "w") as f:
                        yaml.dump(params, f, default_flow_style=False, sort_keys=False)

                    roi_info = ""
                    if current_state.get("roi_mask_generated"):
                        roi_count = len(current_state.get("roi_annotations", []))
                        roi_info = f"🎯 ROI Mask: Generated from {roi_count} annotation(s)"
                    else:
                        roi_info = "🎯 ROI Mask: Not generated"

                    return (
                        f"✅ Configuration saved successfully!\n"
                        f"📁 JSON Config: {output_dir / 'config.json'}\n"
                        f"📄 YAML Config: params.yaml updated\n"
                        f"📹 Video: {Path(current_state['video_path']).name}\n"
                        f"🏷️ Experiment: {current_state['exp_name']}\n"
                        f"🎨 Frame Type: {current_state['frame_dtype']}\n"
                        f"🧱 LEGO Patches: {len(current_state['confirmed_patches'])} selected\n"
                        f"{roi_info}",
                        {"display": "block"},
                        "success",
                        current_state,
                    )

                except Exception as e:
                    return (
                        f"❌ Error saving configuration: {str(e)}",
                        {"display": "block"},
                        "danger",
                        current_state,
                    )

            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        @self.app.callback(
            Output("save-config-btn", "disabled"),
            Input("progress-bar", "value"),
        )
        def update_save_button(progress):
            return progress < 4

    def _create_empty_figure(self, message, color="gray"):
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": message,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "xanchor": "center",
                    "yanchor": "middle",
                    "showarrow": False,
                    "font": {"size": 16, "color": color},
                }
            ],
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )
        return fig

    def _create_complete_summary(self, current_state):
        """Create a complete summary of all selections."""
        if not current_state:
            return self._create_default_summary()

        confirmed_patches = current_state.get("confirmed_patches", {})

        summary_items = []

        # LEGO Patches section
        summary_items.append(html.H6("🧱 LEGO Patches:", className="mt-2"))

        if confirmed_patches:
            patch_items = []
            for patch_key in PATCH_ORDER:
                if patch_key in confirmed_patches:
                    color_info = LEGO_COLOR_INFO[patch_key]
                    coords = confirmed_patches[patch_key]
                    patch_items.append(
                        dbc.ListGroupItem(
                            [
                                html.Div(
                                    [
                                        html.Span("✅", className="me-2 text-success"),
                                        html.Div(
                                            style={
                                                "width": "20px",
                                                "height": "20px",
                                                "backgroundColor": color_info["hex"],
                                                "border": "1px solid #333",
                                                "borderRadius": "3px",
                                                "display": "inline-block",
                                                "marginRight": "8px",
                                            }
                                        ),
                                        html.Span(
                                            [
                                                html.Strong(color_info["name"]),
                                                html.Br(),
                                                html.Small(
                                                    f"[{coords[0]}, {coords[1]}] → [{coords[2]}, {coords[3]}]",
                                                    className="text-muted",
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ]
                        )
                    )
                else:
                    color_info = LEGO_COLOR_INFO[patch_key]
                    patch_items.append(
                        dbc.ListGroupItem(
                            [
                                html.Div(
                                    [
                                        html.Span("⏳", className="me-2 text-warning"),
                                        html.Span(
                                            [
                                                html.Strong(
                                                    color_info["name"], className="text-muted"
                                                ),
                                                html.Br(),
                                                html.Small(
                                                    "Pending selection", className="text-muted"
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            color="light",
                        )
                    )

            summary_items.append(dbc.ListGroup(patch_items, className="mb-3"))
        else:
            summary_items.append(html.P("No patches selected yet.", className="text-muted mb-3"))

        # ROI section
        summary_items.append(html.H6("🎯 Region of Interest:", className="mt-2"))

        roi_mask_generated = current_state.get("roi_mask_generated", False)
        if roi_mask_generated:
            roi_item = dbc.ListGroupItem(
                [
                    html.Div(
                        [
                            html.Span("✅", className="me-2 text-success"),
                            html.Div(
                                style={
                                    "width": "20px",
                                    "height": "20px",
                                    "backgroundColor": "#17a2b8",
                                    "border": "1px solid #333",
                                    "borderRadius": "3px",
                                    "display": "inline-block",
                                    "marginRight": "8px",
                                }
                            ),
                            html.Span(
                                [
                                    html.Strong("ROI Mask Generated"),
                                    html.Br(),
                                    html.Small(
                                        "Binary mask created from annotations",
                                        className="text-muted",
                                    ),
                                ]
                            ),
                        ]
                    )
                ]
            )
        else:
            roi_item = dbc.ListGroupItem(
                [
                    html.Div(
                        [
                            html.Span("⏳", className="me-2 text-warning"),
                            html.Span(
                                [
                                    html.Strong("ROI Mask", className="text-muted"),
                                    html.Br(),
                                    html.Small(
                                        "Draw annotations and generate mask",
                                        className="text-muted",
                                    ),
                                ]
                            ),
                        ]
                    )
                ],
                color="light",
            )

        summary_items.append(dbc.ListGroup([roi_item]))

        return html.Div(summary_items)

    def _create_default_summary(self):
        """Create default summary when no selections are made."""
        return html.P("No selections made yet.", className="text-muted")

    def run(self, debug=True, port=8050, host="127.0.0.1"):
        """Run the Dash application."""
        print(f"🚀 Starting LEGO Selector & ROI App at http://{host}:{port}")
        print("📝 Instructions:")
        print("   1. Select a video from the dataset dropdown")
        print("   2. Configure experiment parameters")
        print("   3. Click 'Load First Frame'")
        print("   4. Use tabs to switch between:")
        print("      🧱 LEGO Patches - Use rectangle tool to select patches")
        print("      🎯 ROI - Use drawing tools to define region, then generate mask")
        print("   5. Save configuration when all selections are complete")
        print(f"\n📁 Looking for videos in: {Path('dataset').absolute()}")

        self.app.run(debug=debug, port=port, host=host)


if __name__ == "__main__":
    app = LegoSelectorApp()
    app.run()
