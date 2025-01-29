from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from src.utils.io import load_params

def _patches2mask(frame_width, frame_height, patches):
    rmasks = []
    # Para cada parche, establecer la región correspondiente en la máscara a 1
    for tl, br in patches:
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        mask[tl[1] : br[1], tl[0] : br[0]] = 1
        rmasks.append(mask)

    return rmasks

def _get_average_Y(image_xyz, mask):
    mask = mask.astype(bool)
    region_L = image_xyz[:, :, 1]
    avg_L = np.median(region_L[mask])
    return avg_L

def apply_gamma_correction(H, gamma=2.4):  # noqa: N803
    return np.where(H < 0.04045, H / 12.92, ((H + 0.055) / 1.055) ** gamma)

def _draw_patch_rectangles(image, patches):
    for top_left, bottom_right in patches:
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Dibuja un rectángulo verde
    return image

def adjust_gamma_for_perfect_r_value(X, framenorm, maskslego, vopath):
    best_gamma = None
    best_r_value = -np.inf
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    gamma_values = []
    r_value_values = []
    # Probar diferentes valores de Gamma
    for gamma in np.linspace(1, 8, 71):  # Ajustar el rango y la resolución según sea necesario
        gamma = round(gamma, 2)
        framelin = apply_gamma_correction(framenorm, gamma)
        frame_xyz = np.dot(framelin, matrix.T)
        current_Y_lego = [_get_average_Y(frame_xyz, maskl) for maskl in maskslego]

        # Calcular la regresión lineal
        slope, intercept, r_value, _, _ = linregress(X, current_Y_lego)

        # Verificar si r_value es lo más cercano a 1
        if abs(r_value - 1) < abs(best_r_value - 1):
            best_r_value = r_value
            best_gamma = gamma

        gamma_values.append(gamma)
        r_value_values.append(r_value)

    # Optimal lineregress
    framelin = apply_gamma_correction(framenorm, best_gamma)
    frame_xyz = np.dot(framelin, matrix.T)
    current_Y_lego = [_get_average_Y(frame_xyz, maskl) for maskl in maskslego]

    # Calculate linear regression
    slope, intercept, r_value, _, _ = linregress(X, current_Y_lego)

    plt.scatter(X, current_Y_lego, color="red", label="Y lin values")
    # Add coordinates to scatter plot points
    for x, y in zip(X, current_Y_lego, strict=False):
        plt.text(
            round(x, 2), round(y, 2), f"({round(x, 2)}, {round(y, 2)})", fontsize=9, ha="right"
        )

    x_point = np.linspace(min(X), max(X), 100)
    y_point = slope * x_point + intercept

    with open(Path(vopath) / "gamma_lin_rect.txt", "w") as f:
        f.write("x_point y_point\n")
        for a, b in zip(x_point, y_point, strict=False):
            f.write(f"{a} {b}\n")

    plt.plot(
        x_point,
        y_point,
        color="red",
        label=f"lin: y={slope:.2f}x + {intercept:.2f}",
        linestyle="--",
    )

    with open(Path(vopath) / "gamma_lin_final_data.txt", "w") as f:
        f.write("X current_Y_lego\n")
        for a, b in zip(X, current_Y_lego, strict=False):
            f.write(f"{a} {b}\n")

    return best_gamma, best_r_value, gamma_values, r_value_values


def extract_gamma(videoitem, vopath):
    # Load experimento params
    params = load_params()

    # Extract LEGA sRGB values and transform to L
    nominal_srgb = [
        params["lego_srgb"]["white"],
        params["lego_srgb"]["clear_grey"],
        params["lego_srgb"]["dark_grey"],
        params["lego_srgb"]["black"],
    ]
    nominal_srgb_norm = np.array(nominal_srgb) / 255
    nominal_srgb_lin = np.where(
        nominal_srgb_norm <= 0.04045,
        nominal_srgb_norm / 12.92,
        ((nominal_srgb_norm + 0.055) / 1.055) ** 2.4,
    )

    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )

    xyz = np.dot(nominal_srgb_lin, matrix.T)
    nominal_Y_lego = xyz[:, 1]

    # Initialize video
    cap = cv2.VideoCapture(videoitem["video_path"])

    # Video features
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Extract patches masks
    maskslego = _patches2mask(frame_width, frame_height, videoitem["patches"])
    combinedlego = np.zeros_like(maskslego[0], dtype=np.uint8)
    for maskl in maskslego:
        combinedlego = cv2.bitwise_or(combinedlego, maskl)

    # read first video frame
    ret, frame = cap.read()
    if ret:
        first_frame_with_rectangles = _draw_patch_rectangles(frame.copy(), videoitem["patches"])
        cv2.imwrite(Path(vopath) / "patches.png", first_frame_with_rectangles)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        framenorm = frame / 255.0
        frame_xyz_cur = np.dot(framenorm, matrix.T)
        current_Y_lego = [_get_average_Y(frame_xyz_cur, maskl) for maskl in maskslego]

        # Regression plot
        plt.figure(figsize=(8, 6))
        plt.scatter(nominal_Y_lego, current_Y_lego, color="blue", label="Y values")

        with open(Path(vopath) / "gamma_lin_initial_data.txt", "w") as f:
            f.write("Y_nominal Y_current\n")
            for a, b in zip(nominal_Y_lego, current_Y_lego, strict=False):
                f.write(f"{a} {b}\n")

        # Add coordinates to scatter plot
        for x, y in zip(nominal_Y_lego, current_Y_lego, strict=False):
            plt.text(
                round(x, 2), round(y, 2), f"({round(x, 2)}, {round(y, 2)})", fontsize=9, ha="right"
            )

        # Calculate optimal Gamma
        optimal_gamma, optimal_r_value, gamma_values, r_value_values = (
            adjust_gamma_for_perfect_r_value(nominal_Y_lego, framenorm, maskslego, vopath)
        )
        print(f"Optimal Gamma: {optimal_gamma}, r_value: {optimal_r_value}")

        plt.xlabel("Frame Y value")
        plt.ylabel("Target Y value")
        plt.legend()
        plt.title("Gamma linearization")
        plt.savefig(Path(vopath) / "gamma_linearization.jpg")

        plt.figure("Gamma vs r_value")
        plt.plot(gamma_values, r_value_values, linestyle="-", color="b", label="r_value vs gamma")
        plt.xlabel("Gamma")
        plt.ylabel("r_value")
        plt.title("Evolución de r_value en función de Gamma")
        plt.savefig(Path(vopath) / "r2_optimization.jpg")
        plt.close()

        with open(Path(vopath) / "plot_data.txt", "w") as f:
            f.write("gamma_values r_value_values\n")
            for gamma_int, r_value_int in zip(gamma_values, r_value_values, strict=False):
                f.write(f"{gamma_int} {r_value_int}\n")

    return optimal_gamma.astype("float32")


def main(videos):
    for i, videoitem in enumerate(videos):
        # Initialize experiment calibration
        video_path = Path(videoitem["video_path"])
        name = video_path.stem
        vopath = Path("output") / name
        vopath.mkdir(parents=True, exist_ok=True)

        # Color calibration
        extract_gamma(videoitem, vopath)


if __name__ == "__main__":
    params = load_params()
    videos = params["videos"]
    main(videos)
