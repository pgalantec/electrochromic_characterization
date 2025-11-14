from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skimage
import skimage.morphology
from scipy import ndimage as ndi
from tqdm import tqdm

from src.opt_gamma import apply_gamma_correction, extract_gamma
from src.plots import plot_analisys, plot_channel, save_channel
from src.utils.io import load_params


def _transform_frame(
    bgr_frame, videoitem, optgamma=2.4, apply_gamma=False, only_1_channel=False, channel=None
):
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb_frame_norm = (rgb_frame / 255).astype("float32")
    R, G, B = cv2.split(rgb_frame_norm)
    # Apply Gamma correction to the first frame
    if apply_gamma:
        R = apply_gamma_correction(R, optgamma)
        G = apply_gamma_correction(G, optgamma)
        B = apply_gamma_correction(B, optgamma)

    # Transform frame
    if not only_1_channel:
        rgb_frame_norm = cv2.merge([R, G, B])
        if videoitem["frame_dtype"] == "gray":
            outframe = skimage.color.rgb2gray(rgb_frame_norm)
        elif videoitem["frame_dtype"] == "Y":
            outframe = skimage.color.rgb2xyz(rgb_frame_norm)[:, :, 1]
        elif videoitem["frame_dtype"] == "L":
            outframe = cv2.cvtColor(rgb_frame_norm, cv2.COLOR_BGR2Lab)[:, :, 0]
        else:
            print("not supportted frame_dtype in params.yaml")
    else:
        if channel == "R":
            outframe = R
        if channel == "B":
            outframe = B
        if channel == "G":
            outframe = G

    return outframe.astype("float32")


def extract_key_img(optgamma, videoitem, vopath: Path):
    # Check if the mask has been previously created for that experiment.
    # If not call create_mask method

    try:
        mask = skimage.io.imread(Path(vopath) / "Mask.png")
        mask = mask > 128
    except:  # noqa: E722
        cap = cv2.VideoCapture(videoitem["video_path"])

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Procesar el primer frame para visualizar los parches
        ret, first_frame = cap.read()
        if ret:
            first_frame = _transform_frame(first_frame, videoitem, optgamma, apply_gamma=True)

        # Restart the video capture to process from the beginning.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        cont = 0
        mcontrast = []
        blobs = []

        # For each video frame extract difference between original and current frame
        with tqdm(total=frame_count, desc="Processing frames", unit="frame") as pbar:
            while cap.isOpened():
                # Leer el siguiente frame
                ret, frame = cap.read()
                if not ret:
                    break

                frame = _transform_frame(frame, videoitem, optgamma, apply_gamma=True)

                # Extract different between first and current frame
                blob = np.abs(np.subtract(first_frame, frame)).astype(np.float32)
                blobs.append(blob)
                mcontrast.append(np.std(frame))

                cont += 1
                if cv2.waitKey(1) == ord("s"):
                    break

                pbar.update(1)

        def actualizar_valor(x):
            return np.median(mcontrast) if x > 2 * np.mean(mcontrast) else x

        mcontrast_corregido = np.array(list(map(actualizar_valor, mcontrast)))
        selected_frame = np.argmax(mcontrast_corregido)

        plot_analisys(mcontrast, mcontrast_corregido, vopath)

        cap.release()
        cv2.destroyAllWindows()

        # Generate mask
        mask = create_mask(selected_frame, blobs, vopath)

    return mask


def gamma_analysis(videoitem, mask, vopath, name, optgamma=2.4):
    def _xlabel(fps, framecurr):
        time = framecurr / fps
        return time

    # Use original or calibrated video
    # if videoitem["calibration"]:
    #     cap = cv2.VideoCapture(str(vopath) + "/corr_video.mp4")
    # else:
    cap = cv2.VideoCapture(videoitem["video_path"])

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    blue_ori_m = []
    blue_ori_g = []
    green_ori_m = []
    green_ori_g = []
    red_ori_m = []
    red_ori_g = []
    Y_post_g = []
    Y_pre_g = []

    # For each video frame obtain red channel value and its value after gamma correction
    with tqdm(total=frame_count, desc="Processing frames", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mask_3d = np.stack([mask] * 3, axis=-1)
            frame = np.where(mask_3d, frame, 0)

            frame_xyz_pre = _transform_frame(frame, videoitem, optgamma, apply_gamma=False)
            frame_xyz_post = _transform_frame(frame, videoitem, optgamma, apply_gamma=True)

            frame_R_pre = _transform_frame(
                frame, videoitem, optgamma, apply_gamma=False, only_1_channel=True, channel="R"
            )
            frame_R_post = _transform_frame(
                frame, videoitem, optgamma, apply_gamma=True, only_1_channel=True, channel="R"
            )

            frame_B_pre = _transform_frame(
                frame, videoitem, optgamma, apply_gamma=False, only_1_channel=True, channel="B"
            )
            frame_B_post = _transform_frame(
                frame, videoitem, optgamma, apply_gamma=True, only_1_channel=True, channel="B"
            )

            frame_G_pre = _transform_frame(
                frame, videoitem, optgamma, apply_gamma=False, only_1_channel=True, channel="G"
            )
            frame_G_post = _transform_frame(
                frame, videoitem, optgamma, apply_gamma=True, only_1_channel=True, channel="G"
            )

            green_ori_m.append(
                np.mean(frame_G_pre[frame_G_pre != 0]) if np.any(frame_G_pre != 0) else 0
            )
            green_ori_g.append(
                np.mean(frame_G_post[frame_G_post != 0]) if np.any(frame_G_post != 0) else 0
            )
            blue_ori_m.append(
                np.mean(frame_B_pre[frame_B_pre != 0]) if np.any(frame_B_pre != 0) else 0
            )
            blue_ori_g.append(
                np.mean(frame_B_post[frame_B_post != 0]) if np.any(frame_B_post != 0) else 0
            )
            red_ori_m.append(
                np.mean(frame_R_pre[frame_R_pre != 0]) if np.any(frame_R_pre != 0) else 0
            )
            red_ori_g.append(
                np.mean(frame_R_post[frame_R_post != 0]) if np.any(frame_R_post != 0) else 0
            )
            Y_pre_g.append(
                np.mean(frame_xyz_pre[frame_xyz_pre != 0]) if np.any(frame_xyz_pre != 0) else 0
            )
            Y_post_g.append(
                np.mean(frame_xyz_post[frame_xyz_post != 0]) if np.any(frame_xyz_post != 0) else 0
            )

            pbar.update(1)

    # red_ori_m_detrend = detrend(red_ori_m)
    # red_ori_g_detrend = detrend(red_ori_g)
    # Y_pre_g_detrend = detrend(Y_pre_g)
    # Y_post_g_detrend = detrend(Y_post_g)

    for plot_color in [
        (red_ori_m, red_ori_g, "red"),
        (blue_ori_m, blue_ori_g, "blue"),
        (green_ori_m, green_ori_g, "green"),
        (Y_pre_g, Y_post_g, "Y"),
    ]:
        plot_channel(plot_color, optgamma, vopath)
        save_channel(plot_color, vopath)

    # # Plot red comparison:
    # plt.figure(figsize=(10, 5))
    # plt.plot(red_ori_m_detrend, label='Original Red', color='red', linestyle='-')
    # plt.plot(red_ori_g_detrend, label='Modified Red (Gamma)', color='orange', linestyle='-')

    # plt.title(f"Red vs Red after {optgamma} Gamma correction")
    # plt.xlabel("Frame")
    # plt.ylabel("R")
    # plt.legend()

    # # Save red values plot
    # plt.savefig(Path(vopath) / "red_channel_comparison_detrend.jpg")

    # Plot blue comparison:
    # plt.figure(figsize=(10, 5))
    # plt.plot(blue_ori_m, label='Original Blue', color='blue', linestyle='-')
    # plt.plot(blue_ori_g, label='Modified Blue (Gamma)', color='orange', linestyle='-')

    # plt.title(f"Blue vs Blue after {optgamma} Gamma correction")
    # plt.xlabel("Frame")
    # plt.ylabel("B")
    # plt.legend()

    # # Save red values plot
    # plt.savefig(Path(vopath) / "blue_channel_comparison.jpg")

    # # Plot red comparison:
    # plt.figure(figsize=(10, 5))
    # plt.plot(Y_pre_g, label='Original Y', color='red', linestyle='-')
    # plt.plot(Y_post_g, label='Modified Y', color='orange', linestyle='-')

    # plt.title(f"Y vs Y after {optgamma} gamma correction")
    # plt.xlabel("Frame")
    # plt.ylabel("Y")
    # plt.legend()

    # # Save red values plot
    # plt.savefig(Path(vopath) / "Y_comparison.jpg")

    # # Plot red comparison:
    # plt.figure(figsize=(10, 5))
    # plt.plot(Y_pre_g_detrend, label='Original Y', color='red', linestyle='-')
    # plt.plot(Y_post_g_detrend, label='Modified Y', color='orange', linestyle='-')

    # plt.title(f"Y vs Y after {optgamma} gamma correction")
    # plt.xlabel("Frame")
    # plt.ylabel("Y")
    # plt.legend()

    # # Save red values plot
    # plt.savefig(Path(vopath) / "Y_comparison_detrend.jpg")

    # # Save red channel values in txt file
    # with open(Path(vopath) / "red_channel_values_detrend.txt", "w") as f:
    #     f.write("OriginalRed\tGammaRed\n")
    #     for original, gamma_corrected in zip(red_ori_m_detrend, red_ori_g_detrend):
    #         f.write(f"{original}\t{gamma_corrected}\n")

    # # Save blue channel values in txt file
    # with open(Path(vopath) / "blue_channel_values.txt", "w") as f:
    #     f.write("OriginalBlue\tGammaBlue\n")
    #     for original, gamma_corrected in zip(blue_ori_m, blue_ori_g):
    #         f.write(f"{original}\t{gamma_corrected}\n")

    # # Save green channel values in txt file
    # with open(Path(vopath) / "green_channel_values.txt", "w") as f:
    #     f.write("OriginalGreen\tGammaGreen\n")
    #     for original, gamma_corrected in zip(green_ori_m, green_ori_g):
    #         f.write(f"{original}\t{gamma_corrected}\n")

    # time = [_xlabel(frame_count, i) for i in range(len(Y_post_g))]

    # with open(Path(vopath) / f"curve_{name}.txt", "w") as file:
    #     file.write("Time(s); Reflectance \n")
    #     for tiempo, intensidad, int_detrend in zip(time, Y_post_g, Y_post_g_detrend):
    #         file.write(f"{tiempo}; {intensidad}; {int_detrend}\n")


def create_mask(frame, blobs, mask_path):
    # Use desired frame to create the mask
    blobframe = blobs[frame]
    thotsu = skimage.filters.threshold_otsu(blobframe)
    maskori = blobframe > thotsu
    maskori = skimage.morphology.remove_small_objects(maskori, 10)

    # Create histogram
    histograma, bins = skimage.exposure.histogram(blobframe * 255)
    plt.figure(figsize=(10, 5))
    sns.histplot(histograma, bins=bins, kde=False, color="blue")
    plt.axvline(
        x=thotsu * 255, color="red", linestyle="--", linewidth=1
    )  # Línea vertical en el umbral
    plt.title("Blob Histogram %d" % thotsu)  # noqa: UP031
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.grid(True)

    # Save hist
    plt.savefig(mask_path / "hist.jpg")
    plt.close()

    kernel = skimage.morphology.disk(5)
    # Apply morphological operations
    mask = skimage.morphology.binary_closing(maskori, kernel)
    mask = skimage.morphology.binary_opening(mask, kernel)

    mask = skimage.segmentation.clear_border(mask)
    mask = skimage.morphology.binary_erosion(mask, kernel)
    mask_filled = ndi.binary_fill_holes(mask)
    mask_obj = skimage.morphology.remove_small_objects(mask_filled, 5000)

    # Save binary mask
    skimage.io.imsave(mask_path / "Mask.png", skimage.img_as_ubyte(mask_obj))

    return mask_obj


def main(videos):
    for i, videoitem in enumerate(videos):
        # Init experiment
        video_path = Path(videoitem["video_path"])
        expname = videoitem["exp_name"]
        name = video_path.stem
        vopath = Path("output") / name / expname
        vopath.mkdir(parents=True, exist_ok=True)
        print(f"1 - Working with video {name}, Experiment {expname}")

        print("2 - Extracting Optimal Gamma")
        optgamma = extract_gamma(videoitem, vopath)

        # Extract video information (Blob and Mask)
        print("3 - Analysing video and creating mask")
        mask = extract_key_img(optgamma, videoitem, vopath)

        # Gamma correction
        print("4 - Applying Gamma correction")
        gamma_analysis(videoitem, mask, vopath, name, optgamma)

        print("Done!")


if __name__ == "__main__":
    params = load_params()
    videos = params["videos"]
    main(videos=videos)
