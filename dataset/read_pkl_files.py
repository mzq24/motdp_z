import os
import pickle
import numpy as np
import cv2

OUT_DIR = "/media/z/data/dataset/carla/mini_data"

def inspect_pkl(pkl_path, show_img=False, max_frames=3):
    print(f"\n=== Inspect {os.path.basename(pkl_path)} ===")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"Total frames: {len(data)}")
    for i, frame in enumerate(data[:max_frames]):
        print(f"\n--- Frame {i} ---")
        for k, v in frame.items():
            print(f"{k}: type={type(v)}, ", end="")
            if k == "rgb_hist_jpg":
                print(f"{k}: {len(v)} frames, each type={type(v[0])}, size={[len(v[j]) if isinstance(v[j], (bytes, bytearray)) else v[j] for j in range(len(v))]}")
                if show_img and isinstance(v[0], (bytes, bytearray)) and len(v[0]) > 0:
                    imgs = []
                    for _, img_bytes in enumerate(v):
                        arr = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            imgs.append(img)
                    if False:
                        h = min(im.shape[0] for im in imgs)
                        resized = [cv2.resize(im, (int(im.shape[1]*h/im.shape[0]), h)) for im in imgs]
                        concat = np.hstack(resized)
                        cv2.imshow("rgb_hist_jpg (history)", concat)
                        cv2.waitKey(3000)
                        cv2.destroyAllWindows()
                    elif isinstance(v, np.ndarray):
                        print(f"{k}: ndarray shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"{k}: {v}")
                else:
                    print(v)
                    breakpoint()
        print("===")


if __name__ == "__main__":
    for fname in os.listdir(OUT_DIR):
        if fname.endswith(".pkl"):
            inspect_pkl(os.path.join(OUT_DIR, fname), show_img=False, max_frames=2)