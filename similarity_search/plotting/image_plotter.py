import numpy as np
import matplotlib.pyplot as plt

class ImagePlotter:
    @staticmethod
    def plot_fetched_images(image_data, n_cols=3, figsize=8, show_info=None, *args, **kwargs):
        show_info = show_info or []
        font_size = kwargs.get("font_size", 5)

        n_images = len(image_data)
        if n_images == 0:
            print("Warning: No images to display.")
            return
        
        n_rows = int(np.ceil(n_images / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize, figsize * n_rows / n_cols))

        if n_rows == 1:
            axes = np.array([axes])
        if n_cols == 1:
            axes = np.array([[ax] for ax in axes])

        missing_keys_warned = set()

        for idx, ax in enumerate(axes.flatten()):
            if idx < n_images:
                img_data = image_data[idx]
                if "image" not in img_data:
                    print(f"Warning: Image missing for index {idx}. Skipping.")
                    ax.axis('off')
                    continue

                img = img_data["image"]
                max_dim = max(img.shape[:2])
                square_img = np.ones((max_dim, max_dim, 3), dtype=np.uint8) * 255
                y_offset = (max_dim - img.shape[0]) // 2
                x_offset = (max_dim - img.shape[1]) // 2
                square_img[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img
                
                ax.imshow(square_img)
                ax.axis('off')

                info_text_list = []
                for key in show_info:
                    if key in img_data:
                        info_text_list.append(f"{key}: {img_data[key]}")
                    elif key not in missing_keys_warned:
                        print(f"Warning: Key '{key}' not found in image data.")
                        missing_keys_warned.add(key)

                if info_text_list:
                    info_text = "\n".join(info_text_list)
                    ax.text(
                        max_dim * 0.02, max_dim * 0.02, info_text, 
                        fontsize=font_size, color='red', backgroundcolor='none', 
                        va='top', ha='left', alpha=0.7
                    )
            else:
                ax.axis('off')

        plt.tight_layout()
        plt.show()
