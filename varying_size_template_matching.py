import cv2
import numpy as np

image_path = "template_images/image.png"
template_path = "template_images/mouse.png"

original_image = cv2.imread(image_path)
template = cv2.imread(template_path)

if original_image is None or template is None:
    raise FileNotFoundError("Could not find image or template. Check your paths!")

orig_height, orig_width = original_image.shape[:2]
new_width, new_height = 800, 1000

scale_x = new_width / orig_width
scale_y = new_height / orig_height

image = cv2.resize(original_image, (new_width, new_height))

template_scale = (scale_x + scale_y) / 2
scale_factor = 0.7
template_new_width = max(1, int(template.shape[1] * template_scale * scale_factor))
template_new_height = max(1, int(template.shape[0] * template_scale * scale_factor))

template = cv2.resize(template, (template_new_width, template_new_height))

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

template_size = template_gray.shape[::-1]  # (width, height)


def cv2_template_matching(image, template):
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        raise ValueError("Error: Template is larger than the main image!")
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_loc, max_val


def my_template_matching(image, template):
    prev_correlation = -1.0
    best_location = (0, 0)
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        return (0, 0), -1.0

    template_height, template_width = template.shape
    template = template.astype(np.float32)
    image = image.astype(np.float32)

    Eg2 = np.mean(template)
    diff_g2 = template - Eg2
    template_var = np.sqrt(np.sum(diff_g2 ** 2))

    for i in range(image.shape[1] - template.shape[1] + 1):
        for j in range(image.shape[0] - template.shape[0] + 1):
            g1 = image[j:j + template_height, i:i + template_width]
            Eg1 = np.mean(g1)
            g1_diff = g1 - Eg1

            numerator = np.sum(g1_diff * diff_g2)
            denominator = np.sqrt(np.sum(g1_diff ** 2)) * template_var

            correlation = numerator / denominator if denominator != 0 else 0
            if correlation > prev_correlation:
                prev_correlation = correlation
                best_location = (i, j)

    return best_location, prev_correlation


def template_matching_varying_template_size(image, template, min_scale=0.5, max_scale=1.5, step=0.1):
    prev_correlation = -1.0
    prev_best_location = (0, 0)
    best_scale = 1.0

    scales = np.arange(max_scale, min_scale - 1e-9, -step)

    for scale in scales:
        print(scale)
        resized_w = int(image.shape[1] * scale)
        resized_h = int(image.shape[0] * scale)

        if resized_w < template.shape[1] or resized_h < template.shape[0]:
            continue

        image_resized = cv2.resize(image, (resized_w, resized_h))

        best_location, best_correlation = my_template_matching(image_resized, template)

        if best_correlation > prev_correlation:
            best_scale = scale
            prev_correlation = best_correlation
            prev_best_location = best_location

    actual_x = int(prev_best_location[0] / best_scale) # scale it to actual image size
    actual_y = int(prev_best_location[1] / best_scale)

    return (actual_x, actual_y), prev_correlation, best_scale


def display_correlation(image, template_size, top_left, best_scale=1.0):
    box_width = int(template_size[0] / best_scale) # scale it to actual image size
    box_height = int(template_size[1] / best_scale)
    
    bottom_right = (top_left[0] + box_width, top_left[1] + box_height)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imshow("Detected Template", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    top_left, correlation, best_scale = template_matching_varying_template_size(
        grayscale_image, 
        template_gray, 
        min_scale=0.5, 
        max_scale=1.5, 
        step=0.1
    )
    
    print(f"Best Scale: {best_scale:.2f}x | Confidence: {correlation:.4f} | Coordinates: {top_left}")
    display_correlation(image, template_size, top_left, best_scale)
