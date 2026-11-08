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
template_new_width = int(template.shape[1] * template_scale)
template_new_height = int(template.shape[0] * template_scale)

template_new_width = max(1, template_new_width)
template_new_height = max(1, template_new_height)

template = cv2.resize(template, (template_new_width, template_new_height))

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

template_size = template_gray.shape[::-1]  # (width, height)



def find_cross_correlation(image, template):
    # Safety Check: Check if template is larger than image
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        raise ValueError("Error: After scaling, the template is larger than the main image!")
        
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc



def my_cross_correlation(image, template):
    prev_correlation = -1
    best_location = (0, 0)
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
            raise ValueError("Error: After scaling, the template is larger than the main image!")

    template_height, template_width = template.shape
    template = template.astype(np.float32)
    image = image.astype(np.float32)

    Eg2 = np.mean(template)
    for i in range(image.shape[1] - template.shape[1] + 1):
        for j in range(image.shape[0] - template.shape[0] + 1):
            Eg1 = np.mean(image[j:j + template_height, i:i + template_width])
            numerator = np.sum((image[j:j + template_height, i:i + template_width] - Eg1) * (template - Eg2))
            denominator = np.sqrt((np.sum((image[j:j + template_height, i:i+ template_width] - Eg1) ** 2))) * np.sqrt(np.sum((template - Eg2) ** 2))

            correlation = numerator/denominator if denominator != 0 else 0
            if correlation > prev_correlation:
                prev_correlation = correlation
                best_location = (i, j)
    return best_location

def display_correlation(image, template, top_left):
    bottom_right = (top_left[0] + template_size[0], top_left[1] + template_size[1])
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imshow("Detected Template", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #top_left = find_cross_correlation(grayscale_image, template_gray)
    top_left = my_cross_correlation(grayscale_image, template_gray)
    display_correlation(image, template, top_left)
