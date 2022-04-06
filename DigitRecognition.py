import os.path
from keras.models import load_model
import logging
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from functools import partial

from PIL import Image, ImageTk


def predict(im):
    im_resized = cv2.resize(255 - im, (18, 18))
    padded_im = np.pad(im_resized, ((5, 5), (5, 5)), "constant", constant_values=0)
    im_final = padded_im.reshape(1, 28, 28, 1)
    ans = model.predict(im_final)
    print(ans)
    numclass = np.where(max(ans))
    return numclass[0][0]


def markDetectedNumbers(detectedNumStat, outpuIM):
    im_clr = cv2.cvtColor(outpuIM, cv2.COLOR_GRAY2BGR)
    for location in detectedNumStat["locations"]:
        cv2.putText(im_clr, "+", tuple(location), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 1)
    return im_clr


def predictNumbers(centroids, number_indics, stats, labels, outpuIM, mask):
    detectedNumStat = {"numbers": [], "locations": []}
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened_im = cv2.dilate(cv2.bitwise_not(outpuIM), kernel, iterations=2)

    contours, hierarchy = cv2.findContours(opened_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for index, cnt in enumerate(contours):
        for i in number_indics:
            retval = cv2.pointPolygonTest(cnt, (int(centroids[i + 1][0]), int(centroids[i + 1][1])), False)
            if retval in (0, 1):
                x = stats[i + 1, cv2.CC_STAT_LEFT]
                y = stats[i + 1, cv2.CC_STAT_TOP]
                w = stats[i + 1, cv2.CC_STAT_WIDTH]
                h = stats[i + 1, cv2.CC_STAT_HEIGHT]
                mask[labels == i + 1] = 0
                rect_mask = mask[y: y + h, x: x + w]
                num_class = predict(rect_mask)

                cv2.putText(outpuIM, str(num_class), (int(centroids[i + 1][0]), int(centroids[i + 1][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0),
                            2)
                user_logger.warning(
                    f"An integer has been detected at coordinate: {(int(centroids[i + 1][0]), int(centroids[i + 1][1]))}. That number is: {num_class}.")

                detectedNumStat["numbers"].append(int(num_class))
                detectedNumStat["locations"].append((int(centroids[i + 1][0]), int(centroids[i + 1][1])))
                mask[:, :] = 255

    outputIM = markDetectedNumbers(detectedNumStat, outpuIM)
    return outputIM


def threshCalculator(arr, n):
    INT_MIN, INT_MAX = float('-inf'), float('inf')

    # Find maximum and minimum in arr[]
    maxVal, minVal = arr[0], arr[0]
    for i in range(1, n):
        maxVal = max(maxVal, arr[i])
        minVal = min(minVal, arr[i])

    # Arrays to store maximum and minimum
    # values in n-1 buckets of differences.
    maxBucket = [INT_MIN] * (n - 1)
    minBucket = [INT_MAX] * (n - 1)

    # Expected gap for every bucket.
    delta = (maxVal - minVal) // (n - 1)

    # Traversing through array elements and filling in appropriate bucket if bucket is empty.
    # Else updating bucket values.
    for i in range(0, n):
        if arr[i] == maxVal or arr[i] == minVal:
            continue

        # Finding index of bucket.
        index = (arr[i] - minVal) // delta

        # Filling/Updating maximum value of bucket
        if maxBucket[index] == INT_MIN:
            maxBucket[index] = arr[i]
        else:
            maxBucket[index] = max(maxBucket[index], arr[i])

        # Filling/Updating minimum value of bucket
        if minBucket[index] == INT_MAX:
            minBucket[index] = arr[i]
        else:
            minBucket[index] = min(minBucket[index], arr[i])

    # Finding maximum difference between maximum value of previous bucket minus minimum of current bucket.
    prev_val, max_gap = minVal, 0

    for i in range(0, n - 1):
        if minBucket[i] == INT_MAX:
            continue
        max_gap = max(max_gap,
                      minBucket[i] - prev_val)
        prev_val = maxBucket[i]

    max_gap = max(max_gap, maxVal - prev_val)

    return max_gap


def shadow_removal(image):
    rgb_planes = cv2.split(image)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        print(f"This is difference for shadow{bg_img}")
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm


def start(input_image_path):
    # Using Connected component Analysis.
    im_BGR = cv2.imread(input_image_path)
    # im_BGR = shadow_removal(im_BGR)
    blurred_im = cv2.medianBlur(im_BGR, 5)
    height, width = blurred_im.shape[0:2]
    ratio_height = height / width
    im_BGR = cv2.resize(im_BGR, (400, int(ratio_height * 400)))
    im_gray = cv2.cvtColor(im_BGR, cv2.COLOR_BGR2GRAY)
    im_binary = cv2.threshold(im_gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    numlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(im_binary)
    mask = np.ones_like(im_gray) * 255
    outpuIM = np.ones_like(im_gray) * 255
    areas = [area[-1] for area in stats[1:]]
    areas = np.array(areas)
    number_indcs = []

    # This is for calculating a threshold to differentiate between numbers and shapes
    thresh_area = threshCalculator(areas, len(areas))
    for i, area in enumerate(areas):
        if area <= int(thresh_area) and area > 18:  # when the component is a number
            number_indcs.append(i)

        elif area >= int(thresh_area):  # when the component is a shape
            outpuIM[labels == i + 1] = 0

    outputIM = predictNumbers(centroids, number_indcs, stats, labels, outpuIM, mask)
    return outputIM


def do_upload_images(input_frame, output_frame):
    input_image_path = filedialog.askopenfilename(initialdir="/", title="Select an image for Digit Recognition tool!")
    if os.path.exists(input_image_path):
        out_name = input_image_path.split("/")[-1]
        input_im = Image.open(input_image_path)
        resized_img = input_im.resize((int((canvas_width / 2) - 10), int(canvas_height - 10)))
        input_im = ImageTk.PhotoImage(resized_img)
        img_input_lbl.config(image=input_im)
        img_input_lbl.place(relx=0.5, rely=0.5, anchor="center")
        output_IM = start(input_image_path)
        resized_img = cv2.resize(output_IM, (int((canvas_width / 2) - 10), int(canvas_height - 10)))
        im = Image.fromarray(resized_img)
        imgtk = ImageTk.PhotoImage(image=im)
        img_out_lbl.config(image=imgtk)
        img_out_lbl.place(relx=0.5, rely=0.5, anchor="center")
        cv2.imwrite("output/" + "result_" + f"{out_name}", output_IM[:, :, ::-1])

        root.mainloop()


def reset():
    img_input_lbl.configure(image="")
    img_out_lbl.configure(image="")


if __name__ == "__main__":
    """ Load the trained model for testing the script"""
    model = load_model("model/model.h5")

    """creating a logger to save the location and detected number """
    user_logger = logging.getLogger("user_logger")
    user_logger.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s:%(message)s')
    handler = logging.FileHandler("output/user_logger.log", mode="w")
    handler.setFormatter(formatter)
    user_logger.addHandler(handler)

    # Setting a GUI for uploading the image
    root = tk.Tk()
    root.title("Digit Recognition Tool")
    window_width = root.winfo_screenwidth()
    window_height = root.winfo_screenheight()
    canvas_width = window_width
    canvas_height = window_height * 0.75
    root.geometry("%dx%d+%d+%d" % (window_width,
                                   window_height,
                                   0,
                                   0))
    Frame_holder = tk.Frame(root, width=canvas_width, height=canvas_height, bg="white")
    Frame_holder.pack(side="top", fill="both")
    input_frame = tk.Frame(Frame_holder, width=(canvas_width / 2) - 10, height=canvas_height - 10, bg="white")
    input_frame.config(borderwidth=6, relief="groove")
    # input_frame.place(relx=0.25, rely=0.42, anchor="center")
    input_frame.pack(side="left", fill="both")

    output_frame = tk.Frame(Frame_holder, width=(canvas_width / 2) - 10, height=canvas_height - 10, bg="white")
    output_frame.config(borderwidth=6, relief="groove")
    # output_frame.place(relx=0.75, rely=0.42, anchor="center")
    output_frame.pack(side="right", fill="both")

    button_frame = tk.Frame(root, width=window_width - 5, height=window_height * 0.14, bg="white")
    button_frame.config(borderwidth=6, relief="groove")
    button_frame.place(relx=0.5, rely=0.92, anchor="center")

    # upload_im = tk.PhotoImage(file="images/upload_image.png")

    upload_images = partial(do_upload_images, input_frame, output_frame)

    upload_btn = tk.Button(button_frame, width=13, height=4, text="Upload Photo", anchor="center",
                           command=upload_images,
                           relief="raised")
    upload_btn.place(relx=0.25, rely=0.5, anchor="center")

    reset_btn = tk.Button(button_frame, width=13, height=4, text="RESET", anchor="center", command=reset,
                          relief="raised")
    reset_btn.place(relx=0.75, rely=0.5, anchor="center")

    img_in_lbl = tk.Label(input_frame, text="Input image will show up here!")
    img_in_lbl.place(relx=0.5, rely=0.5, anchor="center")

    img_input_lbl = tk.Label(input_frame)
    img_input_lbl.place(relx=0.5, rely=0.5, anchor="center")

    img_o_lbl = tk.Label(output_frame, text="Output image will show up here!")
    img_o_lbl.place(relx=0.5, rely=0.5, anchor="center")

    img_out_lbl = tk.Label(output_frame)
    img_out_lbl.place(relx=0.5, rely=0.5, anchor="center")
    root.mainloop()
