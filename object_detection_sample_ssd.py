#!/usr/bin/env python
from __future__ import print_function
import sys
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore

# -------------------------GUI-------------------------------
import tkinter as tk
from tkinter import font as tkFont
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import random
from datetime import datetime

root = tk.Tk()
root.title('Output')
root.geometry('800x710+290-20')

# Logo--------------------------
image_0 = Image.open("open_vino.jpg")
logor = image_0.resize((400, 120))
logor2 = ImageTk.PhotoImage(logor)
logo = tk.Label(root, image=logor2)
logo.pack()
# ------------------------------

canvas = tk.Canvas(root, height=450, width=585)
canvas.pack()

path = tk.StringVar()


# resize the image
def resize(image):
    w, h = image.size
    mlength = max(w, h)  # The maximum length
    mulw = 585 / mlength  # Get the parameter
    mulh = 470 / mlength
    w1 = int(w * mulw)  # get the new height and length
    h1 = int(h * mulh)
    return image.resize((w1, h1))


# show image function
def show_image(path):
    global img  # set the global variables
    image = Image.open(path)  # open picture
    re_image = resize(image)  # resize the image
    img = ImageTk.PhotoImage(re_image)  # to load the pictures
    canvas.create_image(295, 230, anchor='center', image=img)  # Original 200,200
    # print(re_image.size)


# Browse the picture
def openpicture():
    global fileindex, fatherpath, files, file_num
    # debug---------------------------------------------------
    # print("testing")
    # --------------------------------------------------------
    filepath = filedialog.askopenfilename()
    fatherpath = os.path.dirname(filepath)  # get the previous parent folder
    filename = os.path.basename(filepath)  # get the file
    files = os.listdir(fatherpath)  # return the list in the var files
    file_num = len(files)
    fileindex = files.index(filename)  # get the index
    show_image(filepath)


def previous():
    global fileindex, fatherpath, files, file_num
    fileindex -= 1
    if fileindex == -1:
        fileindex = file_num - 1
    filepath1 = os.path.join(fatherpath, files[fileindex])
    # print(filepath1)
    show_image(filepath1)


def scan():
    global fileindex, fatherpath, files, file_num
    # Debug-------------------------------------------
    # print(fatherpath)
    # print(files[fileindex])
    # ------------------------------------------------
    filepath3 = fatherpath + "/" + files[fileindex]
    main(filepath3)
    # Alternative method --------------------------
    # main(fatherpath,files[fileindex])
    # ----------------------------------------------
    show_image(r"C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\samples\python\prototype\out.bmp")


def back():
    global fileindex, fatherpath, files, file_num
    fileindex += 1
    if fileindex == file_num:
        fileindex = 0
    filepath2 = os.path.join(fatherpath, files[fileindex])
    show_image(filepath2)


# ------------------------------------------------------------------

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Path to image file.",
                      required=True, type=str, nargs="+")
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "Absolute path to a shared library with the kernels implementations.",
                      type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; "
                           "CPU, GPU, FPGA or MYRIAD is acceptable. "
                           "Sample will look for a suitable plugin for device specified (CPU by default)",
                      default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Labels mapping file", default=None, type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)

    return parser


def main(input_1):
    # Modify -------------------------------------------------
    # print(input_1) for debugging
    index_var = 0  # set the initial value of the dictionary (for listing)
    # Time ----------------------------------------------------
    now = datetime.now()
    # ---------------------------------------------------------
    logFile = open("LogFile.txt", 'a')  # writing logFile.txt
    name_of_picture = input_1
    name_of_picture1 = name_of_picture.split("/")
    name_of_picture2 = name_of_picture1[10]
    logFile.write(name_of_picture2 + "\n")
    # ---------------------------------------------------------
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info("Loading Inference Engine")
    ie = IECore()
    # --------------------------- 1. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)
    # -----------------------------------------------------------------------------------------------------

    # ------------- 2. Load Plugin for inference engine and extensions library if specified --------------
    log.info("Device info:")
    versions = ie.get_versions(args.device)
    print("{}{}".format(" " * 8, args.device))
    print("{}MKLDNNPlugin version ......... {}.{}".format(" " * 8, versions[args.device].major,
                                                          versions[args.device].minor))
    print("{}Build ........... {}".format(" " * 8, versions[args.device].build_number))

    if args.cpu_extension and "CPU" in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
        log.info("CPU extension loaded: {}".format(args.cpu_extension))

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    # -----------------------------------------------------------------------------------------------------

    # --------------------------- 3. Read and preprocess input --------------------------------------------

    print("inputs number: " + str(len(net.input_info.keys())))

    for input_key in net.input_info:
        print("input shape: " + str(net.input_info[input_key].input_data.shape))
        print("input key: " + input_key)
        if len(net.input_info[input_key].input_data.layout) == 4:
            n, c, h, w = net.input_info[input_key].input_data.shape

    images = np.ndarray(shape=(n, c, h, w))
    images_hw = []
    for i in range(n):
        # image = cv2.imread(args.input[i]) # original code
        image = cv2.imread(input_1)
        # testing
        # -------------------------------------------
        # print(type(image))
        # ------------------------------------------
        ih, iw = image.shape[:-1]
        images_hw.append((ih, iw))
        log.info("File was added: ")
        # log.info("        {}".format(args.input[i]))
        log.info("        {}".format(input_1))
        if (ih, iw) != (h, w):
            # log.warning("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
            log.warning("Image {} is resized from {} to {}".format(input_1, image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[i] = image

    # -----------------------------------------------------------------------------------------------------

    # --------------------------- 4. Configure input & output ---------------------------------------------
    # --------------------------- Prepare input blobs -----------------------------------------------------
    log.info("Preparing input blobs")
    assert (len(net.input_info.keys()) == 1 or len(
        net.input_info.keys()) == 2), "Sample supports topologies only with 1 or 2 inputs"
    out_blob = next(iter(net.outputs))
    input_name, input_info_name = "", ""

    for input_key in net.input_info:
        if len(net.input_info[input_key].layout) == 4:
            input_name = input_key
            log.info("Batch size is {}".format(net.batch_size))
            net.input_info[input_key].precision = 'U8'
        elif len(net.input_info[input_key].layout) == 2:
            input_info_name = input_key
            net.input_info[input_key].precision = 'FP32'
            if net.input_info[input_key].input_data.shape[1] != 3 and net.input_info[input_key].input_data.shape[
                1] != 6 or \
                    net.input_info[input_key].input_data.shape[0] != 1:
                log.error('Invalid input info. Should be 3 or 6 values length.')

    data = {}
    data[input_name] = images

    if input_info_name != "":
        infos = np.ndarray(shape=(n, c), dtype=float)
        for i in range(n):
            infos[i, 0] = h
            infos[i, 1] = w
            infos[i, 2] = 1.0
        data[input_info_name] = infos

    # --------------------------- Prepare output blobs ----------------------------------------------------
    log.info('Preparing output blobs')

    output_name, output_info = "", net.outputs[next(iter(net.outputs.keys()))]
    for output_key in net.outputs:
        if net.layers[output_key].type == "DetectionOutput":
            output_name, output_info = output_key, net.outputs[output_key]

    if output_name == "":
        log.error("Can't find a DetectionOutput layer in the topology")

    output_dims = output_info.shape
    if len(output_dims) != 4:
        log.error("Incorrect output dimensions for SSD model")
    max_proposal_count, object_size = output_dims[2], output_dims[3]

    if object_size != 7:
        log.error("Output item should have 7 as a last dimension")

    output_info.precision = "FP32"
    # -----------------------------------------------------------------------------------------------------

    # --------------------------- Performing inference ----------------------------------------------------
    log.info("Loading model to the device")
    exec_net = ie.load_network(network=net, device_name=args.device)
    log.info("Creating infer request and starting inference")
    res = exec_net.infer(inputs=data)
    # -----------------------------------------------------------------------------------------------------
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    # --------------------------- Read and postprocess output ---------------------------------------------
    log.info("Processing output blobs")
    res = res[out_blob]
    boxes, classes = {}, {}
    data = res[0][0]
    probability = {}
    for number, proposal in enumerate(data):
        if proposal[2] > 0:
            imid = np.int(proposal[0])
            ih, iw = images_hw[imid]
            label = np.int(proposal[1])
            confidence = proposal[2]  # probability
            xmin = np.int(iw * proposal[3])
            ymin = np.int(ih * proposal[4])
            xmax = np.int(iw * proposal[5])
            ymax = np.int(ih * proposal[6])
            print("[{},{}] element, prob = {:.6}    ({},{})-({},{}) batch id : {}" \
                  .format(number, label, confidence, xmin, ymin, xmax, ymax, imid), end="")
            if proposal[2] > 0.5:
                print(" WILL BE PRINTED!")

                if not imid in boxes.keys():
                    boxes[imid] = []
                boxes[imid].append([xmin, ymin, xmax, ymax])
                if not imid in classes.keys():
                    classes[imid] = []
                classes[imid].append(label)
                if not imid in probability.keys():
                    probability[imid] = []
                probability[imid].append(confidence)
            else:
                print()

    for imid in classes:
        # tmp_image = cv2.imread(args.input[imid]) source code
        tmp_image = cv2.imread(input_1)
        for box in boxes[imid]:
            r = random.randint(0, 256)
            b = random.randint(0, 256)
            g = random.randint(0, 256)
            cv2.rectangle(tmp_image, (box[0], box[1]), (box[2], box[3]), (b, g, r), 2)
            class_id = int(box[1])  # coordinate
            # color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
            # Testing--------------------------------------------
            # Set up the variable -------------------------------
            index_num = classes[0][index_var]  # raw id number
            probab = probability[0][index_var]
            probab_1 = round(probab * 100, 2)  # probabilty
            index_var += 1
            id_num = (5 * (index_num - 1)) + 2  # index number
            # --------------------------------------------------
            # Message string-----------------------------------
            time_string = now.strftime("%d/%m/%Y %H:%M:%S")
            message_1 = labels_map[id_num] + " | Probability : " + str(probab_1) + "%"
            message_2 = time_string + " : " + message_1
            print(message_1)  # print on Cmd
            logFile.write(message_2 + "\n")
            # -------------------------------------------------------
            # sample_color = (255, 255, 0)  # set the canyon color
            # ------------------------------------------------------
            det_label = labels_map[id_num] if labels_map else str(class_id)
            # cv2.putText(tmp_image,det_label + ' ' + str(round(box[2] * 100, 1)) + ' %', (xmin, ymin - 7),
            # cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            # cv2.putText(tmp_image, det_label, ((box[0]+box[2])//2, (box[1]+box[3])//2 ),
            # cv2.FONT_HERSHEY_COMPLEX, 0.7, sample_color, 3)
            cv2.putText(tmp_image, det_label, (box[0], box[1] + 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (b, g, r), 3)

        # cv2.imshow("img",tmp_image) # pop up
        kk = "-" * 60
        logFile.write(kk + "\n")
        logFile.close()
        cv2.imwrite("out.bmp", tmp_image)
        # Testing -------------------
        # print(os.getcwd())
        # ----------------------------
        log.info("Image out.bmp created!")
    # -----------------------------------------------------------------------------------------------------

    log.info("Execution successful\n")
    log.info(
        "This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool")


if __name__ == '__main__':
    # GUI
    # button
    helv35 = tkFont.Font(family="Helvetica", size=12, weight=tkFont.BOLD)
    # icon of the button
    image_1 = Image.open("vector_la2.png.jpg")
    left_arrow = image_1.resize((100, 100))

    image_2 = Image.open("vector_ra2.jpg")
    right_arrow = image_2.resize((100, 100))

    image_3 = Image.open("scan.jpg")
    scan_1 = image_3.resize((285, 110))

    image_4 = Image.open("select_a_picture.jpg")
    select_1 = image_4.resize((285, 110))

    scan_2 = ImageTk.PhotoImage(scan_1)
    select_2 = ImageTk.PhotoImage(select_1)
    Left_arrow = ImageTk.PhotoImage(left_arrow)
    Right_arrow = ImageTk.PhotoImage(right_arrow)

    # b1 = tk.Button(root, text='Previous Picture', command=previous).pack(side='left')
    b1 = tk.Button(root, image=Left_arrow, command=previous).pack(side='left')
    # b2 = tk.Button(root, text='Next Picture', command=back).pack(side='right')
    b2 = tk.Button(root, image=Right_arrow, command=back).pack(side='right')

    # b = tk.Button(root, text='Select a picture',font = helv35, command=openpicture,height = 30, width = 30)
    b = tk.Button(root, image=select_2, command=openpicture)
    b.pack(expand=True, side="left")

    # c = tk.Button(root, text="Scan", command=scan,font = helv35,height=30,width=30)
    c = tk.Button(root, image=scan_2, command=scan)
    c.pack(expand=True, side="right")
    root['bg'] = '#0059b3'

    tk.mainloop()
    # sys.exit(main() or 0) source code

# Todo
# fix the font size problem (/)
# put some fancy buttons on it (/)
# the colors (/)
# the window size (/)
# the button coordinate (/)
# the probability (log file)
